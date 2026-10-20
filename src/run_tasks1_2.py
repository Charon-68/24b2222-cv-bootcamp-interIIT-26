"""Task 1 & Task 2 Execution Script: Baselines & Training Dynamics.

Runs 60-epoch training for:
1. Clean Reference Model -> predictions/t1_clean_ref.csv
2. Noisy CE Model -> predictions/t1_noisy_ce.csv, predictions/t2_dynamics.csv, predictions/t2_early_stop.csv

Per-Epoch Dynamics tracked for Noisy CE:
- train_acc_correct: accuracy on correctly labeled training subset (26,891 samples)
- train_acc_mislabeled: accuracy on mislabeled training subset (18,109 samples)
- val_acc_noisy: validation accuracy on val_idx using noisy_label (ONLY metric allowed for early stopping selection)
- val_acc_clean: validation accuracy on val_idx using clean_label (diagnostic metric only)
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Ensure project root is in sys.path
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from check_format import check_dynamics, check_pred_file
from src.config import get_default_config
from src.model_factory import create_model
from src.plotting import plot_dynamics
from src.utils import (
    export_dynamics_csv,
    export_predictions_csv,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip_tf = T.RandomHorizontalFlip(p=0.5)


def batch_augment(imgs_batch):
    return flip_tf(crop_tf(imgs_batch))


class FastTensorDataset(Dataset):
    """Memory-cached fast dataset."""
    def __init__(self, images_tensor, labels_array, indices, clean_labels_array=None):
        super().__init__()
        self.indices = indices
        self.labels = labels_array[indices]
        self.images = images_tensor[indices]
        self.clean_labels = clean_labels_array[indices] if clean_labels_array is not None else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        global_idx = int(self.indices[idx])
        target = int(self.labels[idx])
        if self.clean_labels is not None:
            clean_t = int(self.clean_labels[idx])
            return global_idx, img, target, clean_t
        return global_idx, img, target


def run_clean_reference_experiment(config):
    """Train Task 1 Clean Reference ResNet-18 model for 60 epochs."""
    print("\n" + "=" * 70, flush=True)
    print("      TASK 1: TRAINING CLEAN REFERENCE MODEL (60 EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    set_seed(config.seed)
    device = get_device(config.device)

    labels_df = pd.read_csv(os.path.join(config.data_dir, "labels.csv"))
    all_imgs = np.load(os.path.join(config.data_dir, "train_images.npy"))
    test_imgs = np.load(os.path.join(config.data_dir, "test_images.npy"))

    with open(os.path.join(config.data_dir, "train_idx.txt"), encoding="utf-8") as f:
        train_indices = np.array([int(line.strip()) for line in f if line.strip()])
    with open(os.path.join(config.data_dir, "val_idx.txt"), encoding="utf-8") as f:
        val_indices = np.array([int(line.strip()) for line in f if line.strip()])

    train_clean_labels = labels_df["clean_label"].values[train_indices]
    val_clean_labels = labels_df["clean_label"].values[val_indices]

    mean = torch.tensor(config.mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor(config.std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    raw_tr = (torch.from_numpy(all_imgs[train_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    tr_images_tensor = ((raw_tr - mean) / std).contiguous()

    raw_val = (torch.from_numpy(all_imgs[val_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    val_images_tensor = ((raw_val - mean) / std).contiguous()

    raw_test = (torch.from_numpy(test_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    test_images_tensor = ((raw_test - mean) / std).contiguous()

    tr_clean_labels_t = torch.tensor(train_clean_labels, dtype=torch.long, device=device)
    val_clean_labels_t = torch.tensor(val_clean_labels, dtype=torch.long, device=device)

    model = create_model(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    start_time = time.time()
    best_val_acc = 0.0

    n_train = len(train_indices)
    n_val = len(val_indices)

    for epoch in range(1, config.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0

        # Shuffle indices on GPU
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, config.batch_size):
            b_idx = perm[i : i + config.batch_size]
            imgs = batch_augment(tr_images_tensor[b_idx])
            targets = tr_clean_labels_t[b_idx]

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * len(b_idx)
            ep_correct += (outputs.argmax(dim=1) == targets).sum().item()

        scheduler.step()
        train_acc = (ep_correct / n_train) * 100.0
        avg_train_loss = ep_loss / n_train

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for i in range(0, n_val, config.batch_size):
                imgs = val_images_tensor[i : i + config.batch_size]
                targets = val_clean_labels_t[i : i + config.batch_size]
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        val_acc = (val_correct / n_val) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc},
                os.path.join(config.checkpoint_dir, "t1_clean_ref_best.pth"),
            )

        if epoch % 10 == 0 or epoch == 1 or epoch == config.epochs:
            print(f"Epoch [{epoch:02d}/{config.epochs:02d}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)", flush=True)

    total_time = time.time() - start_time
    save_checkpoint(
        {"epoch": config.epochs, "model_state": model.state_dict(), "val_acc": val_acc},
        os.path.join(config.checkpoint_dir, "t1_clean_ref_final.pth"),
    )

    model.eval()
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(test_images_tensor), config.batch_size):
            batch_imgs = test_images_tensor[i : i + config.batch_size].to(device)
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)

    test_preds = np.array(test_preds)
    clean_ref_csv = os.path.join(config.output_dir, "t1_clean_ref.csv")
    export_predictions_csv(test_preds, clean_ref_csv)
    check_pred_file(clean_ref_csv)

    print(f"Clean Reference Model finished in {total_time:.1f}s. Final Val Acc: {val_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%", flush=True)
    return best_val_acc


def run_noisy_ce_and_task2_experiment(config):
    """Train Task 1 & 2 Noisy CE ResNet-18 model for 60 epochs with full dynamics logging and per-epoch checkpoints."""
    print("\n" + "=" * 70, flush=True)
    print("   TASK 1 & TASK 2: TRAINING NOISY CE MODEL & DYNAMICS (60 EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    set_seed(config.seed)
    device = get_device(config.device)

    labels_df = pd.read_csv(os.path.join(config.data_dir, "labels.csv"))
    all_imgs = np.load(os.path.join(config.data_dir, "train_images.npy"))
    test_imgs = np.load(os.path.join(config.data_dir, "test_images.npy"))

    with open(os.path.join(config.data_dir, "train_idx.txt"), encoding="utf-8") as f:
        train_indices = np.array([int(line.strip()) for line in f if line.strip()])
    with open(os.path.join(config.data_dir, "val_idx.txt"), encoding="utf-8") as f:
        val_indices = np.array([int(line.strip()) for line in f if line.strip()])

    train_noisy_labels = labels_df["noisy_label"].values[train_indices]
    train_clean_labels = labels_df["clean_label"].values[train_indices]

    val_noisy_labels = labels_df["noisy_label"].values[val_indices]
    val_clean_labels = labels_df["clean_label"].values[val_indices]

    mean = torch.tensor(config.mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor(config.std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    raw_tr = (torch.from_numpy(all_imgs[train_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    tr_images_tensor = ((raw_tr - mean) / std).contiguous()

    raw_val = (torch.from_numpy(all_imgs[val_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    val_images_tensor = ((raw_val - mean) / std).contiguous()

    raw_test = (torch.from_numpy(test_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0).to(device)
    test_images_tensor = ((raw_test - mean) / std).contiguous()

    tr_noisy_labels_t = torch.tensor(train_noisy_labels, dtype=torch.long, device=device)
    tr_clean_labels_t = torch.tensor(train_clean_labels, dtype=torch.long, device=device)

    val_noisy_labels_t = torch.tensor(val_noisy_labels, dtype=torch.long, device=device)
    val_clean_labels_t = torch.tensor(val_clean_labels, dtype=torch.long, device=device)

    model = create_model(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    dynamics_rows = []
    epoch_checkpoints_dir = os.path.join(config.checkpoint_dir, "noisy_ce_epochs")
    os.makedirs(epoch_checkpoints_dir, exist_ok=True)

    best_val_acc_noisy = 0.0
    best_noisy_epoch = 1
    start_time = time.time()

    is_correct_mask_t = (tr_noisy_labels_t == tr_clean_labels_t)
    is_mislabeled_mask_t = ~is_correct_mask_t
    n_correct = is_correct_mask_t.sum().item()
    n_mislabeled = is_mislabeled_mask_t.sum().item()

    n_train = len(train_indices)
    n_val = len(val_indices)

    for epoch in range(1, config.epochs + 1):
        model.train()
        ep_loss = 0.0
        correct_sub_hits = 0
        mislabeled_sub_hits = 0

        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, config.batch_size):
            b_idx = perm[i : i + config.batch_size]
            imgs = batch_augment(tr_images_tensor[b_idx])
            targets = tr_noisy_labels_t[b_idx]
            cleans = tr_clean_labels_t[b_idx]

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * len(b_idx)
            preds = outputs.argmax(dim=1)
            match = (preds == targets)
            c_mask = (targets == cleans)
            m_mask = ~c_mask

            correct_sub_hits += (match & c_mask).sum().item()
            mislabeled_sub_hits += (match & m_mask).sum().item()

        scheduler.step()
        acc_correct = (correct_sub_hits / n_correct) * 100.0
        acc_mislabeled = (mislabeled_sub_hits / n_mislabeled) * 100.0
        avg_train_loss = ep_loss / n_train

        # Evaluate on Validation set
        model.eval()
        val_correct_noisy = 0
        val_correct_clean = 0
        with torch.no_grad():
            for i in range(0, n_val, config.batch_size):
                imgs = val_images_tensor[i : i + config.batch_size]
                targets_n = val_noisy_labels_t[i : i + config.batch_size]
                targets_c = val_clean_labels_t[i : i + config.batch_size]

                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct_noisy += (preds == targets_n).sum().item()
                val_correct_clean += (preds == targets_c).sum().item()

        val_acc_noisy = (val_correct_noisy / n_val) * 100.0
        val_acc_clean = (val_correct_clean / n_val) * 100.0

        ckpt_file = os.path.join(epoch_checkpoints_dir, f"noisy_ce_ep_{epoch:02d}.pth")
        save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "val_acc_noisy": val_acc_noisy}, ckpt_file)

        if val_acc_noisy > best_val_acc_noisy:
            best_val_acc_noisy = val_acc_noisy
            best_noisy_epoch = epoch
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(), "val_acc_noisy": val_acc_noisy},
                os.path.join(config.checkpoint_dir, "t1_noisy_ce_best.pth"),
            )

        dynamics_rows.append([epoch - 1, float(acc_correct), float(acc_mislabeled), float(val_acc_noisy), float(val_acc_clean)])

        if epoch % 10 == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"Epoch [{epoch:02d}/{config.epochs:02d}] | Train Loss: {avg_train_loss:.4f} | "
                f"Correct Sub Acc: {acc_correct:.2f}%, Mislabeled Sub Acc: {acc_mislabeled:.2f}% | "
                f"Val Acc (Noisy): {val_acc_noisy:.2f}%, (Clean): {val_acc_clean:.2f}%",
                flush=True,
            )

    total_time = time.time() - start_time
    save_checkpoint(
        {"epoch": config.epochs, "model_state": model.state_dict(), "val_acc_noisy": val_acc_noisy},
        os.path.join(config.checkpoint_dir, "t1_noisy_ce_final.pth"),
    )

    dynamics_csv = os.path.join(config.output_dir, "t2_dynamics.csv")
    export_dynamics_csv(dynamics_rows, dynamics_csv)
    check_dynamics(dynamics_csv)

    # Save predictions/t1_noisy_ce.csv from FINAL epoch (epoch 60)
    model.eval()
    final_preds = []
    with torch.no_grad():
        for i in range(0, len(test_images_tensor), config.batch_size):
            batch_imgs = test_images_tensor[i : i + config.batch_size].to(device)
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            final_preds.extend(preds)

    noisy_ce_csv = os.path.join(config.output_dir, "t1_noisy_ce.csv")
    export_predictions_csv(np.array(final_preds), noisy_ce_csv)
    check_pred_file(noisy_ce_csv)

    # Save predictions/t2_early_stop.csv from EARLY-STOPPING epoch (best noisy val acc)
    print(f"\n[Task 2] Loading checkpoint from Early Stopping Epoch {best_noisy_epoch} (Best Noisy Val Acc: {best_val_acc_noisy:.2f}%)...", flush=True)
    best_ckpt = load_checkpoint(os.path.join(epoch_checkpoints_dir, f"noisy_ce_ep_{best_noisy_epoch:02d}.pth"), map_location=device)
    early_stop_model = create_model(num_classes=config.num_classes).to(device)
    early_stop_model.load_state_dict(best_ckpt["model_state"])

    early_stop_preds = []
    with torch.no_grad():
        for i in range(0, len(test_images_tensor), config.batch_size):
            batch_imgs = test_images_tensor[i : i + config.batch_size].to(device)
            outputs = early_stop_model(batch_imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            early_stop_preds.extend(preds)

    early_stop_csv = os.path.join(config.output_dir, "t2_early_stop.csv")
    export_predictions_csv(np.array(early_stop_preds), early_stop_csv)
    check_pred_file(early_stop_csv)

    plot_path = os.path.join(config.output_dir, "t2_dynamics_plot.png")
    plot_dynamics(dynamics_csv, save_path=plot_path)

    df_dyn = pd.read_csv(dynamics_csv)
    acc_mislabeled_vals = df_dyn["train_acc_mislabeled"].values
    
    memorization_start_epoch = 1
    for ep, acc_m in enumerate(acc_mislabeled_vals, 1):
        if acc_m > 10.0:
            memorization_start_epoch = ep
            break

    return (
        val_acc_noisy,
        best_val_acc_noisy,
        best_noisy_epoch,
        memorization_start_epoch,
        total_time,
    )


def main():
    config = get_default_config()

    final_val_clean = run_clean_reference_experiment(config)

    (
        final_val_noisy,
        best_val_noisy,
        early_stop_epoch,
        memorization_start_epoch,
        noisy_train_time,
    ) = run_noisy_ce_and_task2_experiment(config)

    print("\n" + "=" * 70, flush=True)
    print("            SUMMARY OF TASK 1 & TASK 2 EXECUTION", flush=True)
    print("=" * 70, flush=True)
    print(f"t1_clean_ref.csv:    VALIDATED (10,000 predictions)", flush=True)
    print(f"t1_noisy_ce.csv:     VALIDATED (10,000 predictions)", flush=True)
    print(f"t2_dynamics.csv:     VALIDATED (60 epoch rows)", flush=True)
    print(f"t2_early_stop.csv:   VALIDATED (10,000 predictions from Epoch {early_stop_epoch})", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
