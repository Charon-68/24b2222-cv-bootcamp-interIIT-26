"""Task 5: Synthetic Noise Benchmarking Pipeline.

Evaluates:
1. Baseline Cross Entropy Model trained on synthetic_label -> predictions/t5_baseline_synth.csv
2. Best Task 4 Robust Model trained on synthetic_label -> predictions/t5_best_synth.csv

Strict Rules Enforced:
1. synthetic_label column used for training and validation model selection.
2. Zero clean_label usage in training, weighting, checkpoint selection, or model ranking.
3. Checkpoint selection based strictly on HIGHEST SYNTHETIC VALIDATION ACCURACY.
4. Outputs submission prediction CSVs for both models.
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

from check_format import check_pred_file
from src.config import get_default_config
from src.model_factory import create_model
from src.task4_robust_training import GCELoss, SCELoss, compute_sample_weights
from src.utils import export_predictions_csv, get_device, set_seed

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip_tf = T.RandomHorizontalFlip(p=0.5)


def batch_augment(imgs_batch):
    return flip_tf(crop_tf(imgs_batch))


class FastTensorDatasetWithWeights(Dataset):
    def __init__(self, images_tensor, labels_array, indices, weights_array=None):
        super().__init__()
        self.indices = indices
        self.labels = labels_array[indices]
        self.images = images_tensor[indices]
        self.weights = weights_array[indices] if weights_array is not None else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        global_idx = int(self.indices[idx])
        target = int(self.labels[idx])

        if self.weights is not None:
            w = float(self.weights[idx])
            return global_idx, img, target, w

        return global_idx, img, target


def train_synthetic_model(
    model_name: str,
    loss_type: str,
    config,
    epochs: int = 60,
    sample_weights_dict=None,
    out_csv_filename: str = "t5_baseline_synth.csv",
):
    """Train a model on synthetic_label column for 60 epochs."""
    print("\n" + "=" * 70, flush=True)
    print(f"      TASK 5 SYNTHETIC MODEL: {model_name.upper()} ({epochs} EPOCHS)", flush=True)
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

    # Use synthetic_label column
    train_synth_labels = labels_df["synthetic_label"].values[train_indices]
    val_synth_labels = labels_df["synthetic_label"].values[val_indices]

    mean = torch.tensor(config.mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32).view(1, 3, 1, 1)

    raw_tr = torch.from_numpy(all_imgs[train_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0
    tr_images_tensor = ((raw_tr - mean) / std).contiguous()

    raw_val = torch.from_numpy(all_imgs[val_indices]).permute(0, 3, 1, 2).contiguous().float() / 255.0
    val_images_tensor = ((raw_val - mean) / std).contiguous()

    raw_test = torch.from_numpy(test_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0
    test_images_tensor = ((raw_test - mean) / std).contiguous()

    tr_weights = None
    if sample_weights_dict is not None:
        tr_weights = np.array([sample_weights_dict[idx] for idx in train_indices], dtype=np.float32)

    train_ds = FastTensorDatasetWithWeights(
        tr_images_tensor, train_synth_labels, np.arange(len(train_indices)), weights_array=tr_weights
    )
    val_ds = FastTensorDatasetWithWeights(
        val_images_tensor, val_synth_labels, np.arange(len(val_indices)), weights_array=None
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=config.num_classes).to(device)

    # Instantiate loss function
    if loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == "gce":
        loss_fn = GCELoss(q=0.7)
    elif loss_type == "sce":
        loss_fn = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
    elif loss_type == "reweight":
        loss_fn = None
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.min_lr)

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"t5_{model_name}_best.pt")

    history = []
    start_time = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0

        for batch in train_loader:
            if loss_type == "reweight" and sample_weights_dict is not None:
                _, imgs, targets, weights = batch
                imgs = batch_augment(imgs).to(device)
                targets, weights = targets.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                ce = nn.functional.cross_entropy(outputs, targets, reduction="none")
                loss = (ce * weights).sum() / (weights.sum() + 1e-8)
            else:
                _, imgs, targets = batch
                imgs = batch_augment(imgs).to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * imgs.size(0)
            ep_correct += (outputs.argmax(dim=1) == targets).sum().item()

        scheduler.step()
        train_acc = (ep_correct / len(train_indices)) * 100.0
        avg_train_loss = ep_loss / len(train_indices)

        # Evaluate on Validation set using SYNTHETIC validation labels
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for _, imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        val_acc = (val_correct / len(val_indices)) * 100.0

        history.append({
            "epoch": ep,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_acc_synthetic": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), best_ckpt_path)

        if ep % 10 == 0 or ep == epochs or ep == 1:
            print(f"Epoch [{ep:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Acc (Synthetic): {val_acc:.2f}% (Best: {best_val_acc:.2f}% @ Ep {best_epoch})", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\nSynthetic Model {model_name.upper()} Finished in {elapsed_time:.1f}s. Best Val Acc (Synthetic): {best_val_acc:.2f}% at Epoch {best_epoch}.", flush=True)

    # Load best checkpoint to generate test predictions
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for i in range(0, len(test_images_tensor), config.batch_size):
            batch_imgs = test_images_tensor[i : i + config.batch_size].to(device)
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)

    test_preds = np.array(test_preds)

    csv_path = os.path.join(config.output_dir, out_csv_filename)
    export_predictions_csv(test_preds, csv_path)

    check_pred_file(csv_path)
    print(f"Validated format of {csv_path} (10,000 test predictions)", flush=True)

    summary = {
        "model_name": model_name,
        "loss_type": loss_type,
        "best_val_acc_synthetic": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc_synthetic": history[-1]["val_acc_synthetic"],
        "training_time_seconds": elapsed_time,
        "prediction_file": csv_path,
        "checkpoint_file": best_ckpt_path,
    }
    return summary, history


def main():
    config = get_default_config()

    print("=" * 70, flush=True)
    print("               TASK 5: SYNTHETIC NOISE BENCHMARKING", flush=True)
    print("=" * 70, flush=True)

    # 1. Train Synthetic Cross-Entropy Baseline Model
    sum_base, hist_base = train_synthetic_model(
        model_name="baseline_synth",
        loss_type="ce",
        config=config,
        epochs=60,
        out_csv_filename="t5_baseline_synth.csv"
    )

    # 2. Determine best Task 4 method from results.json (or default to task3_reweight / gce)
    res_path = os.path.join(HERE, "results.json")
    best_t4_method = "task3_reweight"
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)
            if "task4" in res_data and "best_method" in res_data["task4"]:
                best_t4_method = res_data["task4"]["best_method"]

    print(f"\n[Task 5] Selected Best Task 4 Method for Synthetic Training: '{best_t4_method}'", flush=True)

    # Map sample weights if best method is task3_reweight
    sample_weights_dict = None
    if best_t4_method == "task3_reweight":
        t3_csv = os.path.join(config.output_dir, "t3_noise_scores.csv")
        if os.path.exists(t3_csv):
            t3_df = pd.read_csv(t3_csv)
            raw_scores = t3_df.set_index("index")["score"].to_dict()
            score_vals = np.array(list(raw_scores.values()))
            min_s, max_s = score_vals.min(), score_vals.max()
            norm_scores = {idx: (score - min_s) / (max_s - min_s + 1e-8) for idx, score in raw_scores.items()}
            weights_array = compute_sample_weights(list(norm_scores.values()), mode="linear")
            sample_weights_dict = {idx: w for idx, w in zip(norm_scores.keys(), weights_array)}

    # 3. Train Best Task 4 Method on Synthetic Noise
    sum_best, hist_best = train_synthetic_model(
        model_name="best_synth",
        loss_type=best_t4_method,
        config=config,
        epochs=60,
        sample_weights_dict=sample_weights_dict,
        out_csv_filename="t5_best_synth.csv"
    )

    # 4. Record Task 5 findings in results.json
    results_data = {}
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)

    results_data["task5"] = {
        "val_acc_synthetic_labels__baseline": sum_base["best_val_acc_synthetic"],
        "val_acc_synthetic_labels__best_method": sum_best["best_val_acc_synthetic"],
        "best_method_name": best_t4_method,
        "baseline_best_epoch": sum_base["best_epoch"],
        "best_method_best_epoch": sum_best["best_epoch"],
        "baseline_prediction_file": sum_base["prediction_file"],
        "best_method_prediction_file": sum_best["prediction_file"]
    }

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("              TASK 5 SYNTHETIC NOISE EVALUATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Synthetic CE Baseline Val Acc:      {sum_base['best_val_acc_synthetic']:.2f}% (Epoch {sum_base['best_epoch']})", flush=True)
    print(f"Synthetic Best Method ({best_t4_method}): {sum_best['best_val_acc_synthetic']:.2f}% (Epoch {sum_best['best_epoch']})", flush=True)
    print(f"Predictions Saved: t5_baseline_synth.csv & t5_best_synth.csv", flush=True)
    print(f"Results recorded in: {res_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
