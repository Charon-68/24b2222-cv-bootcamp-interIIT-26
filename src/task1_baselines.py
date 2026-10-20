"""Task 1 Baselines & Data Auditing Execution Script.

Trains two ResNet-18 models from scratch for 60 epochs:
1. Clean Reference Model (trained on clean_label) -> predictions/t1_clean_ref.csv
2. Noisy CE Model (trained on noisy_label) -> predictions/t1_noisy_ce.csv

Logs per-epoch training dynamics for Task 2 as well:
- train_acc_correct
- train_acc_mislabeled
- val_acc_noisy
- val_acc_clean
-> predictions/t2_dynamics.csv
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.config import get_default_config
from src.dataset import CIFAR10NDataset, LabelMode, get_dataloaders
from src.metrics import compute_confusion_matrix
from src.model_factory import create_model
from src.trainer import evaluate, predict_test, train_one_epoch
from src.utils import (
    export_dynamics_csv,
    export_predictions_csv,
    get_device,
    save_checkpoint,
    set_seed,
)


def compute_dataset_noise_audit(data_dir: str):
    """Compute and print human noise rate and 10x10 confusion matrix."""
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    idx_path = os.path.join(data_dir, "train_idx.txt")
    with open(idx_path, encoding="utf-8") as f:
        train_idx = [int(line.strip()) for line in f if line.strip()]

    train_labels = labels_df.iloc[train_idx]
    noisy_lbls = train_labels["noisy_label"].values
    clean_lbls = train_labels["clean_label"].values

    mislabeled_mask = noisy_lbls != clean_lbls
    mislabeled_count = mislabeled_mask.sum()
    noise_rate = mislabeled_count / len(train_idx)

    cm = compute_confusion_matrix(noisy_lbls, clean_lbls, num_classes=10)

    print("=" * 70)
    print("                    TASK 1 DATA NOISE AUDIT")
    print("=" * 70)
    print(f"Training set total samples: {len(train_idx)}")
    print(f"Mislabeled samples count:  {mislabeled_count}")
    print(f"Human label noise rate:    {noise_rate:.6f} ({noise_rate * 100:.4f}%)")
    print("\n10x10 Confusion Matrix (Rows: Noisy, Cols: Clean):")
    print(cm)
    print("=" * 70)
    return noise_rate, cm


def train_baseline_model(
    config,
    label_mode: LabelMode,
    model_name: str,
    epochs: int = None,
    log_dynamics: bool = False,
):
    """Train a ResNet-18 baseline model for the specified number of epochs."""
    if epochs is None:
        epochs = config.epochs

    set_seed(config.seed)
    device = get_device(config.device)

    print(f"\n[Task 1] Starting Training: {model_name}")
    print(f"         Label Mode: {label_mode.value}")
    print(f"         Epochs: {epochs}, Batch Size: {config.batch_size}, LR: {config.lr}, Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        label_mode=label_mode,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        return_diagnostics_clean=log_dynamics,
    )

    # Initialize model, loss, optimizer, scheduler
    model = create_model(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.min_lr)

    dynamics_rows = []
    best_val_acc_noisy = 0.0
    best_val_acc_clean = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        if log_dynamics:
            train_loss, train_acc, acc_correct, acc_mislabeled = train_one_epoch(
                model, train_loader, criterion, optimizer, device, return_diagnostics=True
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, return_diagnostics=False
            )
            acc_correct, acc_mislabeled = 0.0, 0.0

        # Evaluate on validation set
        if log_dynamics:
            # val_loader returns (idx, img, noisy_lbl, clean_lbl)
            val_loss_noisy, val_acc_noisy = evaluate(model, val_loader, criterion, device, use_clean_target=False)
            val_loss_clean, val_acc_clean = evaluate(model, val_loader, criterion, device, use_clean_target=True)
        else:
            val_loss_noisy, val_acc_noisy = evaluate(model, val_loader, criterion, device, use_clean_target=False)
            val_acc_clean = val_acc_noisy

        scheduler.step()
        ep_time = time.time() - ep_start

        if val_acc_noisy > best_val_acc_noisy:
            best_val_acc_noisy = val_acc_noisy
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc_noisy},
                            os.path.join(config.checkpoint_dir, f"{model_name}_best.pth"))

        if val_acc_clean > best_val_acc_clean:
            best_val_acc_clean = val_acc_clean

        if log_dynamics:
            # Dynamics format: epoch, train_acc_correct, train_acc_mislabeled, val_acc_noisy, val_acc_clean
            # Note: epoch counting starts at 0 or 1 per check_format.py
            dynamics_rows.append([epoch - 1, float(acc_correct), float(acc_mislabeled), float(val_acc_noisy), float(val_acc_clean)])

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            lr_curr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | LR: {lr_curr:.4f} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Acc (Noisy): {val_acc_noisy:.2f}%, (Clean): {val_acc_clean:.2f}%", flush=True)

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time:.2f} seconds.")
    print(f"Best Val Acc (Noisy): {best_val_acc_noisy:.2f}%, Final Val Acc (Noisy): {val_acc_noisy:.2f}%")

    # Save final epoch checkpoint
    save_checkpoint({"epoch": epochs, "model_state": model.state_dict(), "val_acc": val_acc_noisy},
                    os.path.join(config.checkpoint_dir, f"{model_name}_final.pth"))

    # Generate final-epoch test set predictions
    print(f"Generating test predictions from final epoch (epoch {epochs})...")
    test_preds = predict_test(model, test_loader, device)

    return model, test_preds, dynamics_rows, total_time, best_val_acc_noisy, val_acc_noisy


def main_smoke_test(config):
    """Run a 2-epoch smoke test on both baseline models before full training."""
    print("=" * 70)
    print("           TASK 1 BASELINES - SHORT SMOKE TEST (2 EPOCHS)")
    print("=" * 70)
    
    # Clean Ref smoke test
    model_c, preds_c, _, t_c, _, acc_c = train_baseline_model(
        config, LabelMode.CLEAN_REF, "smoke_clean_ref", epochs=2, log_dynamics=False
    )
    tmp_clean_csv = os.path.join(config.output_dir, "t1_clean_ref_smoke.csv")
    export_predictions_csv(preds_c, tmp_clean_csv)

    # Noisy CE smoke test
    model_n, preds_n, dyn_n, t_n, _, acc_n = train_baseline_model(
        config, LabelMode.NOISY, "smoke_noisy_ce", epochs=2, log_dynamics=True
    )
    tmp_noisy_csv = os.path.join(config.output_dir, "t1_noisy_ce_smoke.csv")
    export_predictions_csv(preds_n, tmp_noisy_csv)

    from check_format import check_pred_file, check_dynamics
    check_pred_file(tmp_clean_csv)
    check_pred_file(tmp_noisy_csv)
    print("\n[SMOKE TEST PASSED] Both baseline models compiled, trained, and produced valid 10,000-row CSVs!")

    # Cleanup smoke CSVs
    if os.path.exists(tmp_clean_csv):
        os.remove(tmp_clean_csv)
    if os.path.exists(tmp_noisy_csv):
        os.remove(tmp_noisy_csv)


def main_full_task1(config):
    """Execute full 60-epoch Task 1 training for Clean Ref and Noisy CE models."""
    print("=" * 70)
    print("         TASK 1 BASELINES - FULL 60-EPOCH TRAINING RUN")
    print("=" * 70)

    noise_rate, cm = compute_dataset_noise_audit(config.data_dir)

    # 1. Clean Reference Model (trained on clean_label for 60 epochs)
    t1_clean_ref_path = os.path.join(config.output_dir, "t1_clean_ref.csv")
    clean_model, clean_preds, _, time_clean, best_val_clean, final_val_clean = train_baseline_model(
        config, LabelMode.CLEAN_REF, "t1_clean_ref", epochs=config.epochs, log_dynamics=False
    )
    export_predictions_csv(clean_preds, t1_clean_ref_path)

    # 2. Noisy CE Baseline Model (trained on noisy_label for 60 epochs)
    t1_noisy_ce_path = os.path.join(config.output_dir, "t1_noisy_ce.csv")
    t2_dynamics_path = os.path.join(config.output_dir, "t2_dynamics.csv")

    noisy_model, noisy_preds, dynamics_rows, time_noisy, best_val_noisy, final_val_noisy = train_baseline_model(
        config, LabelMode.NOISY, "t1_noisy_ce", epochs=config.epochs, log_dynamics=True
    )
    export_predictions_csv(noisy_preds, t1_noisy_ce_path)
    export_dynamics_csv(dynamics_rows, t2_dynamics_path)

    # Validate output files with check_format.py
    from check_format import check_pred_file, check_dynamics
    check_pred_file(t1_clean_ref_path)
    check_pred_file(t1_noisy_ce_path)
    check_dynamics(t2_dynamics_path)

    print("\n" + "=" * 70)
    print("                 TASK 1 BASELINES COMPLETE")
    print("=" * 70)
    print(f"Clean Reference Model Final Val Acc (Noisy Labels): {final_val_clean:.2f}% | Best: {best_val_clean:.2f}%")
    print(f"Noisy CE Baseline Model Final Val Acc (Noisy Labels): {final_val_noisy:.2f}% | Best: {best_val_noisy:.2f}%")
    print(f"Predictions saved: {t1_clean_ref_path}")
    print(f"Predictions saved: {t1_noisy_ce_path}")
    print(f"Dynamics saved:    {t2_dynamics_path}")
    print(f"Total time taken:  {time_clean + time_noisy:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    config = get_default_config()
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main_smoke_test(config)
    else:
        # Step A: Smoke test first
        main_smoke_test(config)
        # Step B: Full training
        main_full_task1(config)
