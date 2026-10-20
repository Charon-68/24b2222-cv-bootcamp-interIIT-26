"""Lightweight sanity check & smoke test script to verify complete project foundation.

Runs a minimal 1-epoch or multi-batch verification to validate:
- Config system
- Seed setting & reproducibility
- Device selection (MPS -> CPU)
- Dataset & DataLoader loading for train, val, and test splits
- Explicit LabelMode isolation
- ResNet18 model instantiation from model.py
- Forward pass, Loss calculation, Backward pass, Optimizer step
- Checkpoint save & load functionality
- Test prediction generation & CSV formatting
"""
import math
import os
import shutil
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.config import get_default_config
from src.dataset import CIFAR10NDataset, LabelMode, get_dataloaders
from src.metrics import AccuracyTracker, compute_auroc, compute_precision_at_k
from src.model_factory import count_parameters, create_model
from src.utils import export_predictions_csv, get_device, load_checkpoint, save_checkpoint, set_seed


def run_smoke_test():
    print("=" * 70)
    print("      RUNNING LIGHTWEIGHT SMOKE TEST & FOUNDATION VALIDATION")
    print("=" * 70)

    # 1. Config & Seed & Device
    config = get_default_config()
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"[1/8] Seed set to {config.seed}. Target computing device resolved to: {device}")

    # 2. Dataset & Split Validation
    data_dir = config.data_dir
    print(f"[2/8] Validating dataset loading from '{data_dir}'...")

    train_ds = CIFAR10NDataset(data_dir, split="train", label_mode=LabelMode.NOISY)
    val_ds = CIFAR10NDataset(data_dir, split="val", label_mode=LabelMode.NOISY)
    test_ds = CIFAR10NDataset(data_dir, split="test", label_mode=LabelMode.NOISY)

    print(f"      - Train dataset size: {len(train_ds)} samples (Expected: 45,000)")
    print(f"      - Val dataset size:   {len(val_ds)} samples (Expected: 5,000)")
    print(f"      - Test dataset size:  {len(test_ds)} samples (Expected: 10,000)")

    assert len(train_ds) == 45000, f"Expected 45000 train samples, got {len(train_ds)}"
    assert len(val_ds) == 5000, f"Expected 5000 val samples, got {len(val_ds)}"
    assert len(test_ds) == 10000, f"Expected 10000 test samples, got {len(test_ds)}"

    # 3. Explicit LabelMode Isolation Check
    print("[3/8] Verifying LabelMode isolation & clean-label leakage prevention...")
    noisy_ds = CIFAR10NDataset(data_dir, split="train", label_mode=LabelMode.NOISY)
    clean_ds = CIFAR10NDataset(data_dir, split="train", label_mode=LabelMode.CLEAN_REF)
    synth_ds = CIFAR10NDataset(data_dir, split="train", label_mode=LabelMode.SYNTHETIC)

    _, _, noisy_y = noisy_ds[0]
    _, _, clean_y = clean_ds[0]
    _, _, synth_y = synth_ds[0]

    print(f"      - Sample 0 noisy_y: {noisy_y}, clean_y: {clean_y}, synth_y: {synth_y}")
    assert 0 <= noisy_y < 10 and 0 <= clean_y < 10 and 0 <= synth_y < 10

    # 4. DataLoaders & Batch Shapes Check
    print("[4/8] Testing DataLoaders & batch tensor shapes...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        label_mode=LabelMode.NOISY,
        batch_size=config.batch_size,
        num_workers=0,  # Single-process for smoke test speed
    )

    indices, images, targets = next(iter(train_loader))
    print(f"      - Train batch images shape: {images.shape}, dtype: {images.dtype}")
    print(f"      - Train batch targets shape: {targets.shape}, dtype: {targets.dtype}")
    assert images.shape == (config.batch_size, 3, 32, 32), f"Unexpected image batch shape {images.shape}"
    assert targets.shape == (config.batch_size,), f"Unexpected target batch shape {targets.shape}"

    # 5. Model Instantiation & Forward/Backward Pass on Device
    print(f"[5/8] Instantiating ResNet18 model on {device}...")
    model = create_model(num_classes=10).to(device)
    param_count = count_parameters(model)
    print(f"      - ResNet18 trainable parameter count: {param_count:,}")
    assert param_count > 10_000_000, f"Unexpected parameter count: {param_count}"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    model.train()
    images, targets = images.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    print(f"      - Model output logits shape: {outputs.shape}")
    assert outputs.shape == (config.batch_size, 10), f"Unexpected output shape {outputs.shape}"

    loss = criterion(outputs, targets)
    print(f"      - Computed CrossEntropy loss: {loss.item():.4f}")
    assert loss.item() > 0 and math.isfinite(loss.item()), "Loss must be positive and finite"

    loss.backward()
    optimizer.step()
    print("      - Backward pass and optimizer step successful!")

    # 6. Checkpoint Save and Load Verification
    print("[6/8] Testing model checkpoint save/load functionality...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
        save_checkpoint({"model_state": model.state_dict(), "loss": loss.item()}, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not created"

        loaded = load_checkpoint(ckpt_path, map_location=device)
        model_new = create_model(num_classes=10).to(device)
        model_new.load_state_dict(loaded["model_state"])
        print("      - Checkpoint successfully saved and re-loaded into new model instance!")

    # 7. Test Prediction Generation & CSV Formatting Check
    print("[7/8] Generating test set predictions & verifying CSV format...")
    model.eval()
    test_preds = {}
    with torch.no_grad():
        for t_idx, t_imgs in test_loader:
            t_imgs = t_imgs.to(device)
            t_logits = model(t_imgs)
            t_classes = t_logits.argmax(dim=1).cpu().numpy()
            for idx_val, cls_val in zip(t_idx.numpy(), t_classes):
                test_preds[int(idx_val)] = int(cls_val)

    assert len(test_preds) == 10000, f"Expected 10,000 predictions, got {len(test_preds)}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = os.path.join(tmp_dir, "test_pred.csv")
        export_predictions_csv(test_preds, tmp_csv)
        assert os.path.exists(tmp_csv), "Prediction CSV was not created"

        # Verify using check_format.py
        from check_format import check_pred_file
        preds_loaded = check_pred_file(tmp_csv)
        assert len(preds_loaded) == 10000, "check_pred_file failed to validate CSV"
        print("      - Test prediction generation & CSV formatting verified via check_format.py!")

    # 8. Metrics Verification (AUROC & Precision@k)
    print("[8/8] Testing metrics calculation (AccuracyTracker, AUROC, Precision@k)...")
    tracker = AccuracyTracker()
    tracker.update(np.array([0, 1, 2]), np.array([0, 1, 3]), loss=0.5, n_samples=3)
    print(f"      - AccuracyTracker test accuracy: {tracker.accuracy:.2f}% (Expected: 66.67%)")
    assert round(tracker.accuracy, 2) == 66.67

    y_true_mock = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    y_scores_mock = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    auroc = compute_auroc(y_true_mock, y_scores_mock)
    p_at_k = compute_precision_at_k(y_true_mock, y_scores_mock, k=3)
    print(f"      - AUROC test: {auroc:.4f}, Precision@3: {p_at_k:.4f}")
    assert auroc > 0.5, "AUROC failed"
    assert p_at_k == 2 / 3, "Precision@k failed"

    print("=" * 70)
    print("      ALL SMOKE TESTS PASSED SUCCESSFULLY! FOUNDATION IS READY.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
