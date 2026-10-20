"""Utility functions for seed setting, device resolution, checkpointing, and prediction CSV writing."""
import csv
import math
import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolve the optimal available computing device (MPS -> CUDA -> CPU)."""
    if requested_device.lower() != "auto":
        return torch.device(requested_device)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_checkpoint(state: dict, filepath: str) -> None:
    """Save model/training state dictionary to checkpoint file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, map_location=None) -> dict:
    """Load model/training state dictionary from checkpoint file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    return torch.load(filepath, map_location=map_location)


def export_predictions_csv(predictions, filepath: str) -> None:
    """Export predictions to CSV adhering strictly to check_format.py rules.
    
    Args:
        predictions: Dictionary or array/list mapping test sample id (0..9999) to class prediction (0..9).
        filepath: Target CSV output path.
    """
    if isinstance(predictions, (list, np.ndarray)):
        preds_dict = {i: int(predictions[i]) for i in range(len(predictions))}
    elif isinstance(predictions, dict):
        preds_dict = predictions
    else:
        raise TypeError(f"Unsupported predictions type: {type(predictions)}")

    if len(preds_dict) != 10000:
        raise ValueError(f"Expected exactly 10,000 predictions, got {len(preds_dict)}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred"])
        for pid in range(10000):
            if pid not in preds_dict:
                raise ValueError(f"Missing prediction for sample id {pid}")
            pred = int(preds_dict[pid])
            if not (0 <= pred <= 9):
                raise ValueError(f"Invalid prediction class {pred} for sample id {pid}")
            writer.writerow([pid, pred])


def export_dynamics_csv(dynamics_rows: list, filepath: str) -> None:
    """Export epoch dynamics rows to t2_dynamics.csv.
    
    Header: epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"])
        for row in dynamics_rows:
            writer.writerow(row)


def export_scores_csv(scores: dict, filepath: str) -> None:
    """Export per-index noise scores to t3_noise_scores.csv.
    
    Header: index,score
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "score"])
        for idx in sorted(scores.keys()):
            val = float(scores[idx])
            if not math.isfinite(val):
                raise ValueError(f"Score for index {idx} must be finite, got {val}")
            writer.writerow([idx, val])
