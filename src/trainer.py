"""Generalized training, evaluation, and test prediction loops."""
import torch
import torch.nn as nn
import numpy as np


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    return_diagnostics: bool = False,
):
    """Train model for one epoch.
    
    If return_diagnostics=True, tracks accuracies on correctly labeled vs mislabeled training subsets.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_samples = 0

    # Diagnostic counters
    correct_lbl_total = 0
    correct_lbl_acc_cnt = 0
    mislbl_total = 0
    mislbl_acc_cnt = 0

    for batch in dataloader:
        if return_diagnostics:
            _, images, targets, clean_targets = batch
            clean_targets = clean_targets.numpy()
        else:
            _, images, targets = batch
            clean_targets = None

        images = images.to(device)
        targets_dev = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets_dev)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = outputs.argmax(dim=1).cpu().numpy()
        targets_np = targets.numpy()

        is_correct_pred = (preds == targets_np)
        correct_samples += is_correct_pred.sum()

        if return_diagnostics and clean_targets is not None:
            is_clean_same = (targets_np == clean_targets)
            is_mislabeled = ~is_clean_same

            correct_lbl_total += is_clean_same.sum()
            correct_lbl_acc_cnt += (is_correct_pred & is_clean_same).sum()

            mislbl_total += is_mislabeled.sum()
            mislbl_acc_cnt += (is_correct_pred & is_mislabeled).sum()

    avg_loss = total_loss / total_samples
    overall_acc = (correct_samples / total_samples) * 100.0

    if return_diagnostics:
        acc_correct_subset = (correct_lbl_acc_cnt / correct_lbl_total * 100.0) if correct_lbl_total > 0 else 0.0
        acc_mislabeled_subset = (mislbl_acc_cnt / mislbl_total * 100.0) if mislbl_total > 0 else 0.0
        return avg_loss, overall_acc, acc_correct_subset, acc_mislabeled_subset

    return avg_loss, overall_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_clean_target: bool = False,
):
    """Evaluate model accuracy and loss on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4 and use_clean_target:
                _, images, _, targets = batch
            elif len(batch) == 4 and not use_clean_target:
                _, images, targets, _ = batch
            else:
                _, images, targets = batch

            images = images.to(device)
            targets_dev = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets_dev)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.numpy()
            correct_samples += (preds == targets_np).sum()

    avg_loss = total_loss / total_samples
    acc = (correct_samples / total_samples) * 100.0
    return avg_loss, acc


def predict_test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Generate test predictions dictionary mapping image index -> predicted class."""
    model.eval()
    predictions = {}
    with torch.no_grad():
        for global_indices, images in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for idx_val, pred_val in zip(global_indices.numpy(), preds):
                predictions[int(idx_val)] = int(pred_val)

    return predictions
