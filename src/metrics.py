"""Metrics evaluation module for tracking accuracies, loss, AUROC, and Precision@k."""
import numpy as np
from sklearn.metrics import roc_auc_score


class AccuracyTracker:
    """Utility class for tracking cumulative loss and classification accuracy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0

    def update(self, preds: np.ndarray, targets: np.ndarray, loss: float = 0.0, n_samples: int = None):
        if n_samples is None:
            n_samples = len(targets)

        self.correct += int((preds == targets).sum())
        self.total += int(n_samples)
        self.loss_sum += float(loss) * int(n_samples)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100.0

    @property
    def avg_loss(self) -> float:
        if self.total == 0:
            return 0.0
        return self.loss_sum / self.total


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under Receiver Operating Characteristic Curve (AUROC).
    
    Args:
        y_true: Binary array (1 for mislabeled/noisy, 0 for clean).
        y_scores: Continuous noise score (higher = more likely mislabeled).
    """
    return float(roc_auc_score(y_true, y_scores))


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Compute Precision@k for binary noise detection.
    
    Args:
        y_true: Binary ground truth array (1 for mislabeled, 0 for clean).
        y_scores: Score array where higher score implies higher noise likelihood.
        k: Top k predictions to evaluate (e.g., k = 18,109 for training set).
    """
    if k <= 0 or k > len(y_true):
        raise ValueError(f"k must be between 1 and {len(y_true)}, got {k}")

    # Top k indices with highest noise scores
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    precision = np.mean(y_true[top_k_indices])
    return float(precision)


def compute_confusion_matrix(y_noisy: np.ndarray, y_clean: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute 10x10 confusion matrix where rows are noisy labels and columns are clean labels."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for n_lbl, c_lbl in zip(y_noisy, y_clean):
        cm[n_lbl, c_lbl] += 1
    return cm
