"""Plotting utilities for visualization of dynamics, confusion matrices, and noise score distributions."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dynamics(dynamics_csv_path: str, save_path: str = None) -> None:
    """Plot Task 2 training dynamics curves: train_acc_correct, train_acc_mislabeled, val_acc_noisy, val_acc_clean."""
    if not os.path.exists(dynamics_csv_path):
        raise FileNotFoundError(f"Dynamics CSV not found at {dynamics_csv_path}")

    df = pd.read_csv(dynamics_csv_path)
    epochs = df["epoch"].values

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, df["train_acc_correct"], label="Train Acc (Correct Labels)", color="green", linewidth=2)
    plt.plot(epochs, df["train_acc_mislabeled"], label="Train Acc (Mislabeled Labels)", color="red", linewidth=2, linestyle="--")
    plt.plot(epochs, df["val_acc_noisy"], label="Val Acc (Noisy Labels)", color="orange", linewidth=2)
    plt.plot(epochs, df["val_acc_clean"], label="Val Acc (Clean Labels)", color="blue", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("CIFAR-10N Training Dynamics & Memorization", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="center right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None) -> None:
    """Plot 10x10 confusion matrix heatmap."""
    plt.figure(figsize=(9, 7), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("10x10 Confusion Matrix (Noisy vs Clean Labels)", fontsize=13, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    plt.xlabel("Clean Class", fontsize=11)
    plt.ylabel("Noisy Class", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
