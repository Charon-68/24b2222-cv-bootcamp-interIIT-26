"""Dataset, preprocessing, and DataLoader utilities for CIFAR-10N (Optimized High-Speed Pipeline)."""
from enum import Enum
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class LabelMode(str, Enum):
    """Explicit label modes to prevent accidental clean-label leakage."""
    NOISY = "noisy"
    CLEAN_REF = "clean_reference"
    SYNTHETIC = "synthetic"


class CIFAR10NDataset(Dataset):
    """Custom Dataset for CIFAR-10N supporting explicit label modes, split indexing, and fast pre-normalized memory caching."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        label_mode: str = LabelMode.NOISY,
        is_train_augment: bool = False,
        return_diagnostics_clean: bool = False,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ):
        """
        Args:
            data_dir: Path to data directory.
            split: "train" (45k), "val" (5k), or "test" (10k).
            label_mode: "noisy", "clean_reference", or "synthetic".
            is_train_augment: If True, applies RandomCrop(32, padding=4) and RandomHorizontalFlip().
            return_diagnostics_clean: Returns (index, image, target, clean_label) if True.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split.lower()
        self.label_mode = LabelMode(label_mode)
        self.is_train_augment = is_train_augment
        self.return_diagnostics_clean = return_diagnostics_clean

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")

        if self.split == "test":
            test_path = os.path.join(data_dir, "test_images.npy")
            raw_images = np.load(test_path)  # (10000, 32, 32, 3) uint8
            self.indices = np.arange(len(raw_images))
            self.labels = None
            self.clean_labels = None
        else:
            train_images_path = os.path.join(data_dir, "train_images.npy")
            all_images = np.load(train_images_path)  # (50000, 32, 32, 3) uint8
            labels_path = os.path.join(data_dir, "labels.csv")
            labels_df = pd.read_csv(labels_path)

            idx_filename = "train_idx.txt" if self.split == "train" else "val_idx.txt"
            idx_path = os.path.join(data_dir, idx_filename)
            with open(idx_path, encoding="utf-8") as f:
                self.indices = np.array([int(line.strip()) for line in f if line.strip()])

            raw_images = all_images[self.indices]

            # Select target labels strictly based on label_mode
            if self.label_mode == LabelMode.NOISY:
                target_col = "noisy_label"
            elif self.label_mode == LabelMode.SYNTHETIC:
                target_col = "synthetic_label"
            elif self.label_mode == LabelMode.CLEAN_REF:
                target_col = "clean_label"
            else:
                raise ValueError(f"Unsupported label mode: {self.label_mode}")

            self.labels = labels_df[target_col].values[self.indices]
            self.clean_labels = labels_df["clean_label"].values[self.indices]

        # Convert images to contiguous float32 normalized PyTorch Tensors of shape (N, 3, 32, 32)
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        raw_t = torch.from_numpy(raw_images).permute(0, 3, 1, 2).contiguous().float() / 255.0
        self.images = ((raw_t - mean_t) / std_t).contiguous()

        # Fast augmentation transforms for training
        if self.is_train_augment:
            self.crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
            self.flip_tf = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        global_idx = int(self.indices[idx])

        if self.is_train_augment:
            img = self.crop_tf(img)
            img = self.flip_tf(img)

        if self.split == "test":
            return global_idx, img

        target = int(self.labels[idx])
        if self.return_diagnostics_clean:
            clean_target = int(self.clean_labels[idx])
            return global_idx, img, target, clean_target

        return global_idx, img, target


def get_dataloaders(
    data_dir: str,
    label_mode: str = LabelMode.NOISY,
    batch_size: int = 128,
    num_workers: int = 0,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    return_diagnostics_clean: bool = False,
):
    """Factory creating DataLoaders for train, validation, and test splits."""
    train_ds = CIFAR10NDataset(
        data_dir=data_dir,
        split="train",
        label_mode=label_mode,
        is_train_augment=True,
        return_diagnostics_clean=return_diagnostics_clean,
        mean=mean,
        std=std,
    )

    val_ds = CIFAR10NDataset(
        data_dir=data_dir,
        split="val",
        label_mode=label_mode,
        is_train_augment=False,
        return_diagnostics_clean=return_diagnostics_clean,
        mean=mean,
        std=std,
    )

    test_ds = CIFAR10NDataset(
        data_dir=data_dir,
        split="test",
        label_mode=label_mode,
        is_train_augment=False,
        return_diagnostics_clean=False,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
