"""Centralized configuration module for CIFAR-10N project."""
from dataclasses import dataclass, field
import os
from typing import Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")


@dataclass
class Config:
    """Central configuration class for data, model, and training parameters."""
    
    # Paths
    data_dir: str = DATA_DIR
    output_dir: str = os.path.join(HERE, "predictions")
    checkpoint_dir: str = os.path.join(HERE, "checkpoints")
    
    # Reproducibility & Device
    seed: int = 42
    device: str = "auto"  # auto resolves to mps -> cuda -> cpu
    
    # Dataset Parameters
    num_classes: int = 10
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
    
    # Training Parameters (Standard CIFAR-10 setup for ResNet-18 on M1)
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 60
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "multistep"
    min_lr: float = 1e-3
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def get_default_config() -> Config:
    """Returns the default configuration object."""
    return Config()
