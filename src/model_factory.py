"""Model factory importing and instantiating the exact provided ResNet18."""
import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in sys.path to import model.py cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from model import ResNet18
except ImportError as e:
    raise ImportError(f"Could not import ResNet18 from model.py at {PROJECT_ROOT}: {e}") from e


def create_model(num_classes: int = 10) -> nn.Module:
    """Instantiate the exact provided ResNet-18 architecture for 32x32 CIFAR images.
    
    Args:
        num_classes: Number of classification output classes (default: 10).
        
    Returns:
        torch.nn.Module: Freshly initialized ResNet18 instance.
    """
    model = ResNet18(num_classes=num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
