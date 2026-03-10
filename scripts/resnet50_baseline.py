"""ResNet-50 baseline model helper for the DS 6050 skin-lesion project.

Usage:
    from resnet50_baseline import build_resnet50
    model = build_resnet50(num_classes=NUM_CLASSES)
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a ResNet-50 classifier with a project-specific output head.

    Args:
        num_classes: Number of target classes.
        pretrained: Whether to load ImageNet weights.
        freeze_backbone: If True, freeze all layers except the final head.

    Returns:
        A torchvision ResNet-50 model.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
