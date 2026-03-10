"""DenseNet-121 baseline model helper for the DS 6050 skin-lesion project.

Usage:
    from densenet121_baseline import build_densenet121
    model = build_densenet121(num_classes=NUM_CLASSES)
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_densenet121(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a DenseNet-121 classifier with a project-specific output head."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model
