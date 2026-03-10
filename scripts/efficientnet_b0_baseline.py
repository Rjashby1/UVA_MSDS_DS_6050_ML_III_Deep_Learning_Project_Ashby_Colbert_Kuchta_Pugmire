"""EfficientNet-B0 baseline model helper for the DS 6050 skin-lesion project.

Usage:
    from efficientnet_b0_baseline import build_efficientnet_b0
    model = build_efficientnet_b0(num_classes=NUM_CLASSES)
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build an EfficientNet-B0 classifier with a project-specific output head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    return model
