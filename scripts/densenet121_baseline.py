
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


METADATA_DIM = 13


class MetadataMLP(nn.Module):
    """2-layer MLP for patient metadata. using dropout."""

    def __init__(self, input_dim=METADATA_DIM, output_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 with an optional metadata fusion toggle.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        Load ImageNet-pretrained weights.
    freeze_backbone : bool
        Freeze all backbone (feature) layers.
    use_metadata : bool
        If False  → standard image-only classifier (baseline behaviour).
        If True   → concatenates DenseNet-121 image features with MetadataMLP
                     output before the final classification head.
    metadata_dim : int
        Number of metadata input features (default matches team constant).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_metadata: bool = False,
        metadata_dim: int = METADATA_DIM,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        #backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Everything except the original classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        cnn_out = backbone.classifier.in_features  # 1024

        # ── metadata branch ──────────────
        if use_metadata:
            self.meta_mlp = MetadataMLP(input_dim=metadata_dim)
            combined_dim = cnn_out + self.meta_mlp.output_dim  # 1024 + 16
        else:
            combined_dim = cnn_out

        # classification head
        self.classifier = nn.Linear(combined_dim, num_classes)

        if freeze_backbone:
            for param in self.classifier.parameters():
                param.requires_grad = True
            if use_metadata:
                for param in self.meta_mlp.parameters():
                    param.requires_grad = True

    def forward(self, image, metadata=None):
        """
        Parameters
        ----------
        image : Tensor  (B, 3, H, W)
        metadata : Tensor (B, metadata_dim) or None
            Required when use_metadata=True.
        """
        x = self.features(image)
        x = nn.functional.relu(x, inplace=True)
        x = self.pool(x).flatten(1)  # (B, 1024)

        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata must be provided if use_metadata is True")
            m = self.meta_mlp(metadata)  # (B, 16)
            x = torch.cat([x, m], dim=1)  # (B, 1040)

        return self.classifier(x)


# convenience builder (backwards-compatible)
def build_densenet121(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    use_metadata: bool = False,
    metadata_dim: int = METADATA_DIM,
) -> nn.Module:
    """Build a DenseNet-121 classifier with a project-specific output head."""
    return DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
    )
