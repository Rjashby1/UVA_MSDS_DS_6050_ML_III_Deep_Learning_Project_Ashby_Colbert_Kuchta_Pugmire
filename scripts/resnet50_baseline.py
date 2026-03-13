"""ResNet-50 baseline model helper for the DS 6050 skin-lesion project.

Usage:
    from resnet50_baseline import build_resnet50
    model = build_resnet50(num_classes=NUM_CLASSES)
"""

from __future__ import annotations

import torch 
import torch.nn as nn
from torchvision import models

 
class MetadataMLP(nn.Module):
    """2-layer MLP for patient metadata. using dropout."""
 
    def __init__(self, input_dim=13, output_dim=16):
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

class ResNet50(nn.Module):
    """Build either a ResNet-50 classifier with a project-specific output head or a ResNet-50 classifier with a branch fused with metadata.

    Args:
        num_classes: Number of target classes.
        pretrained: Whether to load ImageNet weights.
        freeze_backbone: If True, freeze all layers except the final head.
        use_metadata: If True, return a model with a second branch for metadata.
        metadata_dim: The input dimensions of the metadata branch.

    Returns:
        A torchvision ResNet-50 model.
    """
    def __init__(self, num_classes, metadata_dim=None, pretrained = True, freeze_backbone = False, use_metadata=False, ):
        super().__init__()

        self.use_metadata = use_metadata

        # Set the weights for the ResNet 50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None

        # Define the backbone of the 
        self.backbone = models.resnet50(weights=weights)

        # Get the number of features from the ResNet 50 that will be part of the input for the final layer
        in_features = self.backbone.fc.in_features

        # Freeze all layers exepct the final layer
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if self.use_metadata:

            if metadata_dim is None:
                raise ValueError("metadata_dim must be provided if use_metadata is True")

            # Create the metedata branch of the model
            self.metadata_model = MetadataMLP(metadata_dim)

            # Get the number of features from the MLP that will be part of the input for the final layer
            metadata_features = self.metadata_model.output_dim

            # Get the output features from the ResNet 50 instead of the class predictions
            self.backbone.fc = nn.Identity()

            # Define a new final layer for the ResNet 50
            self.classifier = nn.Linear(in_features+metadata_features, num_classes)

        else:
            # Just replace the head of the ResNet 50 with a project-specific classifier head
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, image, metadata=None):
        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata must be provided if use_metadata is True")
            image_features = self.backbone(image)
            metadata_features = self.metadata_model(metadata)
            fused_features = torch.cat((image_features, metadata_features), dim=1)
            output = self.classifier(fused_features)
            return output
        else:
            return self.backbone(image)

def build_resnet50(num_classes: int, metadata_dim: int | None = None, pretrained: bool = True, freeze_backbone: bool = False, use_metadata: bool = False,) -> nn.Module:
    """Build a ResNet-50 classifier with a project-specific output head."""
    model = ResNet50(
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        use_metadata=use_metadata,     
    )
    return model