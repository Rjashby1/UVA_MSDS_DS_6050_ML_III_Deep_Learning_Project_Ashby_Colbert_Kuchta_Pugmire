"""EfficientNet-B0 baseline model helper for the DS 6050 skin-lesion project.

Usage:
    from efficientnet_b0_baseline import build_efficientnet_b0
    model = build_efficientnet_b0(num_classes=NUM_CLASSES)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
# gives us best available ImageNet weights for this architecture
from torchvision.models import EfficientNet_B0_Weights

# this will be constant across all experiments. # of metadata columns
METADATA_DIM = 13
 
class MetadataMLP(nn.Module):
    """2-layer MLP for patient metadata. using dropout."""
 
    def __init__(self, input_dim=METADATA_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )
 
 
    def forward(self, x):
        return self.mlp(x)



class EfficientNetB0(nn.Module):

    def __init__(self, num_classes, pretrained=True, freeze_backbone=False,
                 use_metadata=False, metadata_dim=METADATA_DIM):
        super().__init__()

        # Stores the toggle as an instance variable so forward() can check it later
        self.use_metadata = use_metadata
        
        # load pretrained backbone
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        #the ciritical parts of efficient net:
        #  features(stem+7 MBCONVs + projection)
        #  avgpool - global average pooling
        #  classifier. we don't take theirs because we want to replace it with our own head.
        self.features = model.features
        self.avgpool  = model.avgpool
        # fixed architecture - each image becomes 1280 feature maps.
        cnn_out_dim   = 1280

        # freeze early layers, keep last block + projection head trainable
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            # last MBConv block
            for param in self.features[7].parameters():
                param.requires_grad = True
            # 1x1 projection to 1280 feature maps
            for param in self.features[8].parameters():
                param.requires_grad = True

        # metadata branch
        #  if use_metadata is true. meta_mlp = 16 ( the output size of the last layer of Metadata MLP)
        #  and the classifier takes 1296 inputs
        if use_metadata:
            self.meta_mlp   = MetadataMLP(input_dim=metadata_dim)
            classifier_in   = cnn_out_dim + 16
        # if use_metadata is false. meta_mlp = None and the classifier takes 1280 inputs
        else:
            self.meta_mlp   = None
            classifier_in   = cnn_out_dim
        # the output head
        # no softmax here - CrossEntropLoss does it in the training loop. 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, images, metadata=None):
        # feature extraction
        x = self.features(images)
        # global average pooling
        x = self.avgpool(x)
        # flatten. # the result of global pooling is a box of size 1x1x1280. flattening it to a vector of size 1280
        x = torch.flatten(x, 1)

        # LATE FUSION
        if self.use_metadata and self.meta_mlp is not None:
        # "was this model built to USE metadata?" yes, proceed
            if metadata is None:
            # "did the caller forget to pass metadata?" use zeros as fallback
            # concatenate with metadata. Late Fusion.
                metadata = torch.zeros(images.size(0), METADATA_DIM, device=images.device)
            x = torch.cat([x, self.meta_mlp(metadata)], dim=1)

        return self.classifier(x)


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    use_metadata: bool = False,
    metadata_dim: int = METADATA_DIM,
) -> nn.Module:
    """Build an EfficientNet-B0 classifier with a project-specific output head."""
    model = EfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
    )
    return model



# 
#  PSEUDOCODE
# 
#
# WITHOUT METADATA (Phases 1-4):
# 
# Image (224x224x3)
#     ↓
# EfficientNet Backbone (frozen layers 0-6)
#     ↓
# Fine-tuned layers (features[7], features[8])
#     ↓
# Global Average Pooling
#     ↓
# 1280-dim image vector
#     ↓
# Dropout(0.2) → Linear(1280 → 8)
#     ↓
# 8 class scores
#
################################
#
# WITH METADATA (Phase 5):
# 
# Image (224x224x3)          Patient Record
#     ↓                      (age, sex, site)
# EfficientNet Backbone           ↓
#     ↓                      Linear(13 → 32)
# Fine-tuned layers               ↓
#     ↓                        ReLU
# Global Average Pooling          ↓
#     ↓                      Dropout(0.2)
# 1280-dim vector            Linear(32 → 16)
#     ↓                           ↓
#     └──────── CONCATENATE ───────┘
#                   ↓
#            1296-dim vector
#                   ↓
#        Dropout(0.2) → Linear(1296 → 8)
#                   ↓
#            8 class scores
#
# 