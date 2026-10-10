"""Shared DINOv2 ViT-S/14 visual encoder (Section 3.1)."""

import torch
from torch import nn

from .._vendor.dinov2 import DinoVisionTransformer


class VisualEncoder(nn.Module):
    """Return P=256 patch tokens per 224x224 view, discarding the class token."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DinoVisionTransformer()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
