# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0; see DINOV2_LICENSE.
# Adapted from facebookresearch/dinov2 for CoRP ViT-S/14 inference.
# References: facebookresearch/dino and rwightman/pytorch-image-models.

import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(384, 384 * 3)
        self.proj = nn.Linear(384, 384)

    def forward(self, tokens):
        batch_size, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, length, 3, 6, width // 6)
        query, key, value = (part.transpose(1, 2) for part in torch.unbind(qkv, 2))
        tokens = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return self.proj(
            tokens.transpose(1, 2).contiguous().view(batch_size, length, width)
        )


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, tokens):
        return self.fc2(self.act(self.fc1(tokens)))


class LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))

    def forward(self, tokens):
        return tokens * self.gamma


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, tokens):
        tokens = tokens + self.ls1(self.attn(self.norm1(tokens)))
        return tokens + self.ls2(self.mlp(self.norm2(tokens)))


class DinoVisionTransformer(nn.Module):
    """ViT-S/14 with the checkpoint's 37x37 positional embedding grid."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 37 * 37 + 1, 384))
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)

    def interpolate_pos_encoding(self, tokens, width, height):
        patch_count = tokens.shape[1] - 1
        grid_count = self.pos_embed.shape[1] - 1
        if patch_count == grid_count and width == height:
            return self.pos_embed
        position = self.pos_embed.float()
        grid_size = int(math.sqrt(grid_count))
        patch_width, patch_height = width // 14, height // 14
        # DINOv2's +0.1 offset and scale-factor interpolation are part of the checkpoint contract.
        patches = F.interpolate(
            position[:, 1:].reshape(1, grid_size, grid_size, 384).permute(0, 3, 1, 2),
            scale_factor=(
                (patch_width + 0.1) / grid_size,
                (patch_height + 0.1) / grid_size,
            ),
            mode="bicubic",
            antialias=False,
        )
        patches = patches.permute(0, 2, 3, 1).view(1, -1, 384)
        return torch.cat([position[:, :1], patches], dim=1).to(tokens.dtype)

    def forward(self, images):
        _, _, width, height = images.shape
        tokens = self.patch_embed(images)
        tokens = torch.cat(
            [self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1
        )
        tokens = tokens + self.interpolate_pos_encoding(tokens, width, height)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]
