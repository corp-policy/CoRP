"""DiT action generator p_theta and Euler integration (Section 3.1, Eq. 6)."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .representation.extractor import Representation


class TimeEmbedding(nn.Module):
    """Embed the interpolation time tau with sinusoids followed by an MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=tau.device, dtype=tau.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = tau[:, None] * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(embedding)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def project_qkv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, length, _ = tokens.shape
        qkv = self.qkv(tokens).view(
            batch_size, length, 3, self.num_heads, self.hidden_dim // self.num_heads
        )
        return qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

    def attend(self, query, key, value, mask=None) -> torch.Tensor:
        tokens = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        tokens = (
            tokens.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[2], self.hidden_dim)
        )
        return self.projection(tokens)


class AdaptiveLayerNorm(nn.Module):
    """Language-conditioned adaLN scale and shift for the action tokens."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, rank),
            nn.SiLU(),
            nn.Linear(rank, hidden_dim * 4),
        )

    def forward(self, pooled_language: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net(pooled_language).chunk(4, dim=-1)

    @staticmethod
    def modulate(tokens, scale, shift) -> torch.Tensor:
        return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass(frozen=True)
class PrefixCache:
    key: torch.Tensor
    value: torch.Tensor
    adaln: tuple[torch.Tensor, ...]


class DiTBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, mlp_ratio: float, adaln_rank: int
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = SelfAttention(hidden_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.adaln = AdaptiveLayerNorm(hidden_dim, adaln_rank)

    def encode_prefix(
        self, tokens, pooled_language, mask
    ) -> tuple[torch.Tensor, PrefixCache]:
        adaln = self.adaln(pooled_language)
        query, key, value = self.attention.project_qkv(self.attention_norm(tokens))
        tokens = tokens + self.attention.attend(query, key, value, mask)
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens, PrefixCache(key, value, adaln)

    def forward(self, tokens: torch.Tensor, prefix: PrefixCache) -> torch.Tensor:
        scale_attn, shift_attn, scale_mlp, shift_mlp = prefix.adaln
        attention_input = self.adaln.modulate(
            self.attention_norm(tokens), scale_attn, shift_attn
        )
        query, key, value = self.attention.project_qkv(attention_input)
        key = torch.cat([prefix.key, key], dim=2)
        value = torch.cat([prefix.value, value], dim=2)
        tokens = tokens + self.attention.attend(query, key, value)
        mlp_input = self.adaln.modulate(self.mlp_norm(tokens), scale_mlp, shift_mlp)
        return tokens + self.mlp(mlp_input)


class DiTActionGenerator(nn.Module):
    """Cache z_t once, then integrate the action velocity on tau=n/N."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = config["action_generator"]
        hidden_dim = config["hidden_dim"]
        self.action_dim = config["action_dim"]
        self.action_horizon = config["action_horizon"]
        self.num_sample_steps = cfg["num_sample_steps"]
        self.action_projection = nn.Linear(self.action_dim, hidden_dim)
        self.action_position = nn.Parameter(
            torch.zeros(1, cfg["max_horizon"], hidden_dim)
        )
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim, cfg["num_heads"], cfg["mlp_ratio"], cfg["adaln_rank"]
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_projection = nn.Linear(hidden_dim, self.action_dim)

    def prepare_prefix(self, representation: Representation) -> list[PrefixCache]:
        # Prefix tokens see only z_t; action tokens see z_t and the whole chunk.
        # Keeping these paths separate implements the block attention mask.
        tokens = representation.tokens
        mask = torch.ones(
            tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
        )[None, None]
        caches = []
        for block in self.blocks:
            tokens, cache = block.encode_prefix(
                tokens, representation.pooled_language, mask
            )
            caches.append(cache)
        return caches

    def forward(
        self, action: torch.Tensor, tau: torch.Tensor, prefix: list[PrefixCache]
    ) -> torch.Tensor:
        tokens = (
            self.action_projection(action)
            + self.action_position[:, : action.shape[1]]
            + self.time_embedding(tau).unsqueeze(1)
        )
        for block, cache in zip(self.blocks, prefix, strict=True):
            tokens = block(tokens, cache)
        return self.velocity_projection(self.output_norm(tokens))

    @torch.no_grad()
    def sample(
        self, representation: Representation, num_steps: int | None = None
    ) -> torch.Tensor:
        steps = self.num_sample_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        action = torch.randn(
            (representation.tokens.shape[0], self.action_horizon, self.action_dim),
            device=representation.tokens.device,
            dtype=representation.action_dtype,
        )
        prefix = self.prepare_prefix(representation)
        for step in range(steps):
            tau = torch.full(
                (action.shape[0],),
                step / float(steps),
                device=action.device,
                dtype=action.dtype,
            )
            action = action + self(action, tau, prefix) * (1.0 / float(steps))
        return action
