"""Learned-query vision-language fusion with FiLM modulation (Eq. 2)."""

import math

import torch
from torch import nn


class CrossAttentionResampler(nn.Module):
    """Compress each view to K tokens using queries conditioned on u_bar."""

    def __init__(
        self,
        *,
        view_names: list[str],
        hidden_dim: int,
        tokens_per_view: int,
        num_heads: int,
        film_rank: int,
    ) -> None:
        super().__init__()
        self.view_names = tuple(view_names)
        self.input_projection = nn.Linear(384, hidden_dim)
        self.queries = nn.Parameter(
            torch.randn(len(view_names), tokens_per_view, hidden_dim)
            / math.sqrt(hidden_dim)
        )
        self.view_embedding = nn.Parameter(
            torch.randn(len(view_names), 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Sequential(
            nn.Linear(hidden_dim, film_rank),
            nn.SiLU(),
            nn.Linear(film_rank, 2 * hidden_dim),
        )

    def forward(
        self, patch_tokens: dict[str, torch.Tensor], pooled_language: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.film(pooled_language).unsqueeze(1).chunk(2, dim=-1)
        batch_size = pooled_language.shape[0]
        visual_tokens = []
        for index, name in enumerate(self.view_names):
            patches = self.input_projection(patch_tokens[name]) + self.view_embedding[
                index
            ].unsqueeze(0)
            queries = self.queries[index].unsqueeze(0).expand(batch_size, -1, -1)
            queries = queries * (1.0 + gamma) + beta
            tokens, _ = self.cross_attention(
                queries, patches, patches, need_weights=False
            )
            visual_tokens.append(self.norm(tokens))
        return torch.cat(visual_tokens, dim=1)
