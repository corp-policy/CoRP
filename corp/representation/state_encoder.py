"""Proprioceptive state encoder e_s in Eq. 3."""

import torch
from torch import nn


class StateEncoder(nn.Module):
    """The released e_s is an affine projection plus a learned token offset."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(state_dim, hidden_dim)
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(state).unsqueeze(1) + self.token.expand(
            state.shape[0], -1, -1
        )
