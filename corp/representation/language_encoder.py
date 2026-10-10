"""T5 language tokens u and masked pooled embedding u_bar (Section 3.1)."""

from pathlib import Path

import torch
from torch import nn


class LanguageEncoder(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int) -> None:
        super().__init__()
        from transformers import T5Config, T5EncoderModel, T5TokenizerFast

        model_path = (
            Path(__file__).resolve().parents[1] / "assets" / "t5_efficient_tiny"
        )
        self.max_length = max_length
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_path, local_files_only=True
        )
        config = T5Config.from_pretrained(model_path, local_files_only=True)
        self.backbone = T5EncoderModel(config)
        self.norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, hidden_dim)

    def forward(
        self, instruction: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [instruction] if isinstance(instruction, str) else instruction
        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or not all(isinstance(t, str) for t in texts)
        ):
            raise TypeError(
                "instruction must be a string or a non-empty sequence of strings"
            )
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.projection.weight.device)
            for key, value in encoded.items()
        }
        tokens = self.projection(self.norm(self.backbone(**encoded).last_hidden_state))
        mask = encoded["attention_mask"].to(dtype=tokens.dtype).unsqueeze(-1)
        tokens = tokens * mask
        pooled_language = tokens.sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0))
        return tokens, pooled_language
