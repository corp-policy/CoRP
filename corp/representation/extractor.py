"""Representation extractor f_phi: z_t = [r_t^1; ...; r_t^V; u; e_s(s_t)]."""

from dataclasses import dataclass

import torch
from torch import nn

from .cross_attention_resampler import CrossAttentionResampler
from .language_encoder import LanguageEncoder
from .state_encoder import StateEncoder
from .visual_encoder import VisualEncoder


@dataclass(frozen=True)
class Representation:
    tokens: torch.Tensor
    pooled_language: torch.Tensor
    action_dtype: torch.dtype


class RepresentationExtractor(nn.Module):
    """Encode visual observations, language, and state in the order of Eq. 3."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = config["representation_extractor"]
        self.view_names = tuple(config["view_names"])
        self.visual_encoder = VisualEncoder()
        self.language_encoder = LanguageEncoder(
            config["hidden_dim"], cfg["language_encoder"]["max_length"]
        )
        self.cross_attention_resampler = CrossAttentionResampler(
            view_names=config["view_names"],
            hidden_dim=config["hidden_dim"],
            **cfg["cross_attention_resampler"],
        )
        self.state_encoder = StateEncoder(config["state_dim"], config["hidden_dim"])

    def forward(
        self, images: dict[str, torch.Tensor], state: torch.Tensor, instruction
    ) -> Representation:
        batch_size = state.shape[0]
        patches = self.visual_encoder(
            torch.cat([images[name] for name in self.view_names], dim=0)
        )
        patch_tokens = {
            name: patches[index * batch_size : (index + 1) * batch_size]
            for index, name in enumerate(self.view_names)
        }
        language_tokens, pooled_language = self.language_encoder(instruction)
        if language_tokens.shape[0] != batch_size:
            raise ValueError("instruction count must match the observation batch size")
        visual_tokens = self.cross_attention_resampler(patch_tokens, pooled_language)
        visual_language = torch.cat(
            [visual_tokens, language_tokens.to(visual_tokens.dtype)], dim=1
        )
        state_token = self.state_encoder(state)
        # Preserve the action-noise precision before the state token promotes z_t.
        return Representation(
            tokens=torch.cat([visual_language, state_token], dim=1),
            pooled_language=pooled_language,
            action_dtype=visual_language.dtype,
        )
