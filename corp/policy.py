"""CoRP: representation extraction followed by conditional action generation."""

import torch
from torch import nn

from .action_generator import DiTActionGenerator
from .preprocessing import Normalizer, prepare_observation
from .representation import RepresentationExtractor


class CoRP(nn.Module):
    """The two-stage Compressed Representation Policy in Eq. 1."""

    def __init__(self, config: dict, normalizer: Normalizer) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.representation_extractor = RepresentationExtractor(config)
        self.action_generator = DiTActionGenerator(config)

    @torch.inference_mode()
    def predict(self, observation: dict, num_steps: int | None = None) -> torch.Tensor:
        """Return a [B,H,7] chunk in LIBERO controller units from batched observations.

        Images are uint8 [B,H,W,3] or [B,3,H,W], or floats in [0,1].
        The caller applies the simulator camera transform before this method.
        """
        if self.training:
            raise RuntimeError(
                "Call eval() before CoRP inference, or construct with load_policy()"
            )
        batch = prepare_observation(
            observation,
            config=self.config,
            normalizer=self.normalizer,
            device=next(self.parameters()).device,
        )
        representation = self.representation_extractor(**batch)
        actions = self.action_generator.sample(representation, num_steps=num_steps)
        return self.normalizer.unnormalize_action(actions)
