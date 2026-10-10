"""LIBERO observation normalization and action conversion."""

from collections.abc import Mapping

import torch


class Normalizer:
    """Checkpoint-defined state z-scores and 6D motion min/max conversion."""

    def __init__(self, statistics: dict) -> None:
        if statistics.get("format") != "corp_libero_normalizer_v1":
            raise ValueError("unsupported normalization format")
        for name, size in (
            ("state_mean", 8),
            ("state_std", 8),
            ("action_min", 7),
            ("action_max", 7),
        ):
            value = torch.as_tensor(statistics[name], dtype=torch.float32).clone()
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain {size} finite values")
            setattr(self, name, value)
        # Retain the checkpoint's arithmetic for selecting an equivalent rotation vector.
        self.state_angle_center = (
            (self.state_mean - self.state_std) + (self.state_mean + self.state_std)
        ) * 0.5

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.to(device=state.device, dtype=state.dtype)
        std = self.state_std.to(device=state.device, dtype=state.dtype)
        return (state - mean) / (std + 1.0e-6)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_min.to(device=action.device, dtype=action.dtype)
        high = self.action_max.to(device=action.device, dtype=action.dtype)
        result = action.clone()
        result[..., :6] = 0.5 * (result[..., :6] + 1.0) * (high[:6] - low[:6]) + low[:6]
        return result


def prepare_observation(
    observation: dict, *, config: dict, normalizer: Normalizer, device: torch.device
) -> dict:
    """Accept batched RGB images, 8D state, and one instruction per observation."""
    state = torch.as_tensor(observation["state"], device=device, dtype=torch.float32)
    if state.ndim != 2 or state.shape[1] != config["state_dim"]:
        raise ValueError(f"state must have shape [B,{config['state_dim']}]")
    views = observation["images"]
    if not isinstance(views, Mapping) or set(views) != set(config["view_names"]):
        raise ValueError(f"images must contain exactly {config['view_names']}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = {}
    for name in config["view_names"]:
        image = torch.as_tensor(views[name], device=device)
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        expected = (state.shape[0], 3, config["image_size"], config["image_size"])
        if tuple(image.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected} or the corresponding NHWC shape"
            )
        image = (
            image.float().div_(255.0) if image.dtype == torch.uint8 else image.float()
        )
        images[name] = (image - mean) / std
    return {
        "images": images,
        "state": normalizer.normalize_state(state),
        "instruction": observation["instruction"],
    }
