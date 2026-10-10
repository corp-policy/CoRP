"""Strict loading of the self-contained CoRP LIBERO release."""

import hashlib
import json
from pathlib import Path

import torch

from .policy import CoRP
from .preprocessing import Normalizer


def config_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_policy(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    config_path: str | Path | None = None,
) -> CoRP:
    """Load the adjacent config.json, verify its contract, and require every model tensor."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    config_path = (
        checkpoint.with_name("config.json")
        if config_path is None
        else Path(config_path).expanduser()
    )
    config = json.loads(config_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "corp_libero_v2":
        raise ValueError("expected a corp_libero_v2 checkpoint")
    if payload.get("config_sha256") != config_digest(config):
        raise ValueError("configuration does not match the checkpoint contract")
    policy = CoRP(config, Normalizer(payload["normalizer"]))
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(device).eval()
