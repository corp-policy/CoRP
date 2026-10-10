#!/usr/bin/env python3
"""Verify the release checksum, strict model loading, and action generation."""

import argparse
import hashlib
from pathlib import Path

import torch

from corp import load_policy

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        parser.error("BF16 verification requires a CUDA device")
    checkpoint = ROOT / "checkpoints" / "corp_libero.pt"
    expected = (checkpoint.parent / "SHA256SUMS").read_text().split()[0]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit("checkpoint checksum mismatch")
    torch.manual_seed(7)
    torch.set_num_threads(4)
    policy = load_policy(checkpoint, device=args.device)
    observation = {
        "images": {
            name: torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
            for name in policy.config["view_names"]
        },
        "state": torch.zeros(1, 8),
        "instruction": ["pick up the black bowl and place it on the plate"],
    }
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=args.precision == "bf16",
    ):
        actions = policy.predict(observation)
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    if (
        parameters != 48_689_159
        or actions.shape != (1, 12, 7)
        or not torch.isfinite(actions).all()
    ):
        raise SystemExit("unexpected model size or action output")
    print(
        f"device={args.device} precision={args.precision} "
        f"parameters={parameters} action_shape={tuple(actions.shape)}"
    )
    print(f"sha256={digest.hexdigest()}")
    print("smoke_ok")


if __name__ == "__main__":
    main()
