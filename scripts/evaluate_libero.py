#!/usr/bin/env python3
"""Evaluate the released CoRP policy with the fixed LIBERO inference protocol."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from corp import load_policy

ROOT = Path(__file__).resolve().parents[1]

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
CAMERA_SIZE = 256
SETTLING_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def configure_libero(args) -> None:
    """Resolve simulator assets without requiring demonstration data."""
    import yaml

    source = args.libero_src.expanduser().resolve()
    benchmark_root = source / "libero" / "libero"
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"LIBERO source not found: {source}")
    sys.path.insert(0, str(source))
    config_dir = args.libero_config_path.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        # LIBERO requires this config field even though evaluation never reads it.
        (config_dir / "datasets").mkdir(exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(benchmark_root),
                    "bddl_files": str(benchmark_root / "bddl_files"),
                    "init_states": str(benchmark_root / "init_files"),
                    "assets": str(benchmark_root / "assets"),
                    "datasets": str(config_dir / "datasets"),
                }
            )
        )
    else:
        existing = yaml.safe_load(config_path.read_text())
        if Path(existing["benchmark_root"]).resolve() != benchmark_root:
            raise ValueError(
                "LIBERO config points to another checkout; select a new --libero-config-path"
            )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.render_gpu))
    if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        vendor = config_dir / "nvidia_egl.json"
        vendor.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.0",
                    "ICD": {"library_path": "libEGL_nvidia.so.0"},
                }
            )
        )
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
    for name, value in {"int": int, "float": float, "bool": bool, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def state_from_observation(observation: dict, normalizer) -> np.ndarray:
    rotation = (
        Rotation.from_quat(np.asarray(observation["robot0_eef_quat"], dtype=np.float64))
        .as_rotvec()
        .astype(np.float32)
    )
    state = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            rotation,
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)
    centers = normalizer.state_angle_center.numpy()
    for index in range(3, 6):
        value, center = float(state[index]), float(centers[index])
        while value - center > np.pi:
            value -= 2.0 * np.pi
        while center - value > np.pi:
            value += 2.0 * np.pi
        state[index] = value
    return state


def policy_observation(observation: dict, language: str, policy) -> dict:
    images = {}
    for name, camera in (
        ("agentview", "agentview_image"),
        ("eye_in_hand", "robot0_eye_in_hand_image"),
    ):
        # The released policy expects a 180-degree rotation before bilinear resize.
        image = np.ascontiguousarray(
            np.asarray(observation[camera], dtype=np.uint8)[::-1, ::-1]
        )
        size = policy.config["image_size"]
        if image.shape[:2] != (size, size):
            image = np.asarray(
                Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
            )
        images[name] = torch.from_numpy(np.ascontiguousarray(image).copy()).unsqueeze(0)
    return {
        "images": images,
        "state": torch.from_numpy(
            state_from_observation(observation, policy.normalizer)
        ).unsqueeze(0),
        "instruction": [language],
    }


def is_success(env, info, reward, done) -> bool:
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    checker = getattr(env, "check_success", None)
    return bool(checker()) if checker is not None else bool(done) or float(reward) > 0.0


def task_indices(value: str | None, count: int) -> list[int]:
    selected = []
    if value:
        for part in value.split(","):
            if "-" in part:
                start, end = map(int, part.split("-", 1))
                selected.extend(range(start, end + 1))
            else:
                selected.append(int(part))
    else:
        selected = list(range(count))
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(i < 0 or i >= count for i in selected)
    ):
        raise ValueError(f"tasks must be unique indices in 0..{count - 1}")
    return selected


def evaluate_suite(
    args, suite_name, policy, benchmark, get_libero_path, env_class
) -> dict:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    records = []
    started = time.monotonic()
    device = torch.device(args.device)
    result_path = args.output_dir / f"{suite_name}.jsonl"
    with result_path.open("w") as output:
        for task_id in task_indices(args.tasks, suite.n_tasks):
            task = suite.get_task(task_id)
            bddl = (
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )
            states_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            # Only trusted local LIBERO initial states need the NumPy pickle loader.
            initial_states = torch.load(states_path, weights_only=False)
            if len(initial_states) < args.episodes_per_task:
                raise ValueError(
                    f"{suite_name}/{task_id} has only {len(initial_states)} initial states"
                )
            env = env_class(
                bddl_file_name=str(bddl),
                camera_heights=CAMERA_SIZE,
                camera_widths=CAMERA_SIZE,
                render_gpu_device_id=args.render_gpu,
            )
            try:
                for episode in range(args.episodes_per_task):
                    seed = args.seed + task_id * 1000 + episode
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    env.seed(seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[episode])
                    success, done, steps, settling_steps = False, False, 0, 0
                    for _ in range(SETTLING_STEPS):
                        observation, reward, done, info = env.step(DUMMY_ACTION)
                        settling_steps += 1
                        success = is_success(env, info, reward, done)
                        if success or done:
                            break
                    while not success and not done and steps < MAX_STEPS[suite_name]:
                        batch = policy_observation(observation, task.language, policy)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and args.precision == "bf16",
                        ):
                            chunk = policy.predict(batch)[0].float().cpu().numpy()
                        for action in chunk:
                            observation, reward, done, info = env.step(
                                np.clip(action, -1.0, 1.0)
                            )
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= MAX_STEPS[suite_name]:
                                break
                    record = {
                        "suite": suite_name,
                        "task_id": task_id,
                        "episode_id": episode,
                        "seed": seed,
                        "success": success,
                        "policy_steps": steps,
                        "settling_steps": settling_steps,
                        "language": task.language,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    records.append(record)
                    print(
                        f"{suite_name} task={task_id} episode={episode} success={int(success)} steps={steps}",
                        flush=True,
                    )
            finally:
                env.close()
    successes = sum(record["success"] for record in records)
    return {
        "suite": suite_name,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints" / "corp_libero.pt"
    )
    parser.add_argument("--suite", choices=["all", *MAX_STEPS], default="all")
    parser.add_argument("--tasks", help="Task indices, e.g. 0,2,5-9. Default: all ten.")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument(
        "--libero-src", type=Path, default=ROOT / "third_party" / "LIBERO"
    )
    parser.add_argument(
        "--libero-config-path", type=Path, default=ROOT / ".cache" / "libero"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero"))
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 50:
        parser.error("--episodes-per-task must be in 1..50")
    if args.seed < 0 or args.seed > 2**32 - 1 - 9049:
        parser.error("--seed must leave room for all per-episode NumPy seeds")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suites = list(MAX_STEPS) if args.suite == "all" else [args.suite]
    summaries = [
        evaluate_suite(
            args, name, policy, benchmark, get_libero_path, OffScreenRenderEnv
        )
        for name in suites
    ]
    episodes = sum(s["episodes"] for s in summaries)
    successes = sum(s["successes"] for s in summaries)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "precision": args.precision,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "protocol": {
            "settling_steps": SETTLING_STEPS,
            "camera_size": CAMERA_SIZE,
            "camera_rotation": 180,
            "max_steps": MAX_STEPS,
            "action_horizon": policy.config["action_horizon"],
            "euler_steps": policy.config["action_generator"]["num_sample_steps"],
        },
        "suites": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
