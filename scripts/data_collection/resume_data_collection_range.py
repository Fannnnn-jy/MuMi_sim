#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LumiGrasp.scripts.data_collection.run_data_collection import CUBE_RANGES, _build_config, _build_env_config
from lumigrasp.bridge.robosuite_env import build_robosuite_env, close_env
from lumigrasp.data_collection.pipeline import (
    _build_sample_renderer,
    _run_one_capture,
    _sample_config,
    cube_range_folder_name,
)
from lumigrasp.data_collection.sample import (
    build_sample_placement_initializer,
    sample_eef_pose,
    sample_robot_initialization_noise,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume one PickCube data collection range without overwriting other ranges.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range-index", type=int, help="Index in scripts.run_data_collection.CUBE_RANGES.")
    group.add_argument("--range-name", help="Folder/name of the cube range to resume.")
    parser.add_argument("--target", type=int, default=12000, help="Target complete samples for this range.")
    parser.add_argument(
        "--use-gpu",
        "--usegpu",
        action="store_true",
        default=None,
        help="Use GPU rendering for Blender outputs.",
    )
    parser.add_argument(
        "--cpu-render",
        action="store_false",
        dest="use_gpu",
        help="Force CPU rendering for Blender outputs.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        "--cudavisibledevices",
        help="Value to set for CUDA_VISIBLE_DEVICES before environment and renderer setup.",
    )
    parser.add_argument(
        "--no-rng-skip",
        action="store_true",
        help="Do not advance the EEF RNG past existing complete samples before resuming.",
    )
    return parser.parse_args()


def _select_range(args: argparse.Namespace):
    if args.range_index is not None:
        return CUBE_RANGES[args.range_index], args.range_index
    for index, cube_range in enumerate(CUBE_RANGES):
        if cube_range_folder_name(cube_range) == args.range_name or cube_range.name == args.range_name:
            return cube_range, index
    names = ", ".join(cube_range_folder_name(cube_range) for cube_range in CUBE_RANGES)
    raise SystemExit(f"Unknown range name {args.range_name!r}. Available: {names}")


def _complete_count(range_dir: Path) -> int:
    return sum(1 for _ in range_dir.glob("sample_*/metadata.json"))


def main() -> None:
    args = _parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cube_range, worker_index = _select_range(args)
    config = _build_config()
    if args.use_gpu is not None:
        config = replace(config, use_gpu=args.use_gpu)
    env_config = _build_env_config()
    range_dir = Path(config.output_dir) / cube_range_folder_name(cube_range)
    range_dir.mkdir(parents=True, exist_ok=True)

    complete = _complete_count(range_dir)
    if complete >= args.target:
        print(json.dumps({"range_dir": str(range_dir), "complete": complete, "target": args.target, "remaining": 0}, indent=2))
        return

    seed = config.seed + worker_index * 100003
    rng = np.random.default_rng(seed)
    sample_config = _sample_config(config, cube_range, seed)
    if not args.no_rng_skip:
        for _ in range(complete):
            sample_eef_pose(rng, sample_config)

    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_config)
    renderer = None
    generated: list[str] = []
    attempts = 0
    max_attempts = max(args.target - complete, (args.target - complete) * config.max_attempts_per_sample)
    start_time = time.time()

    try:
        if config.render_blender:
            renderer = _build_sample_renderer(sample_config)
            renderer.setup_scene()

        next_index = complete
        while next_index < args.target and attempts < max_attempts:
            attempts += 1
            result = _run_one_capture(
                env=env,
                renderer=renderer,
                rng=rng,
                config=config,
                sample_config=sample_config,
                cube_range=cube_range,
                output_dir=range_dir,
                index=next_index,
            )
            if result is None:
                print(f"[{cube_range_folder_name(cube_range)}] retry index {next_index:06d}")
                continue
            generated.append(result["sample_dir"])
            next_index += 1
            done = _complete_count(range_dir)
            print(f"[{cube_range_folder_name(cube_range)}] complete {done}/{args.target}")
    finally:
        close_env(env)

    final_complete = _complete_count(range_dir)
    summary = {
        "range_dir": str(range_dir),
        "cube_range": asdict(cube_range),
        "worker_index": worker_index,
        "seed": seed,
        "start_complete": complete,
        "target": args.target,
        "attempts": attempts,
        "generated": len(generated),
        "final_complete": final_complete,
        "use_gpu": config.use_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.time() - start_time,
        "generated_samples": generated,
    }
    summary_path = range_dir / "resume_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **{k: summary[k] for k in ("final_complete", "target", "attempts", "generated")}}, indent=2))


if __name__ == "__main__":
    main()
