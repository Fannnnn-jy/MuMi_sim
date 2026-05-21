#!/usr/bin/env python
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import numpy as np
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig
from lumigrasp.data_collection.pipeline import (
    CubeRangeConfig,
    DataCollectionConfig,
    run_data_collection,
    run_data_collection_worker,
)


OUTPUT_DIR = "/home/ubuntu/Junyi/LumiGrasp/data/pickcube-v1"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
SEED = 42
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
SETTLE_STEPS = 100
USE_GPU = False
RENDER_BLENDER = True
RENDER_MUJOCO = True
RENDER_DEPTH = True
HORIZON = 200
CONTROLLER = "BASIC"
N_RENDER_PROCESSES = 4

# Random initial EEF pose bounds in world frame, solved by IK before capture.
XYZ_LOW = (-0.10, -0.10, 0.88)
XYZ_HIGH = (0.10, 0.10, 1.02)
YAW_LOW = -np.pi
YAW_HIGH = np.pi

# Each entry becomes one output subfolder. Folder names should identify cube xy ranges.
# CUBE_RANGES = [
#     CubeRangeConfig(
#         name="cube_x_-005_005_y_-005_005",
#         object_x_range=(-0.05, 0.05),
#         object_y_range=(-0.05, 0.05),
#         object_rotation=(-np.pi, np.pi),
#         object_reference_pos=(0.0, 0.0, 0.8),
#         object_z_offset=0.01,
#     ),
# ]
N_SAMPLES = 12000
CUBE_RANGES = [
    CubeRangeConfig(
        name=f"cube_x_{x0:+.2f}_{x1:+.2f}_y_{y0:+.2f}_{y1:+.2f}".replace("+", "").replace(".", "p").replace("-", "m"),
        object_x_range=(x0, x1),
        object_y_range=(y0, y1),
        object_rotation=(-np.pi, np.pi),
        object_reference_pos=(0.0, 0.0, 0.8),
        object_z_offset=0.01,
    )
    for x0, x1 in zip(np.linspace(-0.1, 0.1, 3)[:-1], np.linspace(-0.1, 0.1, 3)[1:])
    for y0, y1 in zip(np.linspace(-0.1, 0.1, 3)[:-1], np.linspace(-0.1, 0.1, 3)[1:])
]

XY_THRESHOLD = 0.01
THETA_THRESHOLD = 0.10
XY_THRESHOLD_NOISE = 0.003
THETA_THRESHOLD_NOISE = 0.0
PREGRASP_CLEARANCE = 0.08
GRASP_CUBE_HALF_SIZE = 0.021
MAX_DELTA_POS = 0.5
POSITION_GAIN = 12.0
ORIENTATION_GAIN = 2.0
MAX_ORIENTATION_DELTA = 0.25
GRIPPER_ACTION_OPEN = 0.0

IK_MAX_ITERS = 250
IK_POSITION_TOLERANCE = 0.003
IK_ORIENTATION_TOLERANCE = 0.03
IK_DAMPING = 1e-3
IK_STEP_SIZE = 0.6
IK_LAST_JOINT_YAW_ONLY = True
IK_YAW_TOLERANCE = 0.03

ROBOT_INITIALIZATION_NOISE_TYPE = "uniform"
ROBOT_INITIALIZATION_NOISE_MAGNITUDE = 0.0

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.5
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False


def _build_config() -> DataCollectionConfig:
    return DataCollectionConfig(
        output_dir=OUTPUT_DIR,
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        camera_name=CAMERA_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        robot_asset_dir=ROBOT_ASSET_DIR,
        settle_steps=SETTLE_STEPS,
        use_gpu=USE_GPU,
        render_blender=RENDER_BLENDER,
        render_mujoco=RENDER_MUJOCO,
        render_depth=RENDER_DEPTH,
        xyz_low=XYZ_LOW,
        xyz_high=XYZ_HIGH,
        yaw_low=YAW_LOW,
        yaw_high=YAW_HIGH,
        ik_max_iters=IK_MAX_ITERS,
        ik_position_tolerance=IK_POSITION_TOLERANCE,
        ik_orientation_tolerance=IK_ORIENTATION_TOLERANCE,
        ik_damping=IK_DAMPING,
        ik_step_size=IK_STEP_SIZE,
        ik_last_joint_yaw_only=IK_LAST_JOINT_YAW_ONLY,
        ik_yaw_tolerance=IK_YAW_TOLERANCE,
        xy_threshold=XY_THRESHOLD,
        theta_threshold=THETA_THRESHOLD,
        xy_threshold_noise=XY_THRESHOLD_NOISE,
        theta_threshold_noise=THETA_THRESHOLD_NOISE,
        pregrasp_clearance=PREGRASP_CLEARANCE,
        grasp_cube_half_size=GRASP_CUBE_HALF_SIZE,
        max_delta_pos=MAX_DELTA_POS,
        position_gain=POSITION_GAIN,
        orientation_gain=ORIENTATION_GAIN,
        max_orientation_delta=MAX_ORIENTATION_DELTA,
        gripper_action_open=GRIPPER_ACTION_OPEN,
        robot_initialization_noise_type=ROBOT_INITIALIZATION_NOISE_TYPE,
        robot_initialization_noise_magnitude=ROBOT_INITIALIZATION_NOISE_MAGNITUDE,
        gripper_light_type=GRIPPER_LIGHT_TYPE,
        gripper_light_energy=GRIPPER_LIGHT_ENERGY,
        gripper_light_size=GRIPPER_LIGHT_SIZE,
        gripper_spot_size=GRIPPER_SPOT_SIZE,
        gripper_spot_blend=GRIPPER_SPOT_BLEND,
        show_gripper_light_marker=SHOW_GRIPPER_LIGHT_MARKER,
    )


def _build_env_config() -> RobosuiteEnvConfig:
    return RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )


def _run_worker(args: tuple[int, DataCollectionConfig, CubeRangeConfig, RobosuiteEnvConfig]) -> dict:
    worker_index, config, cube_range, env_config = args
    return run_data_collection_worker(config, cube_range, env_config=env_config, worker_index=worker_index)


def main() -> None:
    config = _build_config()
    env_config = _build_env_config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if N_RENDER_PROCESSES <= 1 or len(CUBE_RANGES) <= 1:
        summary = run_data_collection(config, list(CUBE_RANGES), env_config=env_config)
    else:
        tasks = [(index, config, cube_range, env_config) for index, cube_range in enumerate(CUBE_RANGES)]
        with mp.get_context("spawn").Pool(processes=N_RENDER_PROCESSES) as pool:
            workers = pool.map(_run_worker, tasks)
        summary = {
            "config": config.__dict__,
            "env_config": env_config.__dict__,
            "cube_ranges": [cube_range.__dict__ for cube_range in CUBE_RANGES],
            "workers": workers,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)

    print(json.dumps({
        "summary_path": summary["summary_path"],
        "output_dir": config.output_dir,
        "n_workers": len(summary["workers"]),
        "workers": [
            {
                "range_dir": worker["range_dir"],
                "summary_path": worker["summary_path"],
                "n_samples": worker["n_samples"],
                "attempts": worker["attempts"],
            }
            for worker in summary["workers"]
        ],
    }, indent=2))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
