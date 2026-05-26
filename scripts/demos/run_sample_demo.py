#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig
from lumigrasp.data_collection.sample import SampleDemoConfig, run_sample_demo


OUTPUT_DIR = "outputs/sample_demo"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
SEED = 42
N_SAMPLES = 5
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
USE_GPU = False
RENDER_BLENDER = True
RENDER_MUJOCO = True
RENDER_DEPTH = True
HORIZON = 200

# Cartesian EEF sample bounds in world frame.
XYZ_LOW = (-0.15, -0.15, 0.88)
XYZ_HIGH = (0.15, 0.15, 1.02)
YAW_LOW = -3.141592653589793
YAW_HIGH = 3.141592653589793

OBJECT_X_RANGE = (0.0, 0.0)
OBJECT_Y_RANGE = (0.0, 0.0)
OBJECT_ROTATION = (-0.5, 0.5)
OBJECT_REFERENCE_POS = (0.15, 0.15, 0.8)
OBJECT_Z_OFFSET = 0.01

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.50
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False


def main() -> None:
    config = SampleDemoConfig(
        output_dir=OUTPUT_DIR,
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        camera_name=CAMERA_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        robot_asset_dir=ROBOT_ASSET_DIR,
        use_gpu=USE_GPU,
        render_blender=RENDER_BLENDER,
        render_mujoco=RENDER_MUJOCO,
        render_depth=RENDER_DEPTH,
        xyz_low=XYZ_LOW,
        xyz_high=XYZ_HIGH,
        yaw_low=YAW_LOW,
        yaw_high=YAW_HIGH,
        object_x_range=OBJECT_X_RANGE,
        object_y_range=OBJECT_Y_RANGE,
        object_rotation=OBJECT_ROTATION,
        object_reference_pos=OBJECT_REFERENCE_POS,
        object_z_offset=OBJECT_Z_OFFSET,
        gripper_light_type=GRIPPER_LIGHT_TYPE,
        gripper_light_energy=GRIPPER_LIGHT_ENERGY,
        gripper_light_size=GRIPPER_LIGHT_SIZE,
        gripper_spot_size=GRIPPER_SPOT_SIZE,
        gripper_spot_blend=GRIPPER_SPOT_BLEND,
        show_gripper_light_marker=SHOW_GRIPPER_LIGHT_MARKER,
    )
    env_config = RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    summary = run_sample_demo(config, env_config)
    print(json.dumps({
        "metadata_path": summary["metadata_path"],
        "n_samples": summary["n_samples"],
        "n_success": summary["n_success"],
        "samples": [
            {
                "index": item["index"],
                "sample_dir": item["sample_dir"],
                "ik": item["ik"],
                "render_outputs": {
                    key: value
                    for key, value in item["render_outputs"].items()
                    if key.endswith("_path")
                },
            }
            for item in summary["samples"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
