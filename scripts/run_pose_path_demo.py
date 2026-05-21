#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.pipeline import PosePathRunConfig, run_pose_path_demo
from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig

# ------------------------------ 配置项 ---------------------------------
# 最小 policy / controller demo：
# pipeline 生成 EEF 笛卡尔路径，robosuite BASIC composite controller 负责底层 operational-space 跟踪。

OUTPUT_DIR = "outputs/pose_path_demo"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
SEED = 0
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
SETTLE_STEPS = 0
NO_RENDER = False
USE_GPU = False
RENDER_EVERY = 50

CONTROLLER = "BASIC"
# WAYPOINT_STEPS = 40
POSITION_GAIN = 12.0
MAX_POSITION_DELTA = 0.1
GRIPPER_ACTION = -1.0
HORIZON = 400

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 1.50
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False

# ------------------------------ 配置项 ---------------------------------


def main() -> None:
    run_config = PosePathRunConfig(
        output_dir=OUTPUT_DIR,
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        camera_name=CAMERA_NAME,
        seed=SEED,
        robot_asset_dir=ROBOT_ASSET_DIR,
        settle_steps=SETTLE_STEPS,
        no_render=NO_RENDER,
        use_gpu=USE_GPU,
        render_every=RENDER_EVERY,
        controller=CONTROLLER,
        # waypoint_steps=WAYPOINT_STEPS,
        position_gain=POSITION_GAIN,
        max_position_delta=MAX_POSITION_DELTA,
        gripper_action=GRIPPER_ACTION,
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
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    summary = run_pose_path_demo(run_config, env_config)
    print(json.dumps({
        "metadata_path": summary["metadata_path"],
        "summary_path": str(Path(OUTPUT_DIR) / "summary.json"),
        "rendered_frames": summary["rendered_frames"],
    }, indent=2))


if __name__ == "__main__":
    main()
