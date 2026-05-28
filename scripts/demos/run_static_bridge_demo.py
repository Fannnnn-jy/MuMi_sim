#!/usr/bin/env python
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.pipeline_demo import StaticBridgeRunConfig, run_static_bridge_demo
from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig

# ------------------------------ 配置项 ---------------------------------
# Junyi's notes:
# 本脚本用于展示 demo，为了方便快速交互调整参数参考效果，采用脚本硬编码方法
# 正式运行时实践应该改为在 configs/ 下配置 .yaml 文件
# I hate too many args in command line!

OUTPUT_DIR = "outputs/static_bridge_demo"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64 # 调低并不会显著加快 sample 速度
CAMERA_NAME = "agentview"
SEED = 42
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
USE_GPU = True
# robosuite Lift reset 会用 z_offset=0.01 让 cube 初始悬空 1cm。
# 为了渲染“落稳后的同一个 MuJoCo 物理世界”，先推进 MuJoCo，再同时采集 MuJoCo / Blender。
SETTLE_STEPS = 0
RENDER_DEPTH = True

# 夹爪灯光硬编码配置：
# - GRIPPER_LIGHT_TYPE: "POINT" 为全向点光源；"SPOT" 为射灯。
# - GRIPPER_LIGHT_ENERGY: 灯光强度，越大红 / 蓝光照越明显。
# - GRIPPER_LIGHT_SIZE: Blender shadow_soft_size，控制等效发光半径和阴影软硬；不是可见箭头大小。
# - GRIPPER_SPOT_SIZE: 射灯锥角，单位为弧度；只在 GRIPPER_LIGHT_TYPE="SPOT" 时生效。
# - GRIPPER_SPOT_BLEND: 射灯边缘柔和度，0 边缘硬，1 过渡最柔和；只在 SPOT 时生效。
# - SHOW_GRIPPER_LIGHT_MARKER: 是否渲染红 / 蓝箭头标记。箭头尾部中心是光源点，箭头尖端指向照射方向。
GRIPPER_LIGHT_TYPE = "SPOT" # "POINT" or "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.00
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False

# ------------------------------ 配置项 ---------------------------------

def main() -> None:
    run_config = StaticBridgeRunConfig(
        output_dir=OUTPUT_DIR,
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        camera_name=CAMERA_NAME,
        seed=SEED,
        robot_asset_dir=ROBOT_ASSET_DIR,
        use_gpu=USE_GPU,
        settle_steps=SETTLE_STEPS,
        render_depth=RENDER_DEPTH,
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
    )
    summary = run_static_bridge_demo(run_config, env_config)
    print(json.dumps({
        "mujoco_path": summary["mujoco_path"],
        "blender_path": summary["blender_path"],
        "metadata_path": str(Path(OUTPUT_DIR) / "metadata.json"),
        "camera": summary["camera"],
        "depth_outputs": summary.get("depth_outputs"),
    }, indent=2))


if __name__ == "__main__":
    main()
