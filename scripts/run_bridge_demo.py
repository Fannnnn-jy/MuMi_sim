"""本文档用于测试 mujoco 与 blender 之间的桥接功能，演示在 mujoco 中运行一个环境，并将其渲染结果传递给 blender 进行渲染。"""

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.pipeline import BridgeRunConfig, run_bridge_demo
from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig

# 夹爪灯光硬编码配置：
# - GRIPPER_LIGHT_TYPE: "POINT" 为全向点光源；"SPOT" 为射灯。
# - GRIPPER_LIGHT_ENERGY: 灯光强度，越大红 / 蓝光照越明显。
# - GRIPPER_LIGHT_SIZE: Blender shadow_soft_size，控制等效发光半径和阴影软硬；不是可见箭头大小。
# - GRIPPER_SPOT_SIZE: 射灯锥角，单位为弧度；只在 GRIPPER_LIGHT_TYPE="SPOT" 时生效。
# - GRIPPER_SPOT_BLEND: 射灯边缘柔和度，0 边缘硬，1 过渡最柔和；只在 SPOT 时生效。
# - SHOW_GRIPPER_LIGHT_MARKER: 是否渲染红 / 蓝箭头标记。箭头尾部中心是光源点，箭头尖端指向照射方向。
GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 0.55
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = True
# robosuite Lift reset 会用 z_offset=0.01 让 cube 初始悬空 1cm。
# random demo 初始帧也先推进 MuJoCo，让 Blender 和 MuJoCo 从落稳后的同一个状态开始。
SETTLE_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robosuite-to-Blender bridge demo.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/bridge_demo")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=0.35)
    parser.add_argument("--action-repeat", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--robot-asset-dir", type=str, default="assets/robot_asset/franka_emika_panda")
    parser.add_argument("--env-name", type=str, default="Lift")
    parser.add_argument("--robots", type=str, default="Panda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = BridgeRunConfig(
        steps=args.steps,
        render_every=args.render_every,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        samples=args.samples,
        seed=args.seed,
        action_scale=args.action_scale,
        action_repeat=args.action_repeat,
        no_render=args.no_render,
        robot_asset_dir=args.robot_asset_dir,
        settle_steps=SETTLE_STEPS,
        gripper_light_type=GRIPPER_LIGHT_TYPE,
        gripper_light_energy=GRIPPER_LIGHT_ENERGY,
        gripper_light_size=GRIPPER_LIGHT_SIZE,
        gripper_spot_size=GRIPPER_SPOT_SIZE,
        gripper_spot_blend=GRIPPER_SPOT_BLEND,
        show_gripper_light_marker=SHOW_GRIPPER_LIGHT_MARKER,
    )
    env_config = RobosuiteEnvConfig(env_name=args.env_name, robots=args.robots)
    summary = run_bridge_demo(run_config, env_config)
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
