#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from robosuite.utils.transform_utils import axisangle2quat, quat2mat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.mujoco_state import (  # noqa: E402
    collect_body_poses,
    default_arena_geom_names,
    default_bridge_body_names,
    default_cube_visual_geom_name,
    get_geom_state,
)
from lumigrasp.bridge.robosuite_env import (  # noqa: E402
    RobosuiteEnvConfig,
    build_robosuite_env,
    close_env,
)
from lumigrasp.bridge.utils.control_utils import (  # noqa: E402
    _bounded_orientation_delta,
    _bounded_position_delta,
    _osc_pose_action,
)
from lumigrasp.bridge.utils.coords import _orientation_delta_axis_angle, _quat_wxyz_to_mat  # noqa: E402
from lumigrasp.data_collection.pipeline import (  # noqa: E402
    _render_blender,
    _render_mujoco,
    _sample_config,
    _sync_robot_controller_goals,
    CubeRangeConfig,
    DataCollectionConfig,
)
from lumigrasp.data_collection.sample import (  # noqa: E402
    _build_sample_renderer,
    build_sample_placement_initializer,
    sample_robot_initialization_noise,
)


SAMPLE_DIR = "data/pickcube-v1/cube_x_m0p15_p0p15_y_m0p15_p0p15/sample_000004"
OUTPUT_DIR: str | None = None

WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
USE_GPU = False
RENDER_MUJOCO = True
RENDER_BLENDER = True
RENDER_DEPTH = True
RENDER_BEFORE = True

CONTROLLER = "BASIC"
HORIZON = 500
EXECUTE_MODE = "to_target"  # "to_target" or "single_action"
MAX_STEPS = 200
POSITION_GAIN = 12.0
ORIENTATION_GAIN = 2.0
MAX_DELTA_POS = 0.5
MAX_ORIENTATION_DELTA = 0.25
CLOSE_Z_THRESHOLD = 0.005
POSITION_THRESHOLD: float | None = None
THETA_THRESHOLD: float | None = None
RESTORE_QVEL = False

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.5
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False


def _load_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found under {sample_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _cube_range_from_metadata(metadata: dict[str, Any]) -> CubeRangeConfig:
    cube_range = metadata.get("cube_range") or {}
    return CubeRangeConfig(
        name=str(cube_range.get("name") or ""),
        object_x_range=tuple(cube_range.get("object_x_range", (-0.15, 0.15))),
        object_y_range=tuple(cube_range.get("object_y_range", (-0.15, 0.15))),
        object_rotation=cube_range.get("object_rotation", (-np.pi, np.pi)),
        object_reference_pos=tuple(cube_range.get("object_reference_pos", (0.0, 0.0, 0.8))),
        object_z_offset=float(cube_range.get("object_z_offset", 0.01)),
    )


def _build_sample_config(metadata: dict[str, Any], output_dir: Path) -> Any:
    cube_range = _cube_range_from_metadata(metadata)
    config = DataCollectionConfig(
        output_dir=str(output_dir),
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        camera_name=CAMERA_NAME,
        robot_asset_dir=ROBOT_ASSET_DIR,
        use_gpu=USE_GPU,
        render_mujoco=RENDER_MUJOCO,
        render_blender=RENDER_BLENDER,
        render_depth=RENDER_DEPTH,
        gripper_light_type=GRIPPER_LIGHT_TYPE,
        gripper_light_energy=GRIPPER_LIGHT_ENERGY,
        gripper_light_size=GRIPPER_LIGHT_SIZE,
        gripper_spot_size=GRIPPER_SPOT_SIZE,
        gripper_spot_blend=GRIPPER_SPOT_BLEND,
        show_gripper_light_marker=SHOW_GRIPPER_LIGHT_MARKER,
    )
    return _sample_config(config, cube_range, seed=int(metadata.get("index", 0)))


def _find_cube_free_joint(env: Any) -> str:
    for joint_name in env.sim.model.joint_names:
        if "cube" not in joint_name:
            continue
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(qpos_addr, tuple) and qpos_addr[1] - qpos_addr[0] == 7:
            return joint_name
    raise RuntimeError("Could not find cube free joint in MuJoCo model")


def _restore_gripper_state(env: Any, robot: Any, grippers: dict[str, Any]) -> None:
    for arm, gripper_state in grippers.items():
        joints = getattr(robot, "gripper_joints", {}).get(arm)
        if not joints:
            continue
        qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(gripper_state.get("qvel", []), dtype=np.float64)
        for index, joint_name in enumerate(joints):
            if index < qpos.size:
                qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
                env.sim.data.qpos[qpos_addr] = float(qpos[index])
            if RESTORE_QVEL and index < qvel.size:
                qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
                env.sim.data.qvel[qvel_addr] = float(qvel[index])


def _restore_state(env: Any, metadata: dict[str, Any]) -> None:
    state = metadata["state"]
    robot = env.robots[0]
    robot_state = state["robot_state"]["robots"][0]

    arm_qpos = np.asarray(robot_state["arm_qpos"], dtype=np.float64)
    env.sim.data.qpos[np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)] = arm_qpos
    if RESTORE_QVEL:
        env.sim.data.qvel[np.asarray(robot._ref_arm_joint_vel_indexes, dtype=int)] = np.asarray(
            robot_state.get("arm_qvel", np.zeros_like(arm_qpos)),
            dtype=np.float64,
        )
    else:
        env.sim.data.qvel[np.asarray(robot._ref_arm_joint_vel_indexes, dtype=int)] = 0.0

    _restore_gripper_state(env, robot, robot_state.get("grippers", {}))

    cube_geom = state["cube_geom"]
    cube_qpos = np.concatenate((
        np.asarray(cube_geom["position"], dtype=np.float64),
        np.asarray(cube_geom["quaternion_wxyz"], dtype=np.float64),
    ))
    env.sim.data.set_joint_qpos(_find_cube_free_joint(env), cube_qpos)
    env.sim.forward()
    _sync_robot_controller_goals(env)


def _eef_pose(env: Any, body_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    poses = collect_body_poses(env.sim, body_names)
    eef_pose = poses["gripper0_right_eef"]
    return (
        np.asarray(eef_pose.position, dtype=np.float64),
        _quat_wxyz_to_mat(eef_pose.quaternion_wxyz),
    )


def _target_from_action(env: Any, metadata: dict[str, Any], body_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gt_action = metadata["gt_action"]
    eef_pos, eef_ori = _eef_pose(env, body_names)

    target_pos = gt_action.get("target_pos")
    if target_pos is None:
        target_pos = eef_pos + np.asarray(gt_action["delta_pos"], dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    target_ori = gt_action.get("target_ori")
    if target_ori is not None:
        return target_pos, np.asarray(target_ori, dtype=np.float64)

    orientation_delta = np.asarray(gt_action.get("orientation_delta_axis_angle", [0.0, 0.0, 0.0]), dtype=np.float64)
    delta_mat = quat2mat(axisangle2quat(orientation_delta))
    return target_pos, delta_mat @ eef_ori


def _pregrasp_errors(
    env: Any,
    body_names: list[str],
    target_pos: np.ndarray,
    target_ori: np.ndarray,
) -> dict[str, float]:
    eef_pos, eef_ori = _eef_pose(env, body_names)
    orientation_error = _orientation_delta_axis_angle(target_ori, eef_ori)
    return {
        "position_error": float(np.linalg.norm(target_pos - eef_pos)),
        "xy_error": float(np.linalg.norm(target_pos[:2] - eef_pos[:2])),
        "z_error": float(abs(target_pos[2] - eef_pos[2])),
        "theta_error": float(np.linalg.norm(orientation_error)),
    }


def _execute_single_action(env: Any, metadata: dict[str, Any]) -> None:
    gt_action = metadata["gt_action"]
    action = _osc_pose_action(
        env,
        np.asarray(gt_action["delta_pos"], dtype=np.float64),
        float(gt_action.get("gripper_action", 0.0)),
        orientation_delta=np.asarray(gt_action.get("orientation_delta_axis_angle", [0.0, 0.0, 0.0]), dtype=np.float64),
    )
    env.step(action)


def _execute_to_target(
    env: Any,
    metadata: dict[str, Any],
    body_names: list[str],
    target_pos: np.ndarray,
    target_ori: np.ndarray,
    position_threshold: float,
    theta_threshold: float,
) -> int:
    gripper_action = float(metadata["gt_action"].get("gripper_action", 0.0))
    for step in range(1, MAX_STEPS + 1):
        eef_pos, eef_ori = _eef_pose(env, body_names)
        position_delta = _bounded_position_delta(
            target_pos - eef_pos,
            gain=POSITION_GAIN,
            max_delta=MAX_DELTA_POS,
        )
        orientation_delta = _bounded_orientation_delta(
            _orientation_delta_axis_angle(target_ori, eef_ori),
            gain=ORIENTATION_GAIN,
            max_delta=MAX_ORIENTATION_DELTA,
        )
        env.step(_osc_pose_action(env, position_delta, gripper_action, orientation_delta=orientation_delta))
        errors = _pregrasp_errors(env, body_names, target_pos, target_ori)
        if errors["position_error"] <= position_threshold and errors["theta_error"] <= theta_threshold:
            return step
    return MAX_STEPS


def _render_outputs(env: Any, renderer: Any, sample_config: Any, output_dir: Path, label: str) -> dict[str, str]:
    render_dir = output_dir / label
    render_dir.mkdir(parents=True, exist_ok=True)
    body_names = default_bridge_body_names(env.sim)
    arena_geom_names = default_arena_geom_names()
    cube_visual_geom_name = default_cube_visual_geom_name()
    outputs: dict[str, str] = {}
    outputs.update(_render_mujoco(env, sample_config, render_dir))
    outputs.update(_render_blender(renderer, env, sample_config, render_dir, body_names, arena_geom_names, cube_visual_geom_name))
    return outputs


def _check_sample(sample_dir: Path, output_dir: Path) -> dict[str, Any]:
    metadata = _load_metadata(sample_dir)
    sample_config = _build_sample_config(metadata, output_dir)
    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env_config = RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    env = build_robosuite_env(placement_initializer, initialization_noise, env_config)
    renderer = None
    try:
        env.reset()
        _restore_state(env, metadata)
        body_names = default_bridge_body_names(env.sim)
        target_pos, target_ori = _target_from_action(env, metadata, body_names)

        policy_state = metadata.get("policy_state") or {}
        position_threshold = (
            float(POSITION_THRESHOLD)
            if POSITION_THRESHOLD is not None
            else max(float(policy_state.get("xy_threshold", 0.01)), CLOSE_Z_THRESHOLD)
        )
        theta_threshold = (
            float(THETA_THRESHOLD)
            if THETA_THRESHOLD is not None
            else float(policy_state.get("theta_threshold", 0.10))
        )

        if RENDER_BLENDER:
            renderer = _build_sample_renderer(sample_config)
            renderer.setup_scene()

        before_errors = _pregrasp_errors(env, body_names, target_pos, target_ori)
        before_render_outputs = _render_outputs(env, renderer, sample_config, output_dir, "before") if RENDER_BEFORE else {}

        if metadata["gt_action"].get("already_pregrasp", False):
            executed_steps = 0
        elif EXECUTE_MODE == "single_action":
            _execute_single_action(env, metadata)
            executed_steps = 1
        elif EXECUTE_MODE == "to_target":
            executed_steps = _execute_to_target(
                env,
                metadata,
                body_names,
                target_pos,
                target_ori,
                position_threshold,
                theta_threshold,
            )
        else:
            raise ValueError(f"Unsupported EXECUTE_MODE: {EXECUTE_MODE!r}")

        after_errors = _pregrasp_errors(env, body_names, target_pos, target_ori)
        cube_geom = get_geom_state(env.sim, default_cube_visual_geom_name())
        after_render_outputs = _render_outputs(env, renderer, sample_config, output_dir, "after")
        pregrasp_reached = (
            after_errors["position_error"] <= position_threshold
            and after_errors["theta_error"] <= theta_threshold
        )

        summary = {
            "sample_dir": str(sample_dir),
            "output_dir": str(output_dir),
            "execute_mode": EXECUTE_MODE,
            "executed_steps": executed_steps,
            "restore_qvel": RESTORE_QVEL,
            "target_pos": target_pos.tolist(),
            "target_ori": target_ori.tolist(),
            "cube_position_after": np.asarray(cube_geom.position, dtype=np.float64).tolist(),
            "thresholds": {
                "position_threshold": position_threshold,
                "theta_threshold": theta_threshold,
            },
            "before_errors": before_errors,
            "after_errors": after_errors,
            "pregrasp_reached": bool(pregrasp_reached),
            "render_outputs": {
                "before": before_render_outputs,
                "after": after_render_outputs,
            },
        }
        (output_dir / "check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        close_env(env)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved GT pregrasp action and render the result.")
    parser.add_argument("sample_dir", nargs="?", default=SAMPLE_DIR, help="Saved sample directory containing metadata.json")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for check renders; default: sample_dir/data_check")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sample_dir = Path(args.sample_dir).expanduser()
    if not sample_dir.is_absolute():
        sample_dir = REPO_ROOT / sample_dir
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else sample_dir / "data_check"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _check_sample(sample_dir, output_dir)
    print(json.dumps({
        "summary_path": str(output_dir / "check_summary.json"),
        "after_mujoco_rgb": summary["render_outputs"]["after"].get("mujoco_rgb_path"),
        "after_blender_rgb": summary["render_outputs"]["after"].get("blender_rgb_no_fingerlight_preview_png_path"),
        "position_error": summary["after_errors"]["position_error"],
        "theta_error": summary["after_errors"]["theta_error"],
        "position_threshold": summary["thresholds"]["position_threshold"],
        "theta_threshold": summary["thresholds"]["theta_threshold"],
        "pregrasp_reached": summary["pregrasp_reached"],
        "executed_steps": summary["executed_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
