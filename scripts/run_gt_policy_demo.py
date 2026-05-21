#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumigrasp.bridge.blender_renderer import PrimitiveBlenderRenderer
from lumigrasp.bridge.mujoco_state import (
    collect_body_poses,
    collect_geom_states,
    default_arena_geom_names,
    default_bridge_body_names,
    default_cube_visual_geom_name,
    get_camera_state,
    get_geom_state,
)
from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig, build_robosuite_env, close_env, set_grippers_max_open, _settle_env
from lumigrasp.bridge.utils.control_utils import _osc_pose_action
from lumigrasp.bridge.utils.coords import _quat_wxyz_to_mat
from lumigrasp.data_collection.gt_policy import GTPolicy, GTPolicyConfig
from lumigrasp.data_collection.sample import (
    SampleDemoConfig,
    build_sample_placement_initializer,
    sample_eef_pose,
    sample_robot_initialization_noise,
    solve_eef_ik,
)


OUTPUT_DIR = "outputs/gt_policy_demo"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
SEED = 42
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
SETTLE_STEPS = 100
NO_RENDER = False
USE_GPU = False
RENDER_EVERY = 10
MAX_STEPS = 400
HORIZON = 500
CONTROLLER = "BASIC"

# Random initial EEF pose bounds in world frame, solved by IK before policy rollout.
XYZ_LOW = (-0.15, -0.15, 0.88)
XYZ_HIGH = (0.15, 0.15, 1.02)
YAW_LOW = -3.141592653589793
YAW_HIGH = 3.141592653589793

OBJECT_X_RANGE = (-0.15, 0.15)
OBJECT_Y_RANGE = (-0.15, 0.15)
OBJECT_ROTATION = (-3.141592653589793, 3.141592653589793)
OBJECT_REFERENCE_POS = (0.0, 0.0, 0.8)
OBJECT_Z_OFFSET = 0.01
ROBOT_INITIALIZATION_NOISE_TYPE = "uniform"
ROBOT_INITIALIZATION_NOISE_MAGNITUDE = 0.0

XY_THRESHOLD = 0.01
THETA_THRESHOLD = 0.05
XY_THRESHOLD_NOISE = 0.003
THETA_THRESHOLD_NOISE = 0.0
PREGRASP_CLEARANCE = 0.08
GRASP_CUBE_HALF_SIZE = 0.021
MAX_DELTA_POS = 0.5
POSITION_GAIN = 12.0
ORIENTATION_GAIN = 2.0
MAX_ORIENTATION_DELTA = 0.25
CLOSE_Z_THRESHOLD = 0.005
LIFT_HEIGHT = 0.15
LIFT_SUCCESS_HEIGHT = 0.08

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.50
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False


def _eef_pose_from_body(env: Any, body_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    poses = collect_body_poses(env.sim, body_names)
    eef_pose = poses["gripper0_right_eef"]
    eef_pos = np.asarray(eef_pose.position, dtype=np.float64)
    eef_ori = _quat_wxyz_to_mat(eef_pose.quaternion_wxyz)
    return eef_pos, eef_ori


def _sync_robot_controller_goals(env: Any) -> None:
    env.sim.forward()
    for robot in getattr(env, "robots", []):
        controller = getattr(robot, "composite_controller", None)
        if controller is None:
            continue
        controller.update_state()
        controller.reset()


def _sync_renderer(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    body_names: list[str],
    cube_visual_geom_name: str,
) -> None:
    poses = collect_body_poses(env.sim, body_names)
    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
    renderer.update_poses(poses)
    renderer.sync_cube_geom(cube_geom)
    renderer.set_gripper_finger_lights("left", True)
    renderer.set_gripper_finger_lights("right", True)


def _render_frame(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    body_names: list[str],
    cube_visual_geom_name: str,
    output_dir: Path,
    step: int,
) -> dict[str, Any]:
    _sync_renderer(renderer, env, body_names, cube_visual_geom_name)
    raw_path = output_dir / f"frame_{step:04d}_raw.exr"
    preview_path = output_dir / f"frame_{step:04d}_preview.png"
    paths = renderer.render_rgb_outputs(raw_path, preview_path)
    return {"step": step, **{key: str(value) for key, value in paths.items()}}


def main() -> None:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "summary.json"

    env_config = RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    sample_config = SampleDemoConfig(
        seed=SEED,
        xyz_low=XYZ_LOW,
        xyz_high=XYZ_HIGH,
        yaw_low=YAW_LOW,
        yaw_high=YAW_HIGH,
        object_x_range=OBJECT_X_RANGE,
        object_y_range=OBJECT_Y_RANGE,
        object_rotation=OBJECT_ROTATION,
        object_reference_pos=OBJECT_REFERENCE_POS,
        object_z_offset=OBJECT_Z_OFFSET,
        robot_initialization_noise_type=ROBOT_INITIALIZATION_NOISE_TYPE,
        robot_initialization_noise_magnitude=ROBOT_INITIALIZATION_NOISE_MAGNITUDE,
    )
    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_config)
    renderer = None if NO_RENDER else PrimitiveBlenderRenderer(
        width=WIDTH,
        height=HEIGHT,
        samples=SAMPLES,
        use_gpu=USE_GPU,
        robot_asset_dir=ROBOT_ASSET_DIR,
        gripper_light_type=GRIPPER_LIGHT_TYPE,
        gripper_light_energy=GRIPPER_LIGHT_ENERGY,
        gripper_light_size=GRIPPER_LIGHT_SIZE,
        gripper_spot_size=GRIPPER_SPOT_SIZE,
        gripper_spot_blend=GRIPPER_SPOT_BLEND,
        show_gripper_light_marker=SHOW_GRIPPER_LIGHT_MARKER,
    )

    rendered_frames: list[dict[str, Any]] = []
    success = False
    done = False
    step = 0
    initial_ik = None
    final_phase = None

    try:
        env.reset()
        _settle_env(env, SETTLE_STEPS)

        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        camera = get_camera_state(env.sim, CAMERA_NAME)

        initial_sample = sample_eef_pose(rng, sample_config)
        initial_ik = solve_eef_ik(
            env,
            target_position=np.asarray(initial_sample.position, dtype=np.float64),
            target_orientation=np.asarray(initial_sample.orientation, dtype=np.float64),
            config=sample_config,
        )
        if initial_ik.success:
            env.robots[0].set_robot_joint_positions(np.asarray(initial_ik.qpos, dtype=np.float64))
            set_grippers_max_open(env)
            _sync_robot_controller_goals(env)

        cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
        reference_cube_z = float(cube_geom.position[2])
        gt_policy = GTPolicy(
            GTPolicyConfig(
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
                close_z_threshold=CLOSE_Z_THRESHOLD,
                lift_height=LIFT_HEIGHT,
                lift_success_height=LIFT_SUCCESS_HEIGHT,
            ),
            seed=SEED,
        )
        policy_state = gt_policy.reset()

        if renderer is not None:
            renderer.setup_scene()
            renderer.sync_arena_geoms(arena_geoms)
            renderer.set_camera_from_mujoco(camera)
            rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, 0))

        with metadata_path.open("w", encoding="utf-8") as f:
            for step in range(1, MAX_STEPS + 1):
                cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
                cube_position = np.asarray(cube_geom.position, dtype=np.float64)
                cube_orientation = _quat_wxyz_to_mat(cube_geom.quaternion_wxyz)
                eef_pos_before, eef_ori_before = _eef_pose_from_body(env, body_names)
                policy_action = gt_policy.act_from_pose(
                    eef_pos=eef_pos_before,
                    eef_ori=eef_ori_before,
                    cube_position=cube_position,
                    cube_orientation=cube_orientation,
                    state=policy_state,
                )
                action = _osc_pose_action(
                    env,
                    policy_action.delta_pos,
                    policy_action.gripper_action,
                    orientation_delta=policy_action.orientation_delta,
                )
                obs, reward, done, info = env.step(action)
                cube_geom_after = get_geom_state(env.sim, cube_visual_geom_name)
                cube_position_after = np.asarray(cube_geom_after.position, dtype=np.float64)
                cube_orientation_after = _quat_wxyz_to_mat(cube_geom_after.quaternion_wxyz)
                eef_pos, eef_ori = _eef_pose_from_body(env, body_names)
                success = gt_policy.is_success(cube_position_after, reference_cube_z=reference_cube_z)
                final_phase = policy_action.phase

                render_paths = None
                if renderer is not None and step % RENDER_EVERY == 0:
                    render_paths = _render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, step)
                    rendered_frames.append(render_paths)

                f.write(json.dumps({
                    "step": step,
                    "phase": policy_action.phase,
                    "success": success,
                    "done": bool(done),
                    "reward": float(reward),
                    "cube_position": cube_position_after.tolist(),
                    "cube_orientation": cube_orientation_after.tolist(),
                    "eef_pos": eef_pos.tolist(),
                    "eef_orientation": eef_ori.tolist(),
                    "policy_action": policy_action.as_dict(),
                    "action": action.tolist(),
                    "render_paths": render_paths,
                    "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                }) + "\n")

                print(
                    f"Step {step}: phase={policy_action.phase} "
                    f"success={success} cube_lift={cube_position_after[2] - reference_cube_z:.4f} "
                    f"xy_err={policy_action.xy_err:.4f} theta_err={policy_action.theta_err:.4f}"
                )
                if success or done:
                    break

        if renderer is not None and step % RENDER_EVERY != 0:
            rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, step))

        summary = {
            "config": {
                "output_dir": OUTPUT_DIR,
                "seed": SEED,
                "render_every": RENDER_EVERY,
                "max_steps": MAX_STEPS,
                "controller": CONTROLLER,
                "pregrasp_clearance": PREGRASP_CLEARANCE,
                "grasp_cube_half_size": GRASP_CUBE_HALF_SIZE,
                "lift_height": LIFT_HEIGHT,
                "lift_success_height": LIFT_SUCCESS_HEIGHT,
                "object_x_range": OBJECT_X_RANGE,
                "object_y_range": OBJECT_Y_RANGE,
                "object_rotation": OBJECT_ROTATION,
                "object_reference_pos": OBJECT_REFERENCE_POS,
                "object_z_offset": OBJECT_Z_OFFSET,
                "robot_initialization_noise_type": ROBOT_INITIALIZATION_NOISE_TYPE,
                "robot_initialization_noise_magnitude": ROBOT_INITIALIZATION_NOISE_MAGNITUDE,
            },
            "env_config": env_config.__dict__,
            "sample_config": sample_config.__dict__,
            "initial_sample": initial_sample.__dict__,
            "initial_ik": initial_ik.__dict__ if initial_ik is not None else None,
            "policy_state": policy_state.__dict__,
            "reference_cube_z": reference_cube_z,
            "success": success,
            "done": bool(done),
            "steps": step,
            "final_phase": final_phase,
            "metadata_path": str(metadata_path),
            "rendered_frames": rendered_frames,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
            "success": success,
            "steps": step,
            "rendered_frames": rendered_frames,
        }, indent=2))
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
