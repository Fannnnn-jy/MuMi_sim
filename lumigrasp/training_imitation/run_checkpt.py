#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from lumigrasp.data_collection.sample import (
    SampleDemoConfig,
    build_sample_placement_initializer,
    sample_eef_pose,
    sample_robot_initialization_noise,
    solve_eef_ik,
)
from lumigrasp.training_imitation.dataset import LumiGraspImitationDataset
from lumigrasp.training_imitation.model import SimpleImitationPolicy


DEFAULT_CHECKPOINT = "/home/ubuntu/Junyi/LumiGrasp/runs/imitation/last.pt"
DEFAULT_OUTPUT_DIR = "outputs/checkpoint_policy_demo"
WIDTH = 512
HEIGHT = 512
SAMPLES = 64
CAMERA_NAME = "agentview"
SEED = 42
ROBOT_ASSET_DIR = "assets/robot_asset/franka_emika_panda"
SETTLE_STEPS = 100
USE_GPU = False
MAX_STEPS = 120
HORIZON = 500
CONTROLLER = "BASIC"

XYZ_LOW = (-0.10, -0.10, 0.88)
XYZ_HIGH = (0.10, 0.10, 1.02)
YAW_LOW = -3.141592653589793
YAW_HIGH = 3.141592653589793

OBJECT_X_RANGE = (-0.10, 0.10)
OBJECT_Y_RANGE = (-0.10, 0.10)
OBJECT_ROTATION = (-3.141592653589793, 3.141592653589793)
OBJECT_REFERENCE_POS = (0.0, 0.0, 0.8)
OBJECT_Z_OFFSET = 0.01
ROBOT_INITIALIZATION_NOISE_TYPE = "uniform"
ROBOT_INITIALIZATION_NOISE_MAGNITUDE = 0.0

PREGRASP_CLEARANCE = 0.08
GRASP_CUBE_HALF_SIZE = 0.021
PREGRASP_POS_THRESHOLD = 0.02
STOP_ACTION_NORM = 0.01
GRIPPER_ACTION = 0.0

GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 2.50
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = False

LIGHT_CONFIGS = (
    ("no_fingerlight", False, False),
    ("blue_only", False, True),
    ("red_only", True, False),
)


def _load_replay_metadata(sample_dir: Path | None, metadata_path: Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if sample_dir is None and metadata_path is None:
        return None, None
    if metadata_path is None:
        assert sample_dir is not None
        metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata, metadata_path


def _sample_config_from_metadata(metadata: dict[str, Any] | None, seed: int) -> SampleDemoConfig:
    cube_range = (metadata or {}).get("cube_range") or {}
    return SampleDemoConfig(
        seed=seed,
        xyz_low=XYZ_LOW,
        xyz_high=XYZ_HIGH,
        yaw_low=YAW_LOW,
        yaw_high=YAW_HIGH,
        object_x_range=tuple(cube_range.get("object_x_range", OBJECT_X_RANGE)),
        object_y_range=tuple(cube_range.get("object_y_range", OBJECT_Y_RANGE)),
        object_rotation=cube_range.get("object_rotation", OBJECT_ROTATION),
        object_reference_pos=tuple(cube_range.get("object_reference_pos", OBJECT_REFERENCE_POS)),
        object_z_offset=float(cube_range.get("object_z_offset", OBJECT_Z_OFFSET)),
        robot_initialization_noise_type=ROBOT_INITIALIZATION_NOISE_TYPE,
        robot_initialization_noise_magnitude=ROBOT_INITIALIZATION_NOISE_MAGNITUDE,
    )


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
        for index, joint_name in enumerate(joints):
            if index < qpos.size:
                qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
                env.sim.data.qpos[qpos_addr] = float(qpos[index])


def _restore_state_from_metadata(env: Any, metadata: dict[str, Any]) -> None:
    state = metadata["state"]
    robot = env.robots[0]
    robot_state = state["robot_state"]["robots"][0]

    arm_qpos = np.asarray(robot_state["arm_qpos"], dtype=np.float64)
    env.sim.data.qpos[np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)] = arm_qpos
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


def _as_summary_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


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


def _arm_qpos6(env: Any) -> np.ndarray:
    robot = env.robots[0]
    indexes = getattr(robot, "_ref_arm_joint_pos_indexes", None)
    if indexes is None:
        indexes = getattr(robot, "_ref_joint_pos_indexes", None)
    if indexes is None:
        raise RuntimeError("Could not find robot arm joint qpos indexes")
    qpos = np.asarray(env.sim.data.qpos[indexes], dtype=np.float32)
    if qpos.shape[0] < 6:
        raise RuntimeError(f"Expected at least 6 arm joints, got {qpos.shape[0]}")
    return qpos[:6]


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


def _render_policy_inputs(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    body_names: list[str],
    cube_visual_geom_name: str,
    output_dir: Path,
    step: int,
) -> tuple[list[Path], dict[str, Any]]:
    _sync_renderer(renderer, env, body_names, cube_visual_geom_name)
    step_dir = output_dir / "model_inputs" / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    exr_paths: list[Path] = []
    render_paths: dict[str, Any] = {"step": step, "images": {}}
    for name, left_enabled, right_enabled in LIGHT_CONFIGS:
        renderer.set_gripper_finger_lights("left", left_enabled)
        renderer.set_gripper_finger_lights("right", right_enabled)
        raw_path = step_dir / f"blender_rgb_{name}.exr"
        preview_path = step_dir / f"blender_rgb_{name}.png"
        paths = renderer.render_rgb_outputs(raw_path, preview_path)
        exr_paths.append(Path(paths["raw_exr_path"]))
        render_paths["images"][name] = {key: str(value) for key, value in paths.items()}
    return exr_paths, render_paths


def _policy_input_from_paths(
    image_loader: LumiGraspImitationDataset,
    image_paths: list[Path],
    qpos6: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = [image_loader._load_image(path) for path in image_paths]
    image_tensor = torch.cat(images, dim=0).unsqueeze(0).to(device)
    qpos_tensor = torch.from_numpy(qpos6.astype(np.float32)).unsqueeze(0).to(device)
    return image_tensor, qpos_tensor


def _quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = xyz / sin_half
    return axis * angle


def _quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return _quat_xyzw_to_axis_angle(np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64))


def _load_policy(checkpoint_path: Path, device: torch.device) -> tuple[SimpleImitationPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config") or {})
    model = SimpleImitationPolicy()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _pregrasp_error(cube_position: np.ndarray, eef_position: np.ndarray) -> tuple[np.ndarray, float]:
    target = np.asarray(cube_position, dtype=np.float64) + np.asarray(
        [0.0, 0.0, GRASP_CUBE_HALF_SIZE + PREGRASP_CLEARANCE],
        dtype=np.float64,
    )
    return target, float(np.linalg.norm(target - eef_position))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LumiGrasp imitation checkpoint in the demo environment.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--sample-dir", default=None, help="Saved sample directory containing metadata.json to replay.")
    parser.add_argument("--metadata", default=None, help="Explicit metadata.json path to replay; overrides --sample-dir.")
    parser.add_argument("--no-render-save", action="store_true", help="Render only temporary model inputs and omit extra summary frame paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "summary.json"
    sample_dir = Path(args.sample_dir).expanduser() if args.sample_dir else None
    replay_metadata_path = Path(args.metadata).expanduser() if args.metadata else None
    replay_metadata, replay_metadata_path = _load_replay_metadata(sample_dir, replay_metadata_path)
    replay_mode = replay_metadata is not None

    model, train_config = _load_policy(checkpoint_path, device)
    image_size = int(train_config.get("image_size", 512))
    quat_key = str(train_config.get("quat_key", "delta_quat_xyzw"))
    image_loader = LumiGraspImitationDataset([], image_size=image_size)

    env_config = RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    sample_config = _sample_config_from_metadata(replay_metadata, seed=args.seed)
    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_config)
    renderer = PrimitiveBlenderRenderer(
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

    rendered_inputs: list[dict[str, Any]] = []
    reached_pregrasp = False
    done = False
    step = 0
    initial_ik = None
    initial_sample = None

    try:
        env.reset()
        if replay_mode:
            assert replay_metadata is not None
            _restore_state_from_metadata(env, replay_metadata)
            initial_sample = replay_metadata.get("initial_eef_sample")
            initial_ik = replay_metadata.get("initial_ik")
            print(f"Replay mode: restored scene from {replay_metadata_path}", flush=True)
        else:
            _settle_env(env, SETTLE_STEPS)

        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        camera = get_camera_state(env.sim, CAMERA_NAME)

        if not replay_mode:
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

        renderer.setup_scene()
        renderer.sync_arena_geoms(arena_geoms)
        renderer.set_camera_from_mujoco(camera)

        with metadata_path.open("w", encoding="utf-8") as f:
            for step in range(1, args.max_steps + 1):
                cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
                cube_position_before = np.asarray(cube_geom.position, dtype=np.float64)
                eef_pos_before, eef_ori_before = _eef_pose_from_body(env, body_names)
                pregrasp_target, pregrasp_pos_err_before = _pregrasp_error(cube_position_before, eef_pos_before)

                image_paths, render_paths = _render_policy_inputs(
                    renderer,
                    env,
                    body_names,
                    cube_visual_geom_name,
                    output_dir,
                    step,
                )
                if not args.no_render_save:
                    rendered_inputs.append(render_paths)

                qpos6 = _arm_qpos6(env)
                images, qpos = _policy_input_from_paths(image_loader, image_paths, qpos6, device)
                with torch.no_grad():
                    pred = model(images, qpos).squeeze(0).detach().cpu().numpy().astype(np.float64)

                position_delta = pred[:3]
                quat_delta = pred[3:7]
                if quat_key == "delta_quat_wxyz":
                    orientation_delta = _quat_wxyz_to_axis_angle(quat_delta)
                else:
                    orientation_delta = _quat_xyzw_to_axis_angle(quat_delta)

                action = _osc_pose_action(
                    env,
                    position_delta,
                    GRIPPER_ACTION,
                    orientation_delta=orientation_delta,
                )
                obs, reward, done, info = env.step(action)

                cube_geom_after = get_geom_state(env.sim, cube_visual_geom_name)
                cube_position_after = np.asarray(cube_geom_after.position, dtype=np.float64)
                cube_orientation_after = _quat_wxyz_to_mat(cube_geom_after.quaternion_wxyz)
                eef_pos_after, eef_ori_after = _eef_pose_from_body(env, body_names)
                _, pregrasp_pos_err_after = _pregrasp_error(cube_position_after, eef_pos_after)
                action_norm = float(np.linalg.norm(position_delta))
                reached_pregrasp = bool(
                    pregrasp_pos_err_after <= PREGRASP_POS_THRESHOLD
                    or (pregrasp_pos_err_after <= 2.0 * PREGRASP_POS_THRESHOLD and action_norm <= STOP_ACTION_NORM)
                )

                row = {
                    "step": step,
                    "reached_pregrasp": reached_pregrasp,
                    "done": bool(done),
                    "reward": float(reward),
                    "qpos6": qpos6.tolist(),
                    "cube_position": cube_position_after.tolist(),
                    "cube_orientation": cube_orientation_after.tolist(),
                    "eef_pos_before": eef_pos_before.tolist(),
                    "eef_orientation_before": eef_ori_before.tolist(),
                    "eef_pos": eef_pos_after.tolist(),
                    "eef_orientation": eef_ori_after.tolist(),
                    "pregrasp_target": pregrasp_target.tolist(),
                    "pregrasp_pos_err_before": pregrasp_pos_err_before,
                    "pregrasp_pos_err": pregrasp_pos_err_after,
                    "pred_action": pred.tolist(),
                    "position_delta": position_delta.tolist(),
                    "quat_delta": quat_delta.tolist(),
                    "orientation_delta_axis_angle": orientation_delta.tolist(),
                    "osc_action": action.tolist(),
                    "render_paths": render_paths,
                    "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                }
                f.write(json.dumps(row) + "\n")

                print(
                    f"Step {step}: pregrasp_err={pregrasp_pos_err_after:.4f} "
                    f"dpos_norm={action_norm:.4f} reached_pregrasp={reached_pregrasp} done={bool(done)}"
                )
                if reached_pregrasp or done:
                    break

        summary = {
            "config": {
                "checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
                "seed": args.seed,
                "max_steps": args.max_steps,
                "controller": CONTROLLER,
                "image_size": image_size,
                "quat_key": quat_key,
                "pregrasp_clearance": PREGRASP_CLEARANCE,
                "grasp_cube_half_size": GRASP_CUBE_HALF_SIZE,
                "pregrasp_pos_threshold": PREGRASP_POS_THRESHOLD,
                "replay_mode": replay_mode,
                "replay_metadata_path": str(replay_metadata_path) if replay_metadata_path is not None else None,
                "replay_sample_dir": str(replay_metadata_path.parent) if replay_metadata_path is not None else None,
            },
            "train_config": train_config,
            "checkpoint_epoch": int(torch.load(checkpoint_path, map_location="cpu").get("epoch", -1)),
            "env_config": env_config.__dict__,
            "sample_config": sample_config.__dict__,
            "initial_sample": _as_summary_value(initial_sample),
            "initial_ik": _as_summary_value(initial_ik),
            "reached_pregrasp": reached_pregrasp,
            "done": bool(done),
            "steps": step,
            "metadata_path": str(metadata_path),
            "rendered_inputs": rendered_inputs,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
            "reached_pregrasp": reached_pregrasp,
            "done": bool(done),
            "steps": step,
        }, indent=2))
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
