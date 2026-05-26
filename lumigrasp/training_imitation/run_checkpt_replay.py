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
from lumigrasp.bridge.robosuite_env import RobosuiteEnvConfig, build_robosuite_env, close_env
from lumigrasp.bridge.utils.control_utils import _osc_pose_action
from lumigrasp.bridge.utils.coords import _orientation_delta_axis_angle, _quat_wxyz_to_mat
from lumigrasp.data_collection.gt_policy import GTPolicy, GTPolicyConfig
from lumigrasp.data_collection.sample import build_sample_placement_initializer, sample_robot_initialization_noise
from lumigrasp.training_imitation.dataset import LumiGraspImitationDataset
from lumigrasp.training_imitation.model import SimpleImitationPolicy
from lumigrasp.training_imitation.run_checkpt import (
    CAMERA_NAME,
    CONTROLLER,
    GRASP_CUBE_HALF_SIZE,
    GRIPPER_LIGHT_ENERGY,
    GRIPPER_LIGHT_SIZE,
    GRIPPER_LIGHT_TYPE,
    GRIPPER_SPOT_BLEND,
    GRIPPER_SPOT_SIZE,
    HEIGHT,
    HORIZON,
    LIGHT_CONFIGS,
    PREGRASP_CLEARANCE,
    ROBOT_ASSET_DIR,
    SAMPLES,
    SEED,
    SHOW_GRIPPER_LIGHT_MARKER,
    USE_GPU,
    WIDTH,
    _arm_qpos6,
    _load_policy,
    _load_replay_metadata,
    _policy_input_from_paths,
    _quat_wxyz_to_axis_angle,
    _quat_xyzw_to_axis_angle,
    _restore_state_from_metadata,
    _sample_config_from_metadata,
    _sync_renderer,
)


DEFAULT_CHECKPOINT = "/home/ubuntu/Junyi/LumiGrasp/runs/imitation_overfit_8/best.pt"
DEFAULT_METADATA = "/home/ubuntu/Junyi/LumiGrasp/data/pickcube-v1/cube_x_0p00_0p10_y_0p00_0p10/sample_000000/metadata.json"
DEFAULT_OUTPUT_DIR = "outputs/checkpoint_replay"
MAX_GT_STEPS = 300
MAX_MODEL_STEPS = 20
RENDER_EVERY = 10
POSITION_THRESHOLD = 10.0
THETA_THRESHOLD = 10.0
LIFT_SUCCESS_HEIGHT = 0.08
CLOSE_Z_THRESHOLD = 0.005
LIFT_HEIGHT = 0.15
MAX_DELTA_POS = 0.5
POSITION_GAIN = 12.0
ORIENTATION_GAIN = 2.0
MAX_ORIENTATION_DELTA = 0.25
MODEL_GRIPPER_ACTION = 0.0


def _eef_pose_from_body(env: Any, body_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    poses = collect_body_poses(env.sim, body_names)
    eef_pose = poses["gripper0_right_eef"]
    return np.asarray(eef_pose.position, dtype=np.float64), _quat_wxyz_to_mat(eef_pose.quaternion_wxyz)


def _render_model_inputs(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    body_names: list[str],
    cube_visual_geom_name: str,
    output_dir: Path,
    label: str,
) -> tuple[list[Path], dict[str, Any]]:
    _sync_renderer(renderer, env, body_names, cube_visual_geom_name)
    render_dir = output_dir / "model_inputs" / label
    render_dir.mkdir(parents=True, exist_ok=True)
    paths_out: list[Path] = []
    record: dict[str, Any] = {"label": label, "images": {}}
    for name, left_enabled, right_enabled in LIGHT_CONFIGS:
        renderer.set_gripper_finger_lights("left", left_enabled)
        renderer.set_gripper_finger_lights("right", right_enabled)
        raw_path = render_dir / f"blender_rgb_{name}.exr"
        preview_path = render_dir / f"blender_rgb_{name}.png"
        paths = renderer.render_rgb_outputs(raw_path, preview_path)
        paths_out.append(Path(paths["raw_exr_path"]))
        record["images"][name] = {key: str(value) for key, value in paths.items()}
    return paths_out, record


def _render_frame(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    body_names: list[str],
    cube_visual_geom_name: str,
    output_dir: Path,
    step: int,
    phase: str,
) -> dict[str, Any]:
    _sync_renderer(renderer, env, body_names, cube_visual_geom_name)
    renderer.set_gripper_finger_lights("left", True)
    renderer.set_gripper_finger_lights("right", True)
    raw_path = output_dir / "frames" / f"frame_{step:04d}_{phase}.exr"
    preview_path = output_dir / "frames" / f"frame_{step:04d}_{phase}.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    paths = renderer.render_rgb_outputs(raw_path, preview_path)
    return {"step": step, "phase": phase, **{key: str(value) for key, value in paths.items()}}


def _target_from_metadata(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    gt_action = metadata["gt_action"]
    target_pos = np.asarray(gt_action["target_pos"], dtype=np.float64)
    target_ori = np.asarray(gt_action["target_ori"], dtype=np.float64)
    return target_pos, target_ori


def _pregrasp_errors(eef_pos: np.ndarray, eef_ori: np.ndarray, target_pos: np.ndarray, target_ori: np.ndarray) -> dict[str, float]:
    orientation_error = _orientation_delta_axis_angle(target_ori, eef_ori)
    return {
        "position_error": float(np.linalg.norm(target_pos - eef_pos)),
        "xy_error": float(np.linalg.norm(target_pos[:2] - eef_pos[:2])),
        "z_error": float(abs(target_pos[2] - eef_pos[2])),
        "theta_error": float(np.linalg.norm(orientation_error)),
    }


def _is_pregrasp_reached(errors: dict[str, float], position_threshold: float, theta_threshold: float) -> bool:
    return bool(errors["position_error"] <= position_threshold and errors["theta_error"] <= theta_threshold)


def _model_action(
    model: SimpleImitationPolicy,
    image_loader: LumiGraspImitationDataset,
    image_paths: list[Path],
    qpos6: np.ndarray,
    quat_key: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, qpos = _policy_input_from_paths(image_loader, image_paths, qpos6, device)
    with torch.no_grad():
        pred = model(images, qpos).squeeze(0).detach().cpu().numpy().astype(np.float64)
    position_delta = pred[:3]
    quat_delta = pred[3:7]
    orientation_delta = _quat_wxyz_to_axis_angle(quat_delta) if quat_key == "delta_quat_wxyz" else _quat_xyzw_to_axis_angle(quat_delta)
    return pred, position_delta, orientation_delta


def _force_gt_policy_descend(policy: GTPolicy, target_ori: np.ndarray) -> None:
    policy._phase = "descend"
    policy._target_ori = np.asarray(target_ori, dtype=np.float64)
    policy._grasp_commanded = False
    policy._grasp_steps = 0
    policy._lift_target_pos = None


def _gt_policy_config(metadata: dict[str, Any]) -> GTPolicyConfig:
    policy_state = metadata.get("policy_state") or {}
    return GTPolicyConfig(
        xy_threshold=float(policy_state.get("xy_threshold", POSITION_THRESHOLD)),
        theta_threshold=float(policy_state.get("theta_threshold", THETA_THRESHOLD)),
        xy_threshold_noise=0.0,
        theta_threshold_noise=0.0,
        pregrasp_clearance=PREGRASP_CLEARANCE,
        grasp_cube_half_size=GRASP_CUBE_HALF_SIZE,
        max_delta_pos=MAX_DELTA_POS,
        position_gain=POSITION_GAIN,
        orientation_gain=ORIENTATION_GAIN,
        max_orientation_delta=MAX_ORIENTATION_DELTA,
        close_z_threshold=CLOSE_Z_THRESHOLD,
        lift_height=LIFT_HEIGHT,
        lift_success_height=LIFT_SUCCESS_HEIGHT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one saved sample: model moves to pregrasp once, GT policy finishes grasp.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--sample-dir", default=None, help="Alternative sample dir containing metadata.json.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-gt-steps", type=int, default=MAX_GT_STEPS)
    parser.add_argument("--max-model-steps", type=int, default=MAX_MODEL_STEPS)
    parser.add_argument("--position-threshold", type=float, default=POSITION_THRESHOLD)
    parser.add_argument("--theta-threshold", type=float, default=THETA_THRESHOLD)
    parser.add_argument("--render-every", type=int, default=RENDER_EVERY)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_log_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "summary.json"

    sample_dir = Path(args.sample_dir).expanduser() if args.sample_dir else None
    metadata_arg = Path(args.metadata).expanduser() if args.metadata else None
    replay_metadata, replay_metadata_path = _load_replay_metadata(sample_dir, metadata_arg)
    if replay_metadata is None or replay_metadata_path is None:
        raise ValueError("run_checkpt_replay requires --metadata or --sample-dir")

    checkpoint_path = Path(args.checkpoint).expanduser()
    model, train_config = _load_policy(checkpoint_path, device)
    image_size = int(train_config.get("image_size", 512))
    quat_key = str(train_config.get("quat_key", "delta_quat_xyzw"))
    image_loader = LumiGraspImitationDataset([], image_size=image_size)

    sample_config = _sample_config_from_metadata(replay_metadata, seed=args.seed)
    env_config = RobosuiteEnvConfig(
        env_name="LumiLift",
        robots="Panda",
        controller=CONTROLLER,
        has_offscreen_renderer=True,
        horizon=HORIZON,
    )
    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_config)
    renderer = None if args.no_render else PrimitiveBlenderRenderer(
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
    rendered_model_inputs: dict[str, Any] | None = None
    model_pregrasp_reached = False
    success = False
    done = False
    final_phase = None
    model_row: dict[str, Any] | None = None
    model_steps = 0
    gt_steps = 0

    try:
        env.reset()
        _restore_state_from_metadata(env, replay_metadata)
        print(f"Replay mode: restored scene from {replay_metadata_path}", flush=True)

        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        camera = get_camera_state(env.sim, CAMERA_NAME)
        target_pos, target_ori = _target_from_metadata(replay_metadata)
        cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
        reference_cube_z = float(cube_geom.position[2])

        if renderer is not None:
            renderer.setup_scene()
            renderer.sync_arena_geoms(arena_geoms)
            renderer.set_camera_from_mujoco(camera)
            rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, 0, "initial"))

        with metadata_log_path.open("w", encoding="utf-8") as f:
            while not done and not model_pregrasp_reached and model_steps < args.max_model_steps:
                model_steps += 1
                eef_pos_before, eef_ori_before = _eef_pose_from_body(env, body_names)
                before_errors = _pregrasp_errors(eef_pos_before, eef_ori_before, target_pos, target_ori)
                if renderer is None:
                    if model_steps == 1:
                        source_paths = replay_metadata["render_outputs"]
                        image_paths = [
                            Path(source_paths["blender_rgb_no_fingerlight_raw_exr_path"]),
                            Path(source_paths["blender_rgb_blue_only_raw_exr_path"]),
                            Path(source_paths["blender_rgb_red_only_raw_exr_path"]),
                        ]
                        render_record = {"source": "saved_dataset_exr", "image_paths": [str(path) for path in image_paths]}
                    else:
                        render_record = {"source": "no_render_reuses_initial_dataset_exr", "warning": "Use rendering for valid closed-loop image inputs after step 1."}
                else:
                    image_paths, render_record = _render_model_inputs(
                        renderer,
                        env,
                        body_names,
                        cube_visual_geom_name,
                        output_dir,
                        f"model_step_{model_steps:04d}",
                    )
                    rendered_model_inputs = render_record

                qpos6 = _arm_qpos6(env)
                pred, position_delta, orientation_delta = _model_action(model, image_loader, image_paths, qpos6, quat_key, device)
                action = _osc_pose_action(env, position_delta, MODEL_GRIPPER_ACTION, orientation_delta=orientation_delta)
                obs, reward, done, info = env.step(action)
                eef_pos_after, eef_ori_after = _eef_pose_from_body(env, body_names)
                after_errors = _pregrasp_errors(eef_pos_after, eef_ori_after, target_pos, target_ori)
                model_pregrasp_reached = _is_pregrasp_reached(after_errors, args.position_threshold, args.theta_threshold)
                cube_geom_after = get_geom_state(env.sim, cube_visual_geom_name)
                cube_position_after = np.asarray(cube_geom_after.position, dtype=np.float64)
                cube_orientation_after = _quat_wxyz_to_mat(cube_geom_after.quaternion_wxyz)

                model_row = {
                    "step": model_steps,
                    "source": "model_pregrasp",
                    "phase": "model_pregrasp",
                    "model_pregrasp_reached": model_pregrasp_reached,
                    "done": bool(done),
                    "reward": float(reward),
                    "qpos6": qpos6.tolist(),
                    "target_pos": target_pos.tolist(),
                    "target_ori": target_ori.tolist(),
                    "before_errors": before_errors,
                    "after_errors": after_errors,
                    "eef_pos_before": eef_pos_before.tolist(),
                    "eef_ori_before": eef_ori_before.tolist(),
                    "eef_pos": eef_pos_after.tolist(),
                    "eef_ori": eef_ori_after.tolist(),
                    "cube_position": cube_position_after.tolist(),
                    "cube_orientation": cube_orientation_after.tolist(),
                    "pred_action": pred.tolist(),
                    "position_delta": position_delta.tolist(),
                    "orientation_delta_axis_angle": orientation_delta.tolist(),
                    "osc_action": action.tolist(),
                    "render_paths": render_record,
                    "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                }
                f.write(json.dumps(model_row) + "\n")
                print(
                    f"Model step {model_steps}: "
                    f"pos_err={after_errors['position_error']:.4f} theta_err={after_errors['theta_error']:.4f} "
                    f"reached_pregrasp={model_pregrasp_reached} done={bool(done)}"
                )

                if renderer is not None and args.render_every > 0 and model_steps % args.render_every == 0:
                    rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, model_steps, "model_step"))

            if renderer is not None:
                rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, model_steps, "after_model"))

            if model_pregrasp_reached and not done:
                gt_policy = GTPolicy(_gt_policy_config(replay_metadata), seed=args.seed)
                policy_state = gt_policy.reset()
                _force_gt_policy_descend(gt_policy, target_ori)
                for gt_steps in range(1, args.max_gt_steps + 1):
                    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
                    cube_position = np.asarray(cube_geom.position, dtype=np.float64)
                    cube_orientation = _quat_wxyz_to_mat(cube_geom.quaternion_wxyz)
                    eef_pos, eef_ori = _eef_pose_from_body(env, body_names)
                    policy_action = gt_policy.act_from_pose(
                        eef_pos=eef_pos,
                        eef_ori=eef_ori,
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
                    eef_pos_after, eef_ori_after = _eef_pose_from_body(env, body_names)
                    success = gt_policy.is_success(cube_position_after, reference_cube_z=reference_cube_z)
                    final_phase = policy_action.phase
                    render_paths = None
                    if renderer is not None and args.render_every > 0 and gt_steps % args.render_every == 0:
                        render_paths = _render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, model_steps + gt_steps, policy_action.phase)
                        rendered_frames.append(render_paths)

                    f.write(json.dumps({
                        "step": model_steps + gt_steps,
                        "source": "gt_policy",
                        "phase": policy_action.phase,
                        "success": success,
                        "done": bool(done),
                        "reward": float(reward),
                        "cube_position": cube_position_after.tolist(),
                        "cube_orientation": cube_orientation_after.tolist(),
                        "eef_pos": eef_pos_after.tolist(),
                        "eef_ori": eef_ori_after.tolist(),
                        "policy_action": policy_action.as_dict(),
                        "osc_action": action.tolist(),
                        "render_paths": render_paths,
                        "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                    }) + "\n")
                    print(
                        f"GT step {gt_steps}: phase={policy_action.phase} "
                        f"success={success} cube_lift={cube_position_after[2] - reference_cube_z:.4f} done={bool(done)}"
                    )
                    if success or done:
                        break
            else:
                final_phase = "model_pregrasp_failed" if not model_pregrasp_reached else "done_after_model"

        if renderer is not None and rendered_frames and rendered_frames[-1].get("step") != model_steps + gt_steps:
            rendered_frames.append(_render_frame(renderer, env, body_names, cube_visual_geom_name, output_dir, model_steps + gt_steps, "final"))

        summary = {
            "config": {
                "checkpoint": str(checkpoint_path),
                "metadata": str(replay_metadata_path),
                "output_dir": str(output_dir),
                "device": str(device),
                "seed": args.seed,
                "max_gt_steps": args.max_gt_steps,
                "max_model_steps": args.max_model_steps,
                "position_threshold": args.position_threshold,
                "theta_threshold": args.theta_threshold,
                "render_every": args.render_every,
                "no_render": args.no_render,
                "controller": CONTROLLER,
                "image_size": image_size,
                "quat_key": quat_key,
            },
            "train_config": train_config,
            "env_config": env_config.__dict__,
            "sample_config": sample_config.__dict__,
            "reference_cube_z": reference_cube_z,
            "target_pos": target_pos.tolist(),
            "target_ori": target_ori.tolist(),
            "model_pregrasp_reached": model_pregrasp_reached,
            "model_step": model_row,
            "model_steps": model_steps,
            "gt_steps": gt_steps,
            "success": success,
            "done": bool(done),
            "final_phase": final_phase,
            "metadata_path": str(metadata_log_path),
            "summary_path": str(summary_path),
            "rendered_model_inputs": rendered_model_inputs,
            "rendered_frames": rendered_frames,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_log_path),
            "model_pregrasp_reached": model_pregrasp_reached,
            "model_steps": model_steps,
            "gt_steps": gt_steps,
            "success": success,
            "done": bool(done),
            "final_phase": final_phase,
        }, indent=2))
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
