from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from robosuite.utils.transform_utils import axisangle2quat

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
from lumigrasp.bridge.robosuite_env import (
    RobosuiteEnvConfig,
    build_robosuite_env,
    close_env,
    set_grippers_max_open,
    _settle_env,
)
from lumigrasp.bridge.utils.coords import _quat_wxyz_to_mat
from lumigrasp.bridge.utils.depth import _linearize_mujoco_depth
from lumigrasp.bridge.utils.exr import save_depth_exr
from lumigrasp.bridge.utils.serialization import _serialize_camera, _serialize_geoms, _serialize_poses
from lumigrasp.data_collection.gt_policy import GTPolicy, GTPolicyConfig
from lumigrasp.data_collection.sample import (
    SampleDemoConfig,
    _build_sample_renderer,
    build_sample_placement_initializer,
    sample_eef_pose,
    sample_robot_initialization_noise,
    solve_eef_ik,
)


@dataclass(frozen=True)
class CubeRangeConfig:
    name: str
    object_x_range: tuple[float, float]
    object_y_range: tuple[float, float]
    object_rotation: tuple[float, float] | float | None = (-np.pi, np.pi)
    object_reference_pos: tuple[float, float, float] = (0.0, 0.0, 0.8)
    object_z_offset: float = 0.01


@dataclass(frozen=True)
class DataCollectionConfig:
    output_dir: str = "/home/ubuntu/Junyi/LumiGrasp/data/pickcube-v1"
    width: int = 512
    height: int = 512
    samples: int = 64
    camera_name: str = "agentview"
    seed: int = 0
    n_samples: int = 100
    max_attempts_per_sample: int = 20
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    settle_steps: int = 5
    use_gpu: bool = False
    render_blender: bool = True
    render_mujoco: bool = True
    render_depth: bool = True
    xyz_low: tuple[float, float, float] = (-0.15, -0.15, 0.88)
    xyz_high: tuple[float, float, float] = (0.15, 0.15, 1.02)
    yaw_low: float = -np.pi
    yaw_high: float = np.pi
    ik_max_iters: int = 250
    ik_position_tolerance: float = 0.003
    ik_orientation_tolerance: float = 0.03
    ik_damping: float = 1e-3
    ik_step_size: float = 0.6
    ik_last_joint_yaw_only: bool = True
    ik_yaw_tolerance: float = 0.03
    xy_threshold: float = 0.01
    theta_threshold: float = 0.10
    xy_threshold_noise: float = 0.003
    theta_threshold_noise: float = 0.0
    pregrasp_clearance: float = 0.08
    grasp_cube_half_size: float = 0.021
    max_delta_pos: float = 0.5
    position_gain: float = 12.0
    orientation_gain: float = 2.0
    max_orientation_delta: float = 0.25
    gripper_action_open: float = -1.0
    robot_initialization_noise_type: str = "uniform"
    robot_initialization_noise_magnitude: float = 0.0
    gripper_light_type: str = "SPOT"
    gripper_light_energy: float = 35.0
    gripper_light_size: float = 0.015
    gripper_spot_size: float = 2.5
    gripper_spot_blend: float = 0.35
    show_gripper_light_marker: bool = False


def _range_token(prefix: str, value: tuple[float, float]) -> str:
    def fmt(number: float) -> str:
        return f"{float(number):+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")

    return f"{prefix}{fmt(value[0])}_{fmt(value[1])}"


def cube_range_folder_name(cube_range: CubeRangeConfig) -> str:
    if cube_range.name:
        return cube_range.name
    return "_".join((
        _range_token("x", cube_range.object_x_range),
        _range_token("y", cube_range.object_y_range),
    ))


def _sample_config(config: DataCollectionConfig, cube_range: CubeRangeConfig, seed: int) -> SampleDemoConfig:
    return SampleDemoConfig(
        output_dir=config.output_dir,
        width=config.width,
        height=config.height,
        samples=config.samples,
        camera_name=config.camera_name,
        seed=seed,
        n_samples=config.n_samples,
        robot_asset_dir=config.robot_asset_dir,
        settle_steps=config.settle_steps,
        use_gpu=config.use_gpu,
        render_blender=config.render_blender,
        render_mujoco=config.render_mujoco,
        render_depth=config.render_depth,
        xyz_low=config.xyz_low,
        xyz_high=config.xyz_high,
        yaw_low=config.yaw_low,
        yaw_high=config.yaw_high,
        object_x_range=cube_range.object_x_range,
        object_y_range=cube_range.object_y_range,
        object_rotation=cube_range.object_rotation,
        object_reference_pos=cube_range.object_reference_pos,
        object_z_offset=cube_range.object_z_offset,
        robot_initialization_noise_type=config.robot_initialization_noise_type,
        robot_initialization_noise_magnitude=config.robot_initialization_noise_magnitude,
        ik_max_iters=config.ik_max_iters,
        ik_position_tolerance=config.ik_position_tolerance,
        ik_orientation_tolerance=config.ik_orientation_tolerance,
        ik_damping=config.ik_damping,
        ik_step_size=config.ik_step_size,
        ik_last_joint_yaw_only=config.ik_last_joint_yaw_only,
        ik_yaw_tolerance=config.ik_yaw_tolerance,
        gripper_action_open=config.gripper_action_open,
        gripper_light_type=config.gripper_light_type,
        gripper_light_energy=config.gripper_light_energy,
        gripper_light_size=config.gripper_light_size,
        gripper_spot_size=config.gripper_spot_size,
        gripper_spot_blend=config.gripper_spot_blend,
        show_gripper_light_marker=config.show_gripper_light_marker,
    )


def _gt_policy(config: DataCollectionConfig, seed: int) -> GTPolicy:
    return GTPolicy(
        GTPolicyConfig(
            xy_threshold=config.xy_threshold,
            theta_threshold=config.theta_threshold,
            xy_threshold_noise=config.xy_threshold_noise,
            theta_threshold_noise=config.theta_threshold_noise,
            pregrasp_clearance=config.pregrasp_clearance,
            grasp_cube_half_size=config.grasp_cube_half_size,
            max_delta_pos=config.max_delta_pos,
            position_gain=config.position_gain,
            orientation_gain=config.orientation_gain,
            max_orientation_delta=config.max_orientation_delta,
            open_gripper_action=config.gripper_action_open,
        ),
        seed=seed,
    )


def _sync_robot_controller_goals(env: Any) -> None:
    env.sim.forward()
    for robot in getattr(env, "robots", []):
        controller = getattr(robot, "composite_controller", None)
        if controller is None:
            continue
        controller.update_state()
        controller.reset()


def _robot_state(env: Any) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for robot_index, robot in enumerate(getattr(env, "robots", [])):
        state: dict[str, Any] = {"robot_index": robot_index}
        for key, indexes in (
            ("arm_qpos", getattr(robot, "_ref_arm_joint_pos_indexes", None)),
            ("arm_qvel", getattr(robot, "_ref_arm_joint_vel_indexes", None)),
            ("joint_qpos", getattr(robot, "_ref_joint_pos_indexes", None)),
            ("joint_qvel", getattr(robot, "_ref_joint_vel_indexes", None)),
        ):
            if indexes is not None:
                state[key] = np.asarray(env.sim.data.qpos[indexes] if "qpos" in key else env.sim.data.qvel[indexes], dtype=np.float64).tolist()
        gripper_states: dict[str, Any] = {}
        for arm in getattr(robot, "arms", []):
            joints = getattr(robot, "gripper_joints", {}).get(arm)
            if not joints:
                continue
            qpos_indexes = [env.sim.model.get_joint_qpos_addr(joint) for joint in joints]
            qvel_indexes = [env.sim.model.get_joint_qvel_addr(joint) for joint in joints]
            gripper_states[arm] = {
                "qpos": [float(env.sim.data.qpos[index]) for index in qpos_indexes],
                "qvel": [float(env.sim.data.qvel[index]) for index in qvel_indexes],
            }
        state["grippers"] = gripper_states
        states.append(state)
    return {"robots": states}


def _axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> tuple[list[float], list[float]]:
    quat_xyzw = np.asarray(axisangle2quat(np.asarray(axis_angle, dtype=np.float64)), dtype=np.float64)
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return quat_xyzw.tolist(), quat_wxyz.tolist()


def _eef_pose_from_body(poses: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    eef_pose = poses["gripper0_right_eef"]
    return (
        np.asarray(eef_pose.position, dtype=np.float64),
        _quat_wxyz_to_mat(eef_pose.quaternion_wxyz),
    )


def _align_pregrasp_action(
    policy: GTPolicy,
    policy_state: Any,
    eef_pos: np.ndarray,
    eef_ori: np.ndarray,
    cube_position: np.ndarray,
    cube_orientation: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    policy_action = policy.act_from_pose(
        eef_pos=eef_pos,
        eef_ori=eef_ori,
        cube_position=cube_position,
        cube_orientation=cube_orientation,
        state=policy_state,
    )
    already_pregrasp = policy_action.phase != "align_pregrasp"
    if already_pregrasp:
        delta_pos = np.zeros(3, dtype=np.float64)
        orientation_delta = np.zeros(3, dtype=np.float64)
        gripper_action = 0.0
    else:
        delta_pos = np.asarray(policy_action.delta_pos, dtype=np.float64)
        orientation_delta = np.asarray(policy_action.orientation_delta, dtype=np.float64)
        gripper_action = float(policy_action.gripper_action)

    quat_xyzw, quat_wxyz = _axis_angle_to_quat_wxyz(orientation_delta)
    gt_action = {
        "phase": "pregrasp_aligned" if already_pregrasp else "align_pregrasp",
        "already_pregrasp": bool(already_pregrasp),
        "delta_pos": delta_pos.tolist(),
        "delta_quat_xyzw": quat_xyzw,
        "delta_quat_wxyz": quat_wxyz,
        "orientation_delta_axis_angle": orientation_delta.tolist(),
        "gripper_action": gripper_action,
        "target_pos": policy_action.target_pos.tolist(),
        "target_ori": policy_action.target_ori.tolist(),
        "xy_err": float(policy_action.xy_err),
        "theta_err": float(policy_action.theta_err),
        "raw_policy_action": policy_action.as_dict(),
    }
    return gt_action, policy_action


def _render_mujoco(env: Any, sample_config: SampleDemoConfig, sample_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if not sample_config.render_mujoco:
        return outputs

    rgb_path = sample_dir / "mujoco_rgb.png"
    image = env.sim.render(
        camera_name=sample_config.camera_name,
        width=sample_config.width,
        height=sample_config.height,
        depth=False,
    )
    Image.fromarray(np.flipud(image)).save(rgb_path)
    outputs["mujoco_rgb_path"] = str(rgb_path)

    if sample_config.render_depth:
        _, mujoco_depth = env.sim.render(
            camera_name=sample_config.camera_name,
            width=sample_config.width,
            height=sample_config.height,
            depth=True,
        )
        mujoco_depth = np.flipud(np.asarray(mujoco_depth, dtype=np.float32))
        mujoco_depth = _linearize_mujoco_depth(env.sim, mujoco_depth)
        depth_path = sample_dir / "mujoco_depth.exr"
        save_depth_exr(mujoco_depth, depth_path)
        outputs["mujoco_depth_path"] = str(depth_path)
    return outputs


def _sync_blender_scene(
    renderer: PrimitiveBlenderRenderer,
    env: Any,
    sample_config: SampleDemoConfig,
    body_names: list[str],
    arena_geom_names: list[str],
    cube_visual_geom_name: str,
) -> None:
    poses = collect_body_poses(env.sim, body_names)
    arena_geoms = collect_geom_states(env.sim, arena_geom_names)
    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
    camera = get_camera_state(env.sim, sample_config.camera_name)
    renderer.sync_arena_geoms(arena_geoms)
    renderer.update_poses(poses)
    renderer.sync_cube_geom(cube_geom)
    renderer.set_camera_from_mujoco(camera)


def _render_blender(
    renderer: PrimitiveBlenderRenderer | None,
    env: Any,
    sample_config: SampleDemoConfig,
    sample_dir: Path,
    body_names: list[str],
    arena_geom_names: list[str],
    cube_visual_geom_name: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if renderer is None or not sample_config.render_blender:
        return outputs

    _sync_blender_scene(renderer, env, sample_config, body_names, arena_geom_names, cube_visual_geom_name)
    light_configs = {
        "no_fingerlight": (False, False),
        "red_only": (True, False),
        "blue_only": (False, True),
    }
    for name, (left_enabled, right_enabled) in light_configs.items():
        renderer.set_gripper_finger_lights("left", left_enabled)
        renderer.set_gripper_finger_lights("right", right_enabled)
        raw_path = sample_dir / f"blender_rgb_{name}.exr"
        preview_path = sample_dir / f"blender_rgb_{name}.png"
        paths = renderer.render_rgb_outputs(raw_path, preview_path)
        outputs[f"blender_rgb_{name}_raw_exr_path"] = str(paths["raw_exr_path"])
        outputs[f"blender_rgb_{name}_preview_png_path"] = str(paths["preview_png_path"])

    if sample_config.render_depth:
        renderer.set_gripper_finger_lights("left", False)
        renderer.set_gripper_finger_lights("right", False)
        depth_path = sample_dir / "blender_depth.exr"
        outputs["blender_depth_path"] = str(renderer.render_depth(depth_path))
    return outputs


def _capture_state_metadata(
    env: Any,
    sample_config: SampleDemoConfig,
    body_names: list[str],
    arena_geom_names: list[str],
    cube_visual_geom_name: str,
) -> dict[str, Any]:
    poses = collect_body_poses(env.sim, body_names)
    arena_geoms = collect_geom_states(env.sim, arena_geom_names)
    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
    camera = get_camera_state(env.sim, sample_config.camera_name)
    return {
        "body_names": body_names,
        "arena_geom_names": arena_geom_names,
        "camera": _serialize_camera(camera),
        "poses": _serialize_poses(poses),
        "arena_geoms": _serialize_geoms(arena_geoms),
        "cube_geom": _serialize_geoms({cube_visual_geom_name: cube_geom})[cube_visual_geom_name],
        "robot_state": _robot_state(env),
    }


def _run_one_capture(
    env: Any,
    renderer: PrimitiveBlenderRenderer | None,
    rng: np.random.Generator,
    config: DataCollectionConfig,
    sample_config: SampleDemoConfig,
    cube_range: CubeRangeConfig,
    output_dir: Path,
    index: int,
) -> dict[str, Any] | None:
    env.reset()

    initial_sample = sample_eef_pose(rng, sample_config)
    initial_ik = solve_eef_ik(
        env,
        target_position=np.asarray(initial_sample.position, dtype=np.float64),
        target_orientation=np.asarray(initial_sample.orientation, dtype=np.float64),
        config=sample_config,
    )
    if not initial_ik.success:
        return None

    env.robots[0].set_robot_joint_positions(np.asarray(initial_ik.qpos, dtype=np.float64))
    set_grippers_max_open(env)
    _sync_robot_controller_goals(env)
    _settle_env(env, config.settle_steps)
    set_grippers_max_open(env)
    _sync_robot_controller_goals(env)

    body_names = default_bridge_body_names(env.sim)
    arena_geom_names = default_arena_geom_names()
    cube_visual_geom_name = default_cube_visual_geom_name()
    poses = collect_body_poses(env.sim, body_names)
    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
    eef_pos, eef_ori = _eef_pose_from_body(poses)
    cube_position = np.asarray(cube_geom.position, dtype=np.float64)
    cube_orientation = _quat_wxyz_to_mat(cube_geom.quaternion_wxyz)

    policy = _gt_policy(config, seed=config.seed + index)
    policy_state = policy.reset()
    gt_action, _ = _align_pregrasp_action(
        policy=policy,
        policy_state=policy_state,
        eef_pos=eef_pos,
        eef_ori=eef_ori,
        cube_position=cube_position,
        cube_orientation=cube_orientation,
    )

    sample_dir = output_dir / f"sample_{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    state_metadata = _capture_state_metadata(env, sample_config, body_names, arena_geom_names, cube_visual_geom_name)
    render_outputs = {}
    render_outputs.update(_render_mujoco(env, sample_config, sample_dir))
    render_outputs.update(_render_blender(renderer, env, sample_config, sample_dir, body_names, arena_geom_names, cube_visual_geom_name))

    metadata = {
        "index": index,
        "sample_dir": str(sample_dir),
        "cube_range": asdict(cube_range),
        "initial_eef_sample": asdict(initial_sample),
        "initial_ik": asdict(initial_ik),
        "policy_state": asdict(policy_state),
        "gt_action": gt_action,
        "state": state_metadata,
        "render_outputs": render_outputs,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_data_collection_worker(
    config: DataCollectionConfig,
    cube_range: CubeRangeConfig,
    env_config: RobosuiteEnvConfig | None = None,
    worker_index: int = 0,
) -> dict[str, Any]:
    seed = config.seed + worker_index * 100003
    rng = np.random.default_rng(seed)
    range_dir = Path(config.output_dir) / cube_range_folder_name(cube_range)
    range_dir.mkdir(parents=True, exist_ok=True)

    sample_config = _sample_config(config, cube_range, seed)
    env_cfg = env_config or RobosuiteEnvConfig(env_name="LumiLift", robots="Panda")
    placement_initializer = build_sample_placement_initializer(sample_config)
    initialization_noise = sample_robot_initialization_noise(sample_config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_cfg)
    renderer = None

    results: list[dict[str, Any]] = []
    attempts = 0
    try:
        if config.render_blender:
            renderer = _build_sample_renderer(sample_config)
            renderer.setup_scene()

        max_attempts = max(config.n_samples, config.n_samples * config.max_attempts_per_sample)
        while len(results) < config.n_samples and attempts < max_attempts:
            attempts += 1
            result = _run_one_capture(
                env=env,
                renderer=renderer,
                rng=rng,
                config=config,
                sample_config=sample_config,
                cube_range=cube_range,
                output_dir=range_dir,
                index=len(results),
            )
            if result is not None:
                results.append(result)
                print(f"[{cube_range_folder_name(cube_range)}] sample {len(results)}/{config.n_samples}")
    finally:
        close_env(env)

    summary = {
        "config": asdict(config),
        "sample_config": asdict(sample_config),
        "env_config": asdict(env_cfg),
        "cube_range": asdict(cube_range),
        "worker_index": worker_index,
        "seed": seed,
        "attempts": attempts,
        "n_samples": len(results),
        "range_dir": str(range_dir),
        "samples": [item["sample_dir"] for item in results],
    }
    summary_path = range_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_data_collection(
    config: DataCollectionConfig,
    cube_ranges: list[CubeRangeConfig],
    env_config: RobosuiteEnvConfig | None = None,
) -> dict[str, Any]:
    summaries = [
        run_data_collection_worker(config, cube_range, env_config=env_config, worker_index=index)
        for index, cube_range in enumerate(cube_ranges)
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": asdict(config),
        "env_config": asdict(env_config) if env_config is not None else None,
        "cube_ranges": [asdict(cube_range) for cube_range in cube_ranges],
        "workers": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
