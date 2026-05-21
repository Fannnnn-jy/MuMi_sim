from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.placement_samplers import UniformRandomSampler

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
from lumigrasp.bridge.utils.depth import _linearize_mujoco_depth
from lumigrasp.bridge.utils.exr import save_depth_exr
from lumigrasp.bridge.utils.serialization import _serialize_camera, _serialize_geoms, _serialize_poses


@dataclass(frozen=True)
class eefSample:
    position: list[float]
    yaw: float
    orientation: list[list[float]]


@dataclass(frozen=True)
class IKresult:
    success: bool
    qpos: list[float]
    position_error: float
    orientation_error: float
    iterations: int


@dataclass(frozen=True)
class SampleDemoConfig:
    output_dir: str = "outputs/sample_demo"
    width: int = 512
    height: int = 512
    samples: int = 64
    camera_name: str = "agentview"
    seed: int = 0
    n_samples: int = 1
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    settle_steps: int = 0
    use_gpu: bool = False
    render_blender: bool = True
    render_mujoco: bool = True
    render_depth: bool = True
    xyz_low: tuple[float, float, float] = None
    xyz_high: tuple[float, float, float] = None
    yaw_low: float = -np.pi
    yaw_high: float = np.pi
    object_x_range: tuple[float, float] = (-0.15, 0.15)
    object_y_range: tuple[float, float] = (-0.15, 0.15)
    object_rotation: tuple[float, float] | float | None = (-np.pi, np.pi)
    object_reference_pos: tuple[float, float, float] = (0.0, 0.0, 0.8)
    object_z_offset: float = 0.01
    robot_initialization_noise_type: str = "uniform"
    robot_initialization_noise_magnitude: float = 0.0
    ik_max_iters: int = 250
    ik_position_tolerance: float = 0.003
    ik_orientation_tolerance: float = 0.03
    ik_damping: float = 1e-3
    ik_step_size: float = 0.6
    ik_last_joint_yaw_only: bool = True
    ik_yaw_tolerance: float = 0.03
    gripper_action_open: float = 1.0
    gripper_light_type: str = "SPOT"
    gripper_light_energy: float = 35.0
    gripper_light_size: float = 0.015
    gripper_spot_size: float = 2.5
    gripper_spot_blend: float = 0.35
    show_gripper_light_marker: bool = False


def top_down_orientation_from_yaw(yaw: float) -> np.ndarray:
    """Return a top-down EEF orientation whose in-plane x axis has yaw."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    eef_x = np.asarray([c, s, 0.0], dtype=np.float64)
    eef_z = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    eef_y = np.cross(eef_z, eef_x)
    eef_y /= np.linalg.norm(eef_y)
    return np.column_stack((eef_x, eef_y, eef_z))


def sample_eef_pose(rng: np.random.Generator, config: SampleDemoConfig) -> eefSample:
    position = rng.uniform(np.asarray(config.xyz_low), np.asarray(config.xyz_high)).astype(np.float64)
    yaw = float(rng.uniform(config.yaw_low, config.yaw_high))
    orientation = top_down_orientation_from_yaw(yaw)
    return eefSample(position=position.tolist(), yaw=yaw, orientation=orientation.tolist())


def build_sample_placement_initializer(config: SampleDemoConfig) -> UniformRandomSampler:
    return UniformRandomSampler(
        name="ObjectSampler",
        x_range=list(config.object_x_range),
        y_range=list(config.object_y_range),
        rotation=config.object_rotation,
        ensure_object_boundary_in_range=False,
        reference_pos=tuple(config.object_reference_pos),
        z_offset=config.object_z_offset,
    )


def sample_robot_initialization_noise(config: SampleDemoConfig) -> dict[str, Any]:
    return {
        "type": config.robot_initialization_noise_type,
        "magnitude": config.robot_initialization_noise_magnitude,
    }


def _site_pose(sim: Any, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = sim.model.site_name2id(site_name)
    pos = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
    mat = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat


def _site_jacobian(sim: Any, site_name: str) -> np.ndarray:
    jacp = np.asarray(sim.data.get_site_jacp(site_name), dtype=np.float64).reshape(3, sim.model.nv)
    jacr = np.asarray(sim.data.get_site_jacr(site_name), dtype=np.float64).reshape(3, sim.model.nv)
    return np.vstack((jacp, jacr))


def _damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    lhs = jacobian @ jacobian.T + (damping ** 2) * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(lhs, error)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _top_down_yaw_from_orientation(orientation: np.ndarray) -> float:
    x_axis = np.asarray(orientation[:, 0], dtype=np.float64).copy()
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-8:
        return 0.0
    x_axis /= norm
    return float(np.arctan2(x_axis[1], x_axis[0]))


def _remove_tool_yaw_component(rot_error: np.ndarray, current_orientation: np.ndarray) -> np.ndarray:
    tool_z = np.asarray(current_orientation[:, 2], dtype=np.float64)
    norm = float(np.linalg.norm(tool_z))
    if norm < 1e-8:
        return rot_error
    tool_z = tool_z / norm
    return rot_error - float(np.dot(rot_error, tool_z)) * tool_z


def _top_down_orientation_from_xy_yaw(yaw: float) -> np.ndarray:
    return top_down_orientation_from_yaw(yaw)


def _equivalent_gripper_yaws(yaw: float) -> tuple[float, ...]:
    return (
        _wrap_angle(yaw),
        _wrap_angle(yaw + np.pi),
        _wrap_angle(yaw - np.pi),
    )


def _select_reachable_yaw(
    target_yaw: float,
    current_yaw: float,
    current_q7: float,
    yaw_derivative: float,
    q7_low: float,
    q7_high: float,
) -> float:
    candidates = []
    for candidate in _equivalent_gripper_yaws(target_yaw):
        yaw_error = _wrap_angle(candidate - current_yaw)
        q7_goal = current_q7 + yaw_error / yaw_derivative
        limit_violation = max(float(q7_low - q7_goal), 0.0, float(q7_goal - q7_high))
        candidates.append((limit_violation, abs(yaw_error), candidate))
    return float(min(candidates, key=lambda item: (item[0], item[1]))[2])


def _yaw_derivative_for_joint(
    env: Any,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    qpos: np.ndarray,
    site_name: str,
    joint_index: int,
    low: np.ndarray,
    high: np.ndarray,
    eps: float = 1e-4,
) -> float:
    base = qpos.copy()
    plus = base.copy()
    minus = base.copy()
    plus[joint_index] = np.clip(plus[joint_index] + eps, low[joint_index], high[joint_index])
    minus[joint_index] = np.clip(minus[joint_index] - eps, low[joint_index], high[joint_index])

    env.sim.data.qpos[qpos_idx] = plus
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()
    _, plus_orientation = _site_pose(env.sim, site_name)
    plus_yaw = _top_down_yaw_from_orientation(plus_orientation)

    env.sim.data.qpos[qpos_idx] = minus
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()
    _, minus_orientation = _site_pose(env.sim, site_name)
    minus_yaw = _top_down_yaw_from_orientation(minus_orientation)

    env.sim.data.qpos[qpos_idx] = base
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()

    actual_eps = float(plus[joint_index] - minus[joint_index])
    if abs(actual_eps) < 1e-8:
        return 1.0
    derivative = _wrap_angle(plus_yaw - minus_yaw) / actual_eps
    if abs(derivative) < 1e-6:
        return 1.0
    return float(derivative)


def solve_eef_ik(
    env: Any,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    config: SampleDemoConfig,
    site_name: str = "gripper0_right_grip_site",
) -> IKresult:
    """Solve Panda arm qpos for a sampled top-down EEF pose.

    By default, the first six joints solve Cartesian position plus the tool-z
    top-down constraint, while the last arm joint is the only joint directly
    used to match the sampled in-plane yaw.
    """
    robot = env.robots[0]
    qpos_idx = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
    qvel_idx = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=int)
    qpos = np.asarray(env.sim.data.qpos[qpos_idx], dtype=np.float64).copy()
    initial_qpos = qpos.copy()

    low = np.asarray(env.sim.model.jnt_range[robot._ref_arm_joint_indexes, 0], dtype=np.float64)
    high = np.asarray(env.sim.model.jnt_range[robot._ref_arm_joint_indexes, 1], dtype=np.float64)

    target_position = np.asarray(target_position, dtype=np.float64)
    target_orientation = np.asarray(target_orientation, dtype=np.float64)
    target_yaw = _top_down_yaw_from_orientation(target_orientation)
    last_joint = len(qpos) - 1
    solve_joint_slice = slice(None, last_joint) if config.ik_last_joint_yaw_only else slice(None)
    yaw_target_orientation = target_orientation.copy()

    pos_error_norm = float("inf")
    ori_error_norm = float("inf")
    success = False
    iteration = 0

    for iteration in range(1, config.ik_max_iters + 1):
        env.sim.data.qpos[qpos_idx] = qpos
        env.sim.data.qvel[qvel_idx] = 0.0
        env.sim.forward()

        current_position, current_orientation = _site_pose(env.sim, site_name)
        current_yaw = _top_down_yaw_from_orientation(current_orientation)
        if config.ik_last_joint_yaw_only:
            yaw_derivative = _yaw_derivative_for_joint(
                env, qpos_idx, qvel_idx, qpos, site_name, last_joint, low, high
            )
            selected_yaw = _select_reachable_yaw(
                target_yaw, current_yaw, qpos[last_joint], yaw_derivative, low[last_joint], high[last_joint]
            )
            yaw_target_orientation = _top_down_orientation_from_xy_yaw(selected_yaw)
        else:
            selected_yaw = target_yaw
            yaw_target_orientation = target_orientation

        pos_error = target_position - current_position
        ori_error = orientation_error(yaw_target_orientation, current_orientation)
        pos_error_norm = float(np.linalg.norm(pos_error))
        ori_error_norm = float(np.linalg.norm(ori_error))
        yaw_error = _wrap_angle(selected_yaw - current_yaw)
        if (
            pos_error_norm <= config.ik_position_tolerance
            and ori_error_norm <= config.ik_orientation_tolerance
            and abs(yaw_error) <= config.ik_yaw_tolerance
        ):
            success = True
            break

        jacobian = _site_jacobian(env.sim, site_name)[:, qvel_idx]
        if config.ik_last_joint_yaw_only:
            tilt_error = _remove_tool_yaw_component(ori_error, current_orientation)
            task_jacobian = np.vstack((jacobian[:3, solve_joint_slice], jacobian[3:, solve_joint_slice]))
            task_error = np.concatenate((pos_error, tilt_error))
            dq = _damped_least_squares(task_jacobian, task_error, config.ik_damping)
            qpos[solve_joint_slice] = np.clip(
                qpos[solve_joint_slice] + config.ik_step_size * dq,
                low[solve_joint_slice],
                high[solve_joint_slice],
            )

            env.sim.data.qpos[qpos_idx] = qpos
            env.sim.data.qvel[qvel_idx] = 0.0
            env.sim.forward()
            _, yaw_orientation = _site_pose(env.sim, site_name)
            current_yaw = _top_down_yaw_from_orientation(yaw_orientation)
            yaw_derivative = _yaw_derivative_for_joint(
                env, qpos_idx, qvel_idx, qpos, site_name, last_joint, low, high
            )
            selected_yaw = _select_reachable_yaw(
                target_yaw, current_yaw, qpos[last_joint], yaw_derivative, low[last_joint], high[last_joint]
            )
            yaw_error = _wrap_angle(selected_yaw - current_yaw)
            qpos[last_joint] = float(np.clip(
                qpos[last_joint] + config.ik_step_size * yaw_error / yaw_derivative,
                low[last_joint],
                high[last_joint],
            ))
        else:
            twist_error = np.concatenate((pos_error, ori_error))
            dq = _damped_least_squares(jacobian, twist_error, config.ik_damping)
            qpos = np.clip(qpos + config.ik_step_size * dq, low, high)

    env.sim.data.qpos[qpos_idx] = qpos if success else initial_qpos
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()

    return IKresult(
        success=success,
        qpos=qpos.tolist(),
        position_error=pos_error_norm,
        orientation_error=ori_error_norm,
        iterations=iteration,
    )


# def apply_robot_qpos(env: Any, qpos: list[float]) -> None:
#     env.robots[0].set_robot_joint_positions(np.asarray(qpos, dtype=np.float64))


def _build_sample_renderer(config: SampleDemoConfig) -> PrimitiveBlenderRenderer:
    return PrimitiveBlenderRenderer(
        width=config.width,
        height=config.height,
        samples=config.samples,
        use_gpu=config.use_gpu,
        robot_asset_dir=config.robot_asset_dir,
        gripper_light_type=config.gripper_light_type,
        gripper_light_energy=config.gripper_light_energy,
        gripper_light_size=config.gripper_light_size,
        gripper_spot_size=config.gripper_spot_size,
        gripper_spot_blend=config.gripper_spot_blend,
        show_gripper_light_marker=config.show_gripper_light_marker,
    )


def render_sample_state(
    env: Any,
    config: SampleDemoConfig,
    output_dir: Path,
    renderer: PrimitiveBlenderRenderer | None = None,
) -> dict[str, Any]:
    body_names = default_bridge_body_names(env.sim)
    arena_geom_names = default_arena_geom_names()
    cube_visual_geom_name = default_cube_visual_geom_name()
    poses = collect_body_poses(env.sim, body_names)
    arena_geoms = collect_geom_states(env.sim, arena_geom_names)
    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
    camera = get_camera_state(env.sim, config.camera_name)

    outputs: dict[str, Any] = {
        "body_names": body_names,
        "arena_geom_names": arena_geom_names,
        "camera": _serialize_camera(camera),
        "poses": _serialize_poses(poses),
        "arena_geoms": _serialize_geoms(arena_geoms),
        "cube_geom": _serialize_geoms({cube_visual_geom_name: cube_geom})[cube_visual_geom_name],
    }

    if config.render_mujoco:
        mujoco_rgb_path = output_dir / "mujoco_rgb.png"
        image = env.sim.render(
            camera_name=config.camera_name,
            width=config.width,
            height=config.height,
            depth=False,
        )
        Image.fromarray(np.flipud(image)).save(mujoco_rgb_path)
        outputs["mujoco_rgb_path"] = str(mujoco_rgb_path)

        if config.render_depth:
            _, mujoco_depth = env.sim.render(
                camera_name=config.camera_name,
                width=config.width,
                height=config.height,
                depth=True,
            )
            mujoco_depth = np.flipud(np.asarray(mujoco_depth, dtype=np.float32))
            mujoco_depth = _linearize_mujoco_depth(env.sim, mujoco_depth)
            mujoco_depth_path = output_dir / "mujoco_depth.exr"
            save_depth_exr(mujoco_depth, mujoco_depth_path)
            outputs["mujoco_depth_path"] = str(mujoco_depth_path)

    if config.render_blender:
        blender_rgb_raw_path = output_dir / "blender_rgb_raw.exr"
        blender_rgb_preview_path = output_dir / "blender_rgb_preview.png"
        if renderer is None:
            renderer = _build_sample_renderer(config)
            renderer.setup_scene()
        renderer.sync_arena_geoms(arena_geoms)
        renderer.update_poses(poses)
        renderer.sync_cube_geom(cube_geom)
        renderer.set_gripper_finger_lights("left", True)
        renderer.set_gripper_finger_lights("right", True)
        renderer.set_camera_from_mujoco(camera)
        renderer.render_rgb_outputs(blender_rgb_raw_path, blender_rgb_preview_path)
        outputs["blender_rgb_raw_path"] = str(blender_rgb_raw_path)
        outputs["blender_rgb_preview_path"] = str(blender_rgb_preview_path)

        if config.render_depth:
            blender_depth_path = output_dir / "blender_depth.exr"
            renderer.render_depth(blender_depth_path)
            outputs["blender_depth_path"] = str(blender_depth_path)

    return outputs


def _run_one_sample(
    env: Any,
    rng: np.random.Generator,
    config: SampleDemoConfig,
    output_dir: Path,
    index: int,
    renderer: PrimitiveBlenderRenderer | None = None,
) -> dict[str, Any]:
    env.reset()
    _settle_env(env, config.settle_steps)

    sample = sample_eef_pose(rng, config)
    target_position = np.asarray(sample.position, dtype=np.float64)
    target_orientation = np.asarray(sample.orientation, dtype=np.float64)
    ik = solve_eef_ik(env, target_position, target_orientation, config)
    if ik.success:
        env.robots[0].set_robot_joint_positions(np.asarray(ik.qpos, dtype=np.float64))
        set_grippers_max_open(env)

    set_grippers_max_open(env)
    sample_dir = output_dir / f"sample_{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_outputs = render_sample_state(env, config, sample_dir, renderer=renderer)
    sample_metadata = {
        "index": index,
        "sample_dir": str(sample_dir),
        "sample": asdict(sample),
        "ik": asdict(ik),
        "render_outputs": render_outputs,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(sample_metadata, indent=2), encoding="utf-8")
    return sample_metadata


def run_sample_demo(config: SampleDemoConfig, env_config: RobosuiteEnvConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = env_config or RobosuiteEnvConfig(env_name="LumiLift", robots="Panda")
    placement_initializer = build_sample_placement_initializer(config)
    initialization_noise = sample_robot_initialization_noise(config)
    env = build_robosuite_env(placement_initializer, initialization_noise, env_cfg)
    try:
        renderer = None
        if config.render_blender:
            renderer = _build_sample_renderer(config)
            renderer.setup_scene()

        results = [
            _run_one_sample(env, rng, config, output_dir, index, renderer=renderer)
            for index in range(config.n_samples)
        ]

        metadata = {
            "config": asdict(config),
            "env_config": asdict(env_cfg),
            "n_samples": config.n_samples,
            "n_success": sum(1 for item in results if item["ik"]["success"]),
            "samples": results,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["metadata_path"] = str(metadata_path)
        return metadata
    finally:
        close_env(env)
