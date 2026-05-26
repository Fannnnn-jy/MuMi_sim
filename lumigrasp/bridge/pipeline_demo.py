from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from .blender_renderer import PrimitiveBlenderRenderer
from .mujoco_state import (
    collect_body_poses,
    collect_geom_states,
    default_arena_geom_names,
    default_bridge_body_names,
    default_cube_visual_geom_name,
    get_camera_state,
    get_geom_state,
)
from .robosuite_env import (
    RobosuiteEnvConfig,
    action_bounds,
    build_demo_env,
    close_env,
    sample_random_action,
    _settle_env,
    _position_delta,
)
from .utils.depth import _linearize_mujoco_depth
from .utils.exr import save_depth_exr, save_rgb_exr
from .utils.serialization import _serialize_camera, _serialize_geoms, _serialize_poses
from .utils.control_utils import _bounded_orientation_delta, _bounded_position_delta, _osc_pose_action
from .utils.coords import _quat_wxyz_to_mat, _orientation_delta_axis_angle, _select_top_down_orientation

@dataclass(frozen=True)
class BridgeRunConfig:
    steps: int = 10
    render_every: int = 1
    output_dir: str = "outputs/bridge_demo"
    width: int = 640
    height: int = 480
    samples: int = 64
    seed: int = 0
    action_scale: float = 0.35
    action_repeat: int = 20
    no_render: bool = False
    use_gpu: bool = True
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    settle_steps: int = 0
    render_depth: bool = False
    gripper_light_type: str = "SPOT"
    gripper_light_energy: float = 35.0
    gripper_light_size: float = 0.015
    gripper_spot_size: float = 0.55
    gripper_spot_blend: float = 0.35
    show_gripper_light_marker: bool = True


def run_bridge_demo(run_config: BridgeRunConfig, env_config: RobosuiteEnvConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(run_config.seed)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    env = build_demo_env(env_config)
    renderer = None if run_config.no_render else PrimitiveBlenderRenderer(
        width=run_config.width,
        height=run_config.height,
        samples=run_config.samples,
        use_gpu=run_config.use_gpu,
        robot_asset_dir=run_config.robot_asset_dir,
        gripper_light_type=run_config.gripper_light_type,
        gripper_light_energy=run_config.gripper_light_energy,
        gripper_light_size=run_config.gripper_light_size,
        gripper_spot_size=run_config.gripper_spot_size,
        gripper_spot_blend=run_config.gripper_spot_blend,
        show_gripper_light_marker=run_config.show_gripper_light_marker,
    )

    rendered: list[str] = []
    body_names: list[str] = []
    try:
        env.reset()
        _settle_env(env, run_config.settle_steps)
        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
        if renderer is not None:
            renderer.setup_scene()
            renderer.sync_arena_geoms(arena_geoms)
            initial_poses = collect_body_poses(env.sim, body_names)
            renderer.update_poses(initial_poses)
            renderer.set_gripper_finger_lights("left", True)
            renderer.set_gripper_finger_lights("right", True)
            renderer.sync_cube_geom(cube_geom)
            initial_render_path = renderer.render(output_dir / "frame_0000.exr")
            rendered.append(str(initial_render_path))
        else:
            initial_poses = collect_body_poses(env.sim, body_names)

        with metadata_path.open("w", encoding="utf-8") as f:
            for step_idx in range(run_config.steps):
                action = sample_random_action(env, rng, scale=run_config.action_scale)
                obs: Any = {}
                reward = 0.0
                info: dict[str, Any] = {}
                repeated_steps = 0
                done = False
                while not done:
                # for _ in range(run_config.action_repeat):
                    # if done:
                        #     break
                    obs, reward, done, info = env.step(action)
                    repeated_steps += 1
                poses = collect_body_poses(env.sim, body_names)
                cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
                eef_delta = _position_delta(poses, initial_poses, "gripper0_right_eef")
                print(
                    f"Step {step_idx + 1}: action={action} "
                    f"repeat={repeated_steps} eef_delta_from_initial_m={eef_delta:.4f}"
                )
                render_path = None
                if renderer is not None and step_idx % run_config.render_every == 0:
                    renderer.update_poses(poses)
                    renderer.set_gripper_finger_lights("left", True)
                    renderer.set_gripper_finger_lights("right", True)
                    renderer.sync_cube_geom(cube_geom)
                    render_path = renderer.render(output_dir / f"frame_{step_idx + 1:04d}.exr")
                    rendered.append(str(render_path))

                f.write(json.dumps({
                    "step": step_idx + 1,
                    "action": action.tolist(),
                    "action_repeat": repeated_steps,
                    "eef_delta_from_initial_m": eef_delta,
                    "reward": float(reward),
                    "done": bool(done),
                    "render_path": str(render_path) if render_path else None,
                    "poses": _serialize_poses(poses),
                    "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                }) + "\n")
                if done:
                    break
    finally:
        close_env(env)

    return {
        "config": asdict(run_config),
        "body_names": body_names,
        "arena_geom_names": arena_geom_names,
        "settle_steps": run_config.settle_steps,
        "metadata_path": str(metadata_path),
        "rendered_frames": rendered,
    }


# ------------------------------ 静态桥接 Demo ---------------------------------
# 启动脚本位于 scripts/run_static_bridge_demo.py，配置项也在该脚本中。
# 用于静态比较 Lift + Panda 场景下 mujoco渲染和 blender 渲染的差异

@dataclass(frozen=True)
class StaticBridgeRunConfig:
    output_dir: str = "outputs/static_bridge_demo"
    width: int = 640
    height: int = 480
    samples: int = 64
    camera_name: str = "agentview"
    seed: int = 0
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    use_gpu: bool = True
    settle_steps: int = 0
    render_depth: bool = False
    gripper_light_type: str = "SPOT"
    gripper_light_energy: float = 35.0
    gripper_light_size: float = 0.015
    gripper_spot_size: float = 0.55
    gripper_spot_blend: float = 0.35
    show_gripper_light_marker: bool = True


def run_static_bridge_demo(
    run_config: StaticBridgeRunConfig,
    env_config: RobosuiteEnvConfig | None = None,
) -> dict[str, Any]:
    np.random.seed(run_config.seed)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_demo_env(env_config)
    mujoco_path = output_dir / "mujoco.exr"
    blender_path = output_dir / "blender.exr"
    metadata_path = output_dir / "metadata.json"
    mujoco_depth_path = output_dir / "mujoco_depth.exr"
    blender_depth_path = output_dir / "blender_depth.exr"

    try:
        env.reset()
        _settle_env(env, run_config.settle_steps)
        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        poses = collect_body_poses(env.sim, body_names)
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
        camera = get_camera_state(env.sim, run_config.camera_name)
        
        # print("body_names: \n", body_names)
        # print("poses: \n", poses)
        # print("arena_geoms: \n", arena_geoms)
        # print("camera: \n", camera) # CameraState(name='agentview', position=(0.5, 0.0, 1.35), quaternion_wxyz=(0.6530979720442611, 0.2710406591485372, 0.2710406591485372, 0.6530979720442611), fovy_degrees=45.0)

        mujoco_image = env.sim.render(
            camera_name=run_config.camera_name,
            width=run_config.width,
            height=run_config.height,
            depth=False,
        )
        if run_config.render_depth:
            _, mujoco_depth = env.sim.render(
                camera_name=run_config.camera_name,
                width=run_config.width,
                height=run_config.height,
                depth=True,
            )
            mujoco_depth = np.flipud(np.asarray(mujoco_depth, dtype=np.float32))
            mujoco_depth = _linearize_mujoco_depth(env.sim, mujoco_depth)
            save_depth_exr(mujoco_depth, mujoco_depth_path)

        mujoco_image = np.flipud(mujoco_image)
        save_rgb_exr(mujoco_image, mujoco_path)

        renderer = PrimitiveBlenderRenderer(
            width=run_config.width,
            height=run_config.height,
            samples=run_config.samples,
            use_gpu=run_config.use_gpu,
            robot_asset_dir=run_config.robot_asset_dir,
            gripper_light_type=run_config.gripper_light_type,
            gripper_light_energy=run_config.gripper_light_energy,
            gripper_light_size=run_config.gripper_light_size,
            gripper_spot_size=run_config.gripper_spot_size,
            gripper_spot_blend=run_config.gripper_spot_blend,
            show_gripper_light_marker=run_config.show_gripper_light_marker,
        )
        renderer.setup_scene()
        renderer.sync_arena_geoms(arena_geoms)
        renderer.update_poses(poses)
        renderer.sync_cube_geom(cube_geom)
        
        renderer.set_camera_from_mujoco(camera)

        # import time
        # start = time.perf_counter()
        # renderer.set_gripper_finger_lights("left", False)
        # renderer.set_gripper_finger_lights("right", False)
        # for _ in range(10):
        #     renderer.render(blender_path.with_name("no_light_" + blender_path.name))
        # elapsed = time.perf_counter() - start
        # print(f"Render took {elapsed:.3f}s")
        
        renderer.set_gripper_finger_lights("left", False)
        renderer.set_gripper_finger_lights("right", False)
        renderer.render(blender_path.with_name("no_light_" + blender_path.name))

        renderer.set_gripper_finger_lights("left", True)
        # renderer.set_gripper_finger_lights("right", False)
        renderer.render(blender_path.with_name("left_light_" + blender_path.name))

        renderer.set_gripper_finger_lights("left", False)
        renderer.set_gripper_finger_lights("right", True)
        renderer.render(blender_path.with_name("right_light_" + blender_path.name))

        renderer.set_gripper_finger_lights("left", True)
        # renderer.set_gripper_finger_lights("right", True)
        renderer.render(blender_path.with_name("all_light_" + blender_path.name))

        if run_config.render_depth:
            renderer.render_depth(blender_depth_path)

        metadata = {
            "config": asdict(run_config),
            "body_names": body_names,
            "arena_geom_names": arena_geom_names,
            "settle_steps": run_config.settle_steps,
            "camera": _serialize_camera(camera),
            "poses": _serialize_poses(poses),
            "arena_geoms": _serialize_geoms(arena_geoms),
            "cube_geom": _serialize_geoms({cube_visual_geom_name: cube_geom})[cube_visual_geom_name],
            "mujoco_path": str(mujoco_path),
            "blender_path": str(blender_path),
            "render_depth": run_config.render_depth,
            "depth_outputs": None if not run_config.render_depth else {
                "mujoco_depth_path": str(mujoco_depth_path),
                "blender_depth_path": str(blender_depth_path),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        close_env(env)


# ------------------------------ controller demo ---------------------------------
# 生成 EEF 笛卡尔 waypoint，由 robosuite composite controller 显式控制

DEBUG_RENDER_FINGER_CENTER = True
DEBUG_FINGER_CENTER_RADIUS = 0.003

@dataclass(frozen=True)
class PosePathRunConfig:
    output_dir: str = "outputs/pose_path_demo"
    width: int = 640
    height: int = 480
    samples: int = 64
    camera_name: str = "agentview"
    seed: int = 0
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    settle_steps: int = 0
    no_render: bool = False
    use_gpu: bool = True
    render_every: int = 5
    controller: str = "BASIC"
    # waypoint_steps: int = 40
    position_gain: float = 12.0
    max_position_delta: float = 0.03
    orientation_gain: float = 2.0
    max_orientation_delta: float = 0.25
    pregrasp_clearance: float = 0.08
    grasp_cube_half_size: float = 0.021
    gripper_action: float = 1.0
    gripper_light_type: str = "SPOT"
    gripper_light_energy: float = 35.0
    gripper_light_size: float = 0.015
    gripper_spot_size: float = 0.55
    gripper_spot_blend: float = 0.35
    show_gripper_light_marker: bool = False

def _default_pose_path_waypoints(initial_eef: np.ndarray) -> list[np.ndarray]:
    default_path = (
        (0.0, 0.0, -0.1),
        (0.0, -0.1, -0.1),
        (0.0, -0.1, 0.1),
        (0.0, 0.0, 0.0),
    )
    return [
        initial_eef + np.asarray(offset, dtype=np.float64)
        for offset in default_path
    ]

def run_pose_path_demo(
    run_config: PosePathRunConfig,
    env_config: RobosuiteEnvConfig | None = None,
) -> dict[str, Any]:
    np.random.seed(run_config.seed)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    cfg = env_config or RobosuiteEnvConfig()
    cfg = RobosuiteEnvConfig(
        env_name=cfg.env_name,
        robots=cfg.robots,
        controller=run_config.controller,
        control_freq=cfg.control_freq,
        horizon=cfg.horizon,
        use_object_obs=cfg.use_object_obs,
        has_offscreen_renderer=cfg.has_offscreen_renderer,
    )
    env = build_demo_env(cfg)
    renderer = None if run_config.no_render else PrimitiveBlenderRenderer(
        width=run_config.width,
        height=run_config.height,
        samples=run_config.samples,
        use_gpu=run_config.use_gpu,
        robot_asset_dir=run_config.robot_asset_dir,
        gripper_light_type=run_config.gripper_light_type,
        gripper_light_energy=run_config.gripper_light_energy,
        gripper_light_size=run_config.gripper_light_size,
        gripper_spot_size=run_config.gripper_spot_size,
        gripper_spot_blend=run_config.gripper_spot_blend,
        show_gripper_light_marker=run_config.show_gripper_light_marker,
    )

    rendered: list[str] = []
    body_names: list[str] = []
    try:
        env.reset()
        _settle_env(env, run_config.settle_steps)

        body_names = default_bridge_body_names(env.sim)
        arena_geom_names = default_arena_geom_names()
        cube_visual_geom_name = default_cube_visual_geom_name()
        arena_geoms = collect_geom_states(env.sim, arena_geom_names)
        cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
        camera = get_camera_state(env.sim, run_config.camera_name)
        poses = collect_body_poses(env.sim, body_names)
        initial_eef = np.asarray(poses["gripper0_right_eef"].position, dtype=np.float64)
        initial_eef_ori = _quat_wxyz_to_mat(poses["gripper0_right_eef"].quaternion_wxyz)

        object_position = np.asarray(cube_geom.position, dtype=np.float64)
        # define target position as a pre-grasp pose above the cube, with a fixed clearance ( 0.08 )
        target_position = object_position + np.asarray([0.0, 0.0, run_config.grasp_cube_half_size + run_config.pregrasp_clearance], dtype=np.float64)

        cube_orientation = _quat_wxyz_to_mat(cube_geom.quaternion_wxyz)
        target_orientation = _select_top_down_orientation(initial_eef_ori, cube_orientation)
        waypoints = [target_position]

        if renderer is not None:
            renderer.setup_scene()
            renderer.sync_arena_geoms(arena_geoms)
            renderer.update_poses(poses)
            renderer.set_gripper_finger_lights("left", True)
            renderer.set_gripper_finger_lights("right", True)
            renderer.sync_cube_geom(cube_geom)
            if DEBUG_RENDER_FINGER_CENTER:
                renderer._sync_debug_sphere(
                    "debug_finger_center",
                    tuple(float(x) for x in initial_eef),
                    radius=DEBUG_FINGER_CENTER_RADIUS,
                )
            renderer.set_camera_from_mujoco(camera)
            rendered.append(str(renderer.render(output_dir / "frame_0000.exr")))

        with metadata_path.open("w", encoding="utf-8") as f:
            step_idx = 0
            for waypoint_idx, waypoint in enumerate(waypoints):
                done = False
                while not done:
                    poses = collect_body_poses(env.sim, body_names)
                    eef_position = np.asarray(poses["gripper0_right_eef"].position, dtype=np.float64)
                    eef_orientation = _quat_wxyz_to_mat(poses["gripper0_right_eef"].quaternion_wxyz)
                    raw_orientation_delta = _orientation_delta_axis_angle(target_orientation, eef_orientation)
                    position_delta = _bounded_position_delta(
                        waypoint - eef_position,
                        gain=run_config.position_gain,
                        max_delta=run_config.max_position_delta,
                    )
                    orientation_delta = _bounded_orientation_delta(
                        raw_orientation_delta,
                        gain=run_config.orientation_gain,
                        max_delta=run_config.max_orientation_delta,
                    )
                    action = _osc_pose_action(
                        env,
                        position_delta,
                        run_config.gripper_action,
                        orientation_delta=orientation_delta,
                    )
                    obs, reward, done, info = env.step(action)
                    step_idx += 1

                    poses = collect_body_poses(env.sim, body_names)
                    cube_geom = get_geom_state(env.sim, cube_visual_geom_name)
                    eef_position = np.asarray(poses["gripper0_right_eef"].position, dtype=np.float64)
                    eef_orientation = _quat_wxyz_to_mat(poses["gripper0_right_eef"].quaternion_wxyz)
                    target_error = float(np.linalg.norm(waypoint - eef_position))
                    orientation_target_error = float(np.linalg.norm(_orientation_delta_axis_angle(target_orientation, eef_orientation)))
                    render_path = None
                    if renderer is not None and step_idx % run_config.render_every == 0:
                        renderer.update_poses(poses)
                        renderer.set_gripper_finger_lights("left", True)
                        renderer.set_gripper_finger_lights("right", True)
                        renderer.sync_cube_geom(cube_geom)
                        if DEBUG_RENDER_FINGER_CENTER:
                            renderer._sync_debug_sphere(
                                "debug_finger_center",
                                tuple(float(x) for x in eef_position),
                                radius=DEBUG_FINGER_CENTER_RADIUS,
                            )
                        render_path = renderer.render(output_dir / f"frame_{step_idx:04d}.exr")
                        rendered.append(str(render_path))

                    f.write(json.dumps({
                        "step": step_idx,
                        "waypoint": waypoint_idx,
                        "initial_eef_position": initial_eef.tolist(),
                        "object_position": np.asarray(cube_geom.position, dtype=np.float64).tolist(),
                        "target_position": waypoint.tolist(),
                        "target_orientation": target_orientation.tolist(),
                        "cube_orientation": cube_orientation.tolist(),
                        "eef_position": eef_position.tolist(),
                        "target_error": target_error,
                        "orientation_target_error": orientation_target_error,
                        "position_delta": position_delta.tolist(),
                        "orientation_delta": orientation_delta.tolist(),
                        "action": action.tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "render_path": str(render_path) if render_path else None,
                        "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
                        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
                    }) + "\n")
                    if done or (target_error < 0.003 and orientation_target_error < 0.03):
                        break
                if done:
                    break

        summary = {
            "config": asdict(run_config),
            "env_config": asdict(cfg),
            "body_names": body_names,
            "arena_geom_names": arena_geom_names,
            "initial_eef_position": initial_eef.tolist(),
            "initial_object_position": object_position.tolist(),
            "target_mode": "pregrasp_pose",
            "target_positions": [waypoint.tolist() for waypoint in waypoints],
            "target_orientation": target_orientation.tolist(),
            "cube_orientation": cube_orientation.tolist(),
            "metadata_path": str(metadata_path),
            "rendered_frames": rendered,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        close_env(env)
