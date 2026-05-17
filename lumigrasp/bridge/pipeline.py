from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
from .robosuite_env import RobosuiteEnvConfig, build_demo_env, close_env, sample_random_action, _settle_env, _position_delta
from .utils.depth import _linearize_mujoco_depth, _save_depth_l2_outputs, _save_depth_png
from .utils.serialization import _serialize_camera, _serialize_geoms, _serialize_poses

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
            renderer.sync_cube_geom(cube_geom)
            initial_render_path = renderer.render(output_dir / "frame_0000.png")
            rendered.append(str(initial_render_path))
        else:
            initial_poses = collect_body_poses(env.sim, body_names)

        with metadata_path.open("w", encoding="utf-8") as f:
            for step_idx in range(run_config.steps):
                action = sample_random_action(env, rng, scale=run_config.action_scale)
                obs: Any = {}
                reward = 0.0
                done = False
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
                    renderer.sync_cube_geom(cube_geom)
                    render_path = renderer.render(output_dir / f"frame_{step_idx + 1:04d}.png")
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
    mujoco_path = output_dir / "mujoco.png"
    blender_path = output_dir / "blender.png"
    metadata_path = output_dir / "metadata.json"
    mujoco_depth_path = output_dir / "mujoco_depth.npy"
    mujoco_depth_png_path = output_dir / "mujoco_depth.png"
    blender_depth_path = output_dir / "blender_depth.npy"
    blender_depth_exr_path = output_dir / "blender_depth.exr"
    blender_depth_png_path = output_dir / "blender_depth.png"
    depth_l2_error_path = output_dir / "depth_l2_error.npy"
    depth_l2_error_png_path = output_dir / "depth_l2_error.png"

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
        mujoco_depth = None
        if run_config.render_depth:
            _, mujoco_depth = env.sim.render(
                camera_name=run_config.camera_name,
                width=run_config.width,
                height=run_config.height,
                depth=True,
            )
            mujoco_depth = np.flipud(np.asarray(mujoco_depth, dtype=np.float32))
            mujoco_depth = _linearize_mujoco_depth(env.sim, mujoco_depth)
            np.save(mujoco_depth_path, mujoco_depth)
            _save_depth_png(mujoco_depth, mujoco_depth_png_path)
            
        mujoco_image = np.flipud(mujoco_image)
        Image.fromarray(mujoco_image).save(mujoco_path)

        renderer = PrimitiveBlenderRenderer(
            width=run_config.width,
            height=run_config.height,
            samples=run_config.samples,
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
        renderer.render(blender_path)

        depth_metrics = None
        if run_config.render_depth:
            blender_depth = renderer.render_depth(blender_depth_exr_path)
            np.save(blender_depth_path, blender_depth)
            _save_depth_png(blender_depth, blender_depth_png_path)
            depth_metrics = _save_depth_l2_outputs(
                mujoco_depth=mujoco_depth,
                blender_depth=blender_depth,
                error_path=depth_l2_error_path,
                error_png_path=depth_l2_error_png_path,
            )

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
                "mujoco_depth_png_path": str(mujoco_depth_png_path),
                "blender_depth_path": str(blender_depth_path),
                "blender_depth_exr_path": str(blender_depth_exr_path),
                "blender_depth_png_path": str(blender_depth_png_path),
                "depth_l2_error_path": str(depth_l2_error_path),
                "depth_l2_error_png_path": str(depth_l2_error_png_path),
                "metrics": depth_metrics,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        close_env(env)
