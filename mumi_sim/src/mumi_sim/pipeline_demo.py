from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
# from .blender_renderer import PrimitiveBlenderRenderer
from .mitsuba_renderer import MitsubaRenderer

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
    build_demo_env,
    _settle_env,
    close_env,
    # sample_random_action,
    # action_bounds,
    # _position_delta,
)
from .utils.depth import _linearize_mujoco_depth
from .utils.exr import save_depth_exr, save_rgb_exr
from .utils.serialization import _serialize_camera, _serialize_geoms, _serialize_poses
from .utils.control_utils import _bounded_orientation_delta, _bounded_position_delta, _osc_pose_action
from .utils.coords import _quat_wxyz_to_mat, _orientation_delta_axis_angle, _select_top_down_orientation

@dataclass(frozen=True)
class StaticBridgeRunConfig:
    output_dir: str = "outputs/static_bridge_demo"
    width: int = 84
    height: int = 84
    samples: int = 128
    camera_name: str = "agentview"
    seed: int = 0
    # robot_asset_dir: str | None = "assets/robot_asset/robotstudio_so101"
    robot_asset_dir: str | None = "assets/robot_asset/franka_emika_panda"
    use_gpu: bool = True
    settle_steps: int = 100
    render_depth: bool = True

    preview_exposure: float = 0.5
    preview_tonemap: str = "auto_exposure"
    auto_exposure_percentile: float = 0.98



def run_static_bridge_demo(
    run_config: StaticBridgeRunConfig,
    env_config: RobosuiteEnvConfig | None = None,
) -> dict[str, Any]:
    np.random.seed(run_config.seed)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_demo_env(env_config)
    mujoco_path = output_dir / "mujoco.exr"
    mitsuba_path = output_dir / "mitsuba.exr"
    metadata_path = output_dir / "metadata.json"
    mujoco_depth_path = output_dir / "mujoco_depth.exr"
    mitsuba_depth_path = output_dir / "mitsuba_depth.exr"

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

        renderer = MitsubaRenderer(
            width=run_config.width,
            height=run_config.height,
            samples=run_config.samples,
            use_gpu=run_config.use_gpu,
            robot_asset_dir=run_config.robot_asset_dir,
            preview_exposure=run_config.preview_exposure,
            preview_tonemap=run_config.preview_tonemap,
            auto_exposure_percentile=run_config.auto_exposure_percentile,
        )
        renderer.setup_scene()
        renderer.sync_arena_geoms(arena_geoms)
        renderer.update_poses(poses)
        renderer.sync_cube_geom(cube_geom)
        renderer.set_camera_from_mujoco(camera)


        # # naive rendering
        # renderer.set_gripper_finger_lights("left", False)
        # renderer.set_gripper_finger_lights("right", False)
        # renderer.render(mitsuba_path.with_name("no_light_" + mitsuba_path.name))

        # renderer.set_gripper_finger_lights("left", True)
        # renderer.set_gripper_finger_lights("right", False)
        # renderer.render(mitsuba_path.with_name("left_light_" + mitsuba_path.name))

        # renderer.set_gripper_finger_lights("left", False)
        # renderer.set_gripper_finger_lights("right", True)
        # renderer.render(mitsuba_path.with_name("right_light_" + mitsuba_path.name))

        # renderer.set_gripper_finger_lights("left", True)
        # renderer.set_gripper_finger_lights("right", True)
        # renderer.render(mitsuba_path.with_name("all_light_" + mitsuba_path.name))

        # load_dict-render detached rendering
        import time
        t0 = time.perf_counter()
        scene_dict_rgb = renderer._scene_dict("path")
        t1 = time.perf_counter()
        # scene_dict_depth = renderer._scene_dict("depth")

        # scene_depth = renderer._mi_load_dict(scene_dict_depth)
        scene_rgb, params_rgb = renderer._mi_load_dict(scene_dict_rgb)
        t2 = time.perf_counter()

        # renderer.set_gripper_finger_lights(None, "left", True)
        # renderer.set_gripper_finger_lights(None, "right", True)

        cases = [
            ("no_light_", False, False),
            ("left_light_", True, False),
            ("right_light_", False, True),
            ("all_light_", True, True),
        ]

        case_timings = []

        for prefix, left_on, right_on in cases:
            c0 = time.perf_counter()

            renderer.set_gripper_finger_lights(params_rgb, "left", left_on)
            renderer.set_gripper_finger_lights(params_rgb, "right", right_on)

            c1 = time.perf_counter()

            renderer.render_with_scene(
                scene=scene_rgb,
                params=params_rgb,
                output_path=mitsuba_path.with_name(prefix + mitsuba_path.name),
                render_depth=False,
            )

            c2 = time.perf_counter()

            case_timings.append((prefix, c1 - c0, c2 - c1, c2 - c0))

        # depth_time = None
        # if run_config.render_depth:
        #     d0 = time.perf_counter()
        #     renderer.render_depth(mitsuba_depth_path)
        #     d1 = time.perf_counter()
        #     depth_time = d1 - d0

        t_total1 = time.perf_counter()

        print("----------Batch Mitsuba Timing----------")
        print(f"build scene_dict:  {(t1 - t0) * 1000:.1f} ms")
        print(f"load_dict+traverse:{(t2 - t1) * 1000:.1f} ms")

        for prefix, set_t, render_t, total_t in case_timings:
            print(f"{prefix:<12} set lights: {set_t * 1000:.1f} ms | render/write: {render_t * 1000:.1f} ms | total: {total_t * 1000:.1f} ms")

        # if depth_time is not None:
        #     print(f"depth render:      {depth_time * 1000:.1f} ms")

        print(f"Total batch render time: {(t_total1 - t0) * 1000:.1f} ms")


        ### debug
        # xml = env.model.get_xml()
        # for processor in env._xml_processors:
        #     xml = processor(xml)
        # with open("robosuite_lift_processed.xml", "w") as f:
        #     f.write(xml)

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
            "mitsuba_path": str(mitsuba_path),
            "render_depth": run_config.render_depth,
            "depth_outputs": None if not run_config.render_depth else {
                "mujoco_depth_path": str(mujoco_depth_path),
                "mitsuba_depth_path": str(mitsuba_depth_path),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        close_env(env)

if __name__ == "__main__":
    result = run_static_bridge_demo(
        StaticBridgeRunConfig(),
        RobosuiteEnvConfig(has_offscreen_renderer=True),
    )
    print(RobosuiteEnvConfig(has_offscreen_renderer=True))
    # print(json.dumps(result, indent=2))
