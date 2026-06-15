from __future__ import annotations

import ctypes
import gc
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from omegaconf import OmegaConf


class StepType(Enum):
    FIRST = 0
    MID = 1
    LAST = 2


class ArraySpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any
    name: str


class BoundedArraySpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any
    minimum: np.ndarray
    maximum: np.ndarray
    name: str


class ExtendedTimeStep(NamedTuple):
    step_type: StepType
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


@dataclass(frozen=True)
class LumiEnvConfig:
    mumi_sim_root: str = "/home/ubuntu/Junyi/LumiGrasp/mumi_sim"
    width: int = 84
    height: int = 84
    samples: int = 32
    camera_name: str = "agentview"
    action_repeat: int = 1
    horizon: int = 200
    control_freq: int = 20
    seed: int = 1
    use_gpu: bool = True
    robot_asset_dir: str = "/home/ubuntu/Junyi/LumiGrasp/mumi_sim/assets/robot_asset/franka_emika_panda"
    robosuite_asset_dir: str = "/home/ubuntu/Junyi/LumiGrasp/mumi_sim/assets/robosuite_assets"
    gripper_light_config_path: str = "/home/ubuntu/Junyi/LumiGrasp/mumi_sim/configs/gripper_light.json"
    observation_exposure: float = 1.0
    light_diff_exposure: float = 1.0
    depth_min: float = 0.0
    depth_max: float = 2.0
    depth_invert: bool = True
    cube_x_range: tuple[float, float] = (-0.03, 0.03)
    cube_y_range: tuple[float, float] = (-0.03, 0.03)
    cube_rotation_range: tuple[float, float] = (-0.5, 0.5)
    cube_reference_pos: tuple[float, float, float] = (0.15, 0.15, 0.8)
    cube_z_offset: float = 0.01
    robot_initialization_noise: float = 0.0
    mitsuba_flush_interval: int = 100
    observation_mode: str = 'light_diff'
    frame_stack: int = 3


def _as_tuple(value, dtype=float):
    if isinstance(value, str):
        value = value.strip("[]()").split(",")
    return tuple(dtype(v) for v in value)


def _config_from_omegaconf(cfg) -> LumiEnvConfig:
    values = OmegaConf.to_container(cfg, resolve=True)
    for key in (
        "cube_x_range",
        "cube_y_range",
        "cube_rotation_range",
        "cube_reference_pos",
    ):
        if key in values:
            values[key] = _as_tuple(values[key])
    return LumiEnvConfig(**values)


class LumiGraspEnv:
    """DrQ-v2 adapter for LumiGrasp robosuite physics plus Mitsuba light images."""

    def __init__(self, cfg: LumiEnvConfig):
        self.cfg = cfg
        self._add_mumi_sim_to_path()

        import drjit as dr
        import mitsuba as mi
        from robosuite.utils.placement_samplers import UniformRandomSampler

        from mumi_sim.mitsuba_renderer import MitsubaRenderer
        from mumi_sim.mujoco_state import (
            collect_body_poses,
            collect_geom_states,
            default_arena_geom_names,
            default_bridge_body_names,
            default_cube_visual_geom_name,
            get_camera_state,
            get_geom_state,
        )
        from mumi_sim.robosuite_env import (
            RobosuiteEnvConfig,
            build_robosuite_env,
            close_env,
        )

        self._dr = dr
        self._mi = mi
        self._collect_body_poses = collect_body_poses
        self._collect_geom_states = collect_geom_states
        self._default_arena_geom_names = default_arena_geom_names
        self._default_bridge_body_names = default_bridge_body_names
        self._default_cube_visual_geom_name = default_cube_visual_geom_name
        self._get_camera_state = get_camera_state
        self._get_geom_state = get_geom_state
        self._close_env = close_env

        np.random.seed(cfg.seed)
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=list(cfg.cube_x_range),
            y_range=list(cfg.cube_y_range),
            rotation=list(cfg.cube_rotation_range),
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=cfg.cube_reference_pos,
            z_offset=cfg.cube_z_offset,
        )
        initialization_noise = {
            "type": "uniform",
            "magnitude": cfg.robot_initialization_noise,
        }
        env_cfg = RobosuiteEnvConfig(
            horizon=cfg.horizon,
            control_freq=cfg.control_freq,
            has_offscreen_renderer=True,
        )
        self._env = build_robosuite_env(
            placement_initializer=placement_initializer,
            initialization_noise=initialization_noise,
            config=env_cfg,
        )

        self._renderer = MitsubaRenderer(
            width=cfg.width,
            height=cfg.height,
            samples=cfg.samples,
            use_gpu=cfg.use_gpu,
            robot_asset_dir=cfg.robot_asset_dir,
            robosuite_asset_dir=cfg.robosuite_asset_dir,
            gripper_light_config_path=cfg.gripper_light_config_path,
        )
        self._renderer.setup_scene()

        action_low, action_high = self._env.action_spec
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        self._action_spec = BoundedArraySpec(
            shape=action_low.shape,
            dtype=np.float32,
            minimum=np.full(action_low.shape, -1.0, dtype=np.float32),
            maximum=np.full(action_high.shape, 1.0, dtype=np.float32),
            name="action",
        )
        self._raw_action_low = action_low
        self._raw_action_high = action_high
        obs_channels = self._observation_channels()
        self._observation_spec = BoundedArraySpec(
            shape=(obs_channels, cfg.height, cfg.width),
            dtype=np.uint8,
            minimum=np.zeros((obs_channels, cfg.height, cfg.width), dtype=np.uint8),
            maximum=np.full((obs_channels, cfg.height, cfg.width), 255, dtype=np.uint8),
            name="observation",
        )
        self._last_observation: np.ndarray | None = None
        self._rgb_frames = deque([], maxlen=int(cfg.frame_stack))
        self._num_render_observations = 0

    def _add_mumi_sim_to_path(self) -> None:
        src = Path(self.cfg.mumi_sim_root) / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        self._env.reset()
        self._rgb_frames.clear()
        observation = self._render_observation_by_mode()
        action = np.zeros(self._action_spec.shape, dtype=np.float32)
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=observation,
            action=action,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        env_action = self._scale_action(action)
        reward = 0.0
        done = False
        for _ in range(self.cfg.action_repeat):
            _, step_reward, done, _ = self._env.step(env_action)
            reward += float(step_reward)
            if done:
                break
        observation = self._render_observation_by_mode()
        return ExtendedTimeStep(
            step_type=StepType.LAST if done else StepType.MID,
            reward=reward,
            discount=0.0 if done else 1.0,
            observation=observation,
            action=action,
        )

    def render(self):
        if self._last_observation is None:
            self._last_observation = self._render_observation_by_mode()
        frame = self._last_observation[-3:] if self._use_rgb_stack() else self._last_observation[:3]
        return frame.transpose(1, 2, 0)

    def cube_height_above_table(self) -> float:
        sim = self._env.sim
        cube_geom = self._get_geom_state(sim, self._default_cube_visual_geom_name())
        return float(cube_geom.position[2] - self._env.table_offset[2])

    def close(self):
        self._close_env(self._env)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        midpoint = (self._raw_action_high + self._raw_action_low) * 0.5
        half_range = (self._raw_action_high - self._raw_action_low) * 0.5
        return midpoint + action * half_range


    def _use_rgb_stack(self) -> bool:
        return str(self.cfg.observation_mode).lower() in {'rgb', 'rgb_stack'}

    def _use_rgbd(self) -> bool:
        return str(self.cfg.observation_mode).lower() == 'rgbd'

    def _observation_channels(self) -> int:
        if self._use_rgb_stack():
            return 3 * int(self.cfg.frame_stack)
        if self._use_rgbd():
            return 4
        return 9

    def _render_observation_by_mode(self) -> np.ndarray:
        if self._use_rgb_stack():
            return self._render_observation_rgb()
        if self._use_rgbd():
            return self._render_observation_rgbd()
        return self._render_observation()

    def _sync_renderer(self) -> None:
        sim = self._env.sim
        body_names = self._default_bridge_body_names(sim)
        arena_geom_names = self._default_arena_geom_names()
        cube_visual_geom_name = self._default_cube_visual_geom_name()
        poses = self._collect_body_poses(sim, body_names)
        arena_geoms = self._collect_geom_states(sim, arena_geom_names)
        cube_geom = self._get_geom_state(sim, cube_visual_geom_name)
        camera = self._get_camera_state(sim, self.cfg.camera_name)

        self._renderer.sync_arena_geoms(arena_geoms)
        self._renderer.update_poses(poses)
        self._renderer.sync_cube_geom(cube_geom)
        self._renderer.set_camera_from_mujoco(camera)

    def _render_observation(self) -> np.ndarray:
        """ambient + left diff light + right diff light, each with 3 channels RGB"""
        self._sync_renderer()
        scene_dict = self._renderer._scene_dict("path")
        scene, params = self._renderer._mi_load_dict(scene_dict)
        try:
            ambient = self._render_lights(scene, params, left_on=False, right_on=False)
            left = self._render_lights(scene, params, left_on=True, right_on=False)
            right = self._render_lights(scene, params, left_on=False, right_on=True)
            left_diff = np.maximum(left - ambient, 0.0)
            right_diff = np.maximum(right - ambient, 0.0)
            obs = np.concatenate(
                [
                    self._do_gamma_and_to_chw_uint8(ambient, self.cfg.observation_exposure),
                    self._do_gamma_and_to_chw_uint8(left_diff, self.cfg.light_diff_exposure),
                    self._do_gamma_and_to_chw_uint8(right_diff, self.cfg.light_diff_exposure),
                ],
                axis=0,
            )
        finally:
            del scene, params

        ### visualization for debugging
        # import imageio.v2 as imageio
        # vis_dir = Path("debug_obs_vis")
        # vis_dir.mkdir(parents=True, exist_ok=True)
        # ambient_vis = np.transpose(obs[0:3], (1, 2, 0))
        # left_diff_vis = np.transpose(obs[3:6], (1, 2, 0))
        # right_diff_vis = np.transpose(obs[6:9], (1, 2, 0))
        # vis = np.concatenate([ambient_vis, left_diff_vis, right_diff_vis], axis=1)
        # imageio.imwrite(vis_dir / "obs_debug.png", vis)
        # print("saved debug obs to", vis_dir / "obs_debug.png")

        self._last_observation = obs
        self._num_render_observations += 1
        self._maybe_flush_mitsuba_cache()
        # self.clean_up()
        return obs
    
    def _render_observation_rgb(self) -> np.ndarray:
        """Temporal RGB stack, matching DrQ-v2's 3-frame pixel input."""
        self._sync_renderer()
        scene_dict = self._renderer._scene_dict("path")
        scene, params = self._renderer._mi_load_dict(scene_dict)
        try:
            ambient = self._render_lights(scene, params, left_on=False, right_on=False)
            rgb = self._do_gamma_and_to_chw_uint8(ambient, self.cfg.observation_exposure)
            if len(self._rgb_frames) == 0:
                for _ in range(int(self.cfg.frame_stack)):
                    self._rgb_frames.append(rgb)
            else:
                self._rgb_frames.append(rgb)
            obs = np.concatenate(list(self._rgb_frames), axis=0)
        finally:
            del scene, params

        # import imageio.v2 as imageio
        # vis_dir = Path("debug_obs_vis")
        # vis_dir.mkdir(parents=True, exist_ok=True)
        # if self._num_render_observations >= 3:
        #     f0 = obs[0:3].astype(np.uint8).transpose(1, 2, 0)  # H,W,3
        #     f1 = obs[3:6].astype(np.uint8).transpose(1, 2, 0)
        #     f2 = obs[6:9].astype(np.uint8).transpose(1, 2, 0)
        #     print(np.abs(f1 - f0).sum(), np.abs(f2 - f0).sum())
        #     vis = np.concatenate([f0, f1, f2], axis=1)
        #     imageio.imwrite(vis_dir / "obs_debug.png", vis)
        #     print("saved debug obs to", vis_dir / "obs_debug.png")
        #     assert False

        self._last_observation = obs
        self._num_render_observations += 1
        self._maybe_flush_mitsuba_cache()
        # self.clean_up()
        return obs

    def _render_observation_rgbd(self) -> np.ndarray:
        """Ambient RGB plus one metric depth channel."""
        self._sync_renderer()
        scene_rgb_dict = self._renderer._scene_dict("path")
        scene_depth_dict = self._renderer._scene_dict("depth")
        scene_rgb, params_rgb = self._renderer._mi_load_dict(scene_rgb_dict)
        scene_depth, params_depth = self._renderer._mi_load_dict(scene_depth_dict)

        try:
            ambient = self._render_lights(scene_rgb, params_rgb, left_on=False, right_on=False)
            depth = self._render_depth(scene_depth, params_depth)
            obs = np.concatenate(
                [
                    self._do_gamma_and_to_chw_uint8(ambient, self.cfg.observation_exposure),
                    self._depth_to_chw_uint8(depth),
                ],
                axis=0,
            )
        finally:
            del scene_rgb, params_rgb, scene_depth, params_depth

        # ## visualization for debugging
        # import imageio.v2 as imageio
        # vis_dir = Path("debug_obs_vis")
        # vis_dir.mkdir(parents=True, exist_ok=True)
        # ambient_vis = np.transpose(obs[0:3], (1, 2, 0))
        # depth_vis = np.repeat(np.transpose(obs[3:4], (1, 2, 0)), 3, axis=2)
        # vis = np.concatenate([ambient_vis, depth_vis], axis=1)
        # imageio.imwrite(vis_dir / "obs_debug.png", vis)
        # print("saved debug obs to", vis_dir / "obs_debug.png")

        self._last_observation = obs
        self._num_render_observations += 1
        self._maybe_flush_mitsuba_cache()
        return obs
    
    def _maybe_flush_mitsuba_cache(self) -> None:
        interval = int(self.cfg.mitsuba_flush_interval)
        if interval <= 0 or self._num_render_observations % interval != 0:
            return
        gc.collect()
        self._dr.flush_malloc_cache()
        # Repeated mi.load_dict() calls allocate substantial native host memory.
        # Dr.Jit's cache flush does not return glibc heap arenas to the OS.
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass
    
    # def clean_up(self) -> None:
    #     """renference from https://github.com/mitsuba-renderer/mitsuba3/discussions/1171"""
    #     """not working......"""
    #     '''
    #     Ensure no leftover instances from other tests are still in registry, wait
    #     for all running kernels to finish and reset the JitFlag to default. Also
    #     periodically frees the malloc cache to prevent the testcases from hogging
    #     all system memory.
    #     '''
    #     gc.collect()
    #     gc.collect()

    #     self._dr.kernel_history_clear()
    #     self._dr.flush_malloc_cache()
    #     # self._dr.malloc_clear_statistics()
    #     self._dr.flush_kernel_cache()

    #     if hasattr(self._dr, 'sync_thread'):
    #         self._dr.sync_thread()
    #         # self._dr.registry_clear()
    #         # self._dr.set_flags(self._dr.JitFlag.Default)

    def _render_lights(self, scene, params, left_on: bool, right_on: bool) -> np.ndarray:
        self._renderer.set_gripper_finger_lights(params, "left", left_on)
        self._renderer.set_gripper_finger_lights(params, "right", right_on)
        image = self._mi.render(scene, params=params)
        self._dr.eval(image)
        self._dr.sync_thread()
        np_image = np.asarray(image, dtype=np.float32).copy()
        del image
        return np.nan_to_num(np_image, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _render_depth(self, scene, params) -> np.ndarray:
        image = self._mi.render(scene, params=params)
        self._dr.eval(image)
        self._dr.sync_thread()
        depth = np.asarray(image, dtype=np.float32).copy()
        del image
        depth = np.nan_to_num(
            depth,
            nan=float(self.cfg.depth_max),
            posinf=float(self.cfg.depth_max),
            neginf=float(self.cfg.depth_min),
            copy=False,
        )
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

    @staticmethod
    def _do_gamma_and_to_chw_uint8(image: np.ndarray, exposure: float, gamma: float = 2.2) -> np.ndarray:
        image = np.maximum(image * float(exposure), 0.0)
        image = np.clip(image, 0.0, 1.0)
        # linear -> display-like gamma encoding
        image = np.power(image, 1.0 / gamma)

        return (image * 255.0).astype(np.uint8).transpose(2, 0, 1).copy()

    def _depth_to_chw_uint8(self, depth: np.ndarray) -> np.ndarray:
        depth_min = float(self.cfg.depth_min)
        depth_max = float(self.cfg.depth_max)
        if depth_max <= depth_min:
            raise ValueError(f"depth_max must be greater than depth_min, got {depth_max} <= {depth_min}")
        depth = np.clip(depth, depth_min, depth_max)
        depth = (depth - depth_min) / (depth_max - depth_min)
        if self.cfg.depth_invert:
            depth = 1.0 - depth
        return (depth * 255.0).astype(np.uint8)[None, :, :].copy()


def make(cfg):
    return LumiGraspEnv(_config_from_omegaconf(cfg))
