from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

from .mujoco_state import Pose


class LumiLift(Lift):
    """A custom robosuite Lift environment with several necessary characteristics different from the default."""
    """A fixed-size cube is used for consistent demos."""

    cube_half_size = 0.020
    
    def __init__(self, **kwargs):
        print("Initializing LumiLift environment...")
        super(LumiLift, self).__init__(**kwargs)

    def _load_model(self) -> None:
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube_size = [self.cube_half_size] * 3
        self.cube = BoxObject(
            name="cube",
            size=cube_size,
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        ) # overwrite the default cube object with a fixed-size cube

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )


@dataclass(frozen=True)
class RobosuiteEnvConfig:
    env_name: str = "LumiLift"
    robots: str = "Panda"
    controller: str | None = None
    control_freq: int = 20
    horizon: int = 200
    use_object_obs: bool = True
    has_offscreen_renderer: bool = False


# def _resolved_env_name(env_name: str) -> str:
#     return "LumiLift" if env_name == "Lift" else env_name


def _make_env(
    cfg: RobosuiteEnvConfig,
    placement_initializer: UniformRandomSampler,
    initialization_noise: dict[str, Any],
) -> Any:
    kwargs: dict[str, Any] = {
        "env_name": cfg.env_name,
        "robots": cfg.robots,
        "has_renderer": False,
        "has_offscreen_renderer": cfg.has_offscreen_renderer,
        "use_camera_obs": False,
        "use_object_obs": cfg.use_object_obs,
        "control_freq": cfg.control_freq,
        "horizon": cfg.horizon,
        "initialization_noise": initialization_noise,
        "placement_initializer": placement_initializer,
    }
    if cfg.controller:
        kwargs["controller_configs"] = suite.load_composite_controller_config(controller=cfg.controller)
    env = suite.make(**kwargs)
    _patch_reset_open_gripper(env)
    return env


def _gripper_max_open_qpos(env: Any, robot: Any, arm: str) -> np.ndarray:
    joint_ids = np.asarray([env.sim.model.joint_name2id(joint) for joint in robot.gripper_joints[arm]], dtype=int)
    joint_ranges = np.asarray(env.sim.model.jnt_range[joint_ids], dtype=np.float64)
    init_qpos = np.asarray(robot.gripper[arm].init_qpos, dtype=np.float64)
    return np.where(init_qpos >= 0.0, joint_ranges[:, 1], joint_ranges[:, 0])


def set_grippers_max_open(env: Any) -> None:
    for robot in getattr(env, "robots", []):
        for arm in getattr(robot, "arms", []):
            if not getattr(robot, "has_gripper", {}).get(arm, False):
                continue
            max_open_qpos = _gripper_max_open_qpos(env, robot, arm)
            robot.set_gripper_joint_positions(max_open_qpos, gripper_arm=arm)
            robot.gripper[arm].current_action = np.sign(max_open_qpos).astype(np.float64)
    env.sim.forward()


def _patch_reset_open_gripper(env: Any) -> None:
    if getattr(env, "_lumigrasp_open_gripper_reset", False):
        return
    original_reset = env.reset

    def reset_with_open_gripper(*args: Any, **kwargs: Any) -> Any:
        obs = original_reset(*args, **kwargs)
        set_grippers_max_open(env)
        return obs

    env.reset = reset_with_open_gripper
    env._lumigrasp_open_gripper_reset = True


def build_demo_env(config: RobosuiteEnvConfig | None = None) -> Any:
    """Build a fixed-cube robosuite Lift environment for demos."""
    cfg = config or RobosuiteEnvConfig()
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.0, 0.0],
        y_range=[-0.0, 0.0],
        rotation=[-0.5, 0.5],
        ensure_object_boundary_in_range=False,
        reference_pos=(0.15, 0.15, 0.8),
        z_offset=0.01,
    )
    initialization_noise = {
        "type": "uniform",
        "magnitude": 0.0,
    }
    return _make_env(cfg, placement_initializer, initialization_noise)


def build_robosuite_env(
    placement_initializer: UniformRandomSampler,
    initialization_noise: dict[str, Any],
    config: RobosuiteEnvConfig | None = None,
) -> Any:
    """Build a fixed-cube robosuite env with custom placement and robot noise."""
    cfg = config or RobosuiteEnvConfig()
    return _make_env(cfg, placement_initializer, initialization_noise)


def action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray]:
    low, high = env.action_spec
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def sample_random_action(env: Any, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    low, high = action_bounds(env)
    action = rng.uniform(low=low, high=high)
    return np.clip(action * scale, low, high)


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()

def _settle_env(env: Any, steps: int) -> None:
    for _ in range(steps):
        env.sim.step()


def _position_delta(poses: dict[str, Pose], reference: dict[str, Pose], name: str) -> float:
    if name not in poses or name not in reference:
        return 0.0
    pos = np.asarray(poses[name].position)
    ref = np.asarray(reference[name].position)
    return float(np.linalg.norm(pos - ref))
