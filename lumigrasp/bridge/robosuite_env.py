from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import robosuite as suite

from .mujoco_state import Pose



@dataclass(frozen=True)
class RobosuiteEnvConfig:
    env_name: str = "Lift"
    robots: str = "Panda"
    controller: str | None = None
    control_freq: int = 20
    horizon: int = 200
    use_object_obs: bool = True
    has_offscreen_renderer: bool = False


def build_demo_env(config: RobosuiteEnvConfig | None = None) -> Any:
    """使用 robosuite 接口构造一个环境实例。 https://robosuite.ai/docs/basicusage.html"""
    cfg = config or RobosuiteEnvConfig()
    kwargs: dict[str, Any] = {
        "env_name": cfg.env_name,
        "robots": cfg.robots,
        "has_renderer": False,
        "has_offscreen_renderer": cfg.has_offscreen_renderer,
        "use_camera_obs": False,
        "use_object_obs": cfg.use_object_obs,
        "control_freq": cfg.control_freq,
        "horizon": cfg.horizon,
    }
    if cfg.controller:
        kwargs["controller_configs"] = suite.load_controller_config(default_controller=cfg.controller)
    return suite.make(**kwargs)


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