import numpy as np
from typing import Any

from ..robosuite_env import action_bounds


def _bounded_position_delta(delta: np.ndarray, gain: float, max_delta: float) -> np.ndarray:
    command = gain * delta
    norm = float(np.linalg.norm(command))
    if norm > max_delta:
        command = command / norm * max_delta
    return command


def _bounded_orientation_delta(delta: np.ndarray, gain: float, max_delta: float) -> np.ndarray:
    command = gain * delta
    norm = float(np.linalg.norm(command))
    if norm > max_delta:
        command = command / norm * max_delta
    return command


def _osc_pose_action(
    env: Any,
    position_delta: np.ndarray,
    gripper_action: float,
    orientation_delta: np.ndarray | None = None,
) -> np.ndarray:
    low, high = action_bounds(env)
    action = np.zeros_like(low, dtype=np.float64)
    action[:3] = position_delta
    if action.shape[0] >= 7:
        if orientation_delta is not None:
            action[3:6] = orientation_delta
        action[-1] = gripper_action
    return np.clip(action, low, high)
