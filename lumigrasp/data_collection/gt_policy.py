from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from robosuite.utils.transform_utils import euler2mat

from lumigrasp.bridge.utils.control_utils import _bounded_orientation_delta, _bounded_position_delta
from lumigrasp.bridge.utils.coords import _orientation_delta_axis_angle, _select_top_down_orientation


@dataclass(frozen=True)
class GTPolicyConfig:
    xy_threshold: float = 0.01
    theta_threshold: float = 0.10
    xy_threshold_noise: float = 0.003
    theta_threshold_noise: float = 0.015
    pregrasp_clearance: float = 0.08
    grasp_cube_half_size: float = 0.021
    max_delta_pos: float = 0.5
    position_gain: float = 1.0
    orientation_gain: float = 1.0
    max_orientation_delta: float = 0.25
    close_z_threshold: float = 0.005
    lift_height: float = 0.15
    lift_success_height: float = 0.08 # Minimum height above the reference cube Z for a successful lift
    open_gripper_action: float = -1.0
    close_gripper_action: float = 1.0
    grasp_hold_steps: int = 10


@dataclass(frozen=True)
class GTPolicyState:
    xy_threshold: float
    theta_threshold: float


@dataclass(frozen=True)
class GTPolicyAction:
    delta_pos: np.ndarray
    target_pos: np.ndarray
    target_ori: np.ndarray
    orientation_delta: np.ndarray
    gripper_action: float
    phase: str
    xy_err: float
    theta_err: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_pos": self.delta_pos.tolist(),
            "target_pos": self.target_pos.tolist(),
            "target_ori": self.target_ori.tolist(),
            "orientation_delta": self.orientation_delta.tolist(),
            "gripper_action": float(self.gripper_action),
            "phase": self.phase,
            "xy_err": float(self.xy_err),
            "theta_err": float(self.theta_err),
        }


def _as_vec(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _as_mat(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}")
    return array



def cube_pose_to_matrix(cube_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cube_pose = _as_vec(cube_pos, 6, "cube_pos")
    return cube_pose[:3], euler2mat(cube_pose[3:])


def cube_position_from_pose(cube_pos: np.ndarray) -> np.ndarray:
    cube_pose = np.asarray(cube_pos, dtype=np.float64)
    if cube_pose.shape == (6,):
        return cube_pose[:3]
    if cube_pose.shape == (3,):
        return cube_pose
    raise ValueError(f"cube_pos must have shape (3,) or (6,), got {cube_pose.shape}")


def is_cube_lifted(cube_pos: np.ndarray, reference_cube_z: float, min_lift_height: float = 0.05) -> bool:
    cube_position = cube_position_from_pose(cube_pos)
    return bool(float(cube_position[2]) - float(reference_cube_z) >= float(min_lift_height))


def grasp_success(cube_pos: np.ndarray, reference_cube_z: float, min_lift_height: float = 0.05) -> bool:
    return is_cube_lifted(cube_pos, reference_cube_z=reference_cube_z, min_lift_height=min_lift_height)


def eef_pose_from_env(env: Any, site_name: str = "gripper0_right_grip_site") -> tuple[np.ndarray, np.ndarray]:
    site_id = env.sim.model.site_name2id(site_name)
    eef_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    eef_ori = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    return eef_pos, eef_ori


class GTPolicy:
    """Ground-truth grasp policy for cube pre-grasp alignment and descent.

    The image observation is intentionally not used by this oracle. The policy
    consumes the same cube pose signal used by the dataset generator and expects
    the caller to provide the current EEF pose from simulation / FK.
    """

    def __init__(self, config: GTPolicyConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GTPolicyConfig()
        self.rng = np.random.default_rng(seed)
        self.state = self.sample_state()
        self._phase = "align_pregrasp"
        self._grasp_commanded = False
        self._grasp_steps = 0
        self._lift_target_pos: np.ndarray | None = None
        self._target_ori: np.ndarray | None = None

    def sample_state(self) -> GTPolicyState:
        cfg = self.config
        xy_threshold = cfg.xy_threshold + float(self.rng.uniform(-cfg.xy_threshold_noise, cfg.xy_threshold_noise))
        theta_threshold = cfg.theta_threshold + float(
            self.rng.uniform(-cfg.theta_threshold_noise, cfg.theta_threshold_noise)
        )
        return GTPolicyState(
            xy_threshold=max(0.0, xy_threshold),
            theta_threshold=max(0.0, theta_threshold),
        )

    def reset(self) -> GTPolicyState:
        self.state = self.sample_state()
        self._phase = "align_pregrasp"
        self._grasp_commanded = False
        self._grasp_steps = 0
        self._lift_target_pos = None
        self._target_ori = None
        return self.state

    def act(
        self,
        eef_pos: np.ndarray,
        eef_ori: np.ndarray,
        cube_pos: np.ndarray,
        state: GTPolicyState | None = None,
    ) -> GTPolicyAction:
        cube_position, cube_orientation = cube_pose_to_matrix(cube_pos)
        return self.act_from_pose(
            eef_pos=eef_pos,
            eef_ori=eef_ori,
            cube_position=cube_position,
            cube_orientation=cube_orientation,
            state=state,
        )

    def act_from_pose(
        self,
        eef_pos: np.ndarray,
        eef_ori: np.ndarray,
        cube_position: np.ndarray,
        cube_orientation: np.ndarray,
        state: GTPolicyState | None = None,
    ) -> GTPolicyAction:
        cfg = self.config
        thresholds = state or self.state
        eef_position = _as_vec(eef_pos, 3, "eef_pos")
        eef_orientation = _as_mat(eef_ori, "eef_ori")
        cube_position = _as_vec(cube_position, 3, "cube_position")
        cube_orientation = _as_mat(cube_orientation, "cube_orientation")

        if self._target_ori is None:
            self._target_ori = _select_top_down_orientation(eef_orientation, cube_orientation)
        target_ori = self._target_ori

        orientation_error = _orientation_delta_axis_angle(target_ori, eef_orientation)
        xy_err = float(np.linalg.norm(eef_position[:2] - cube_position[:2]))
        theta_err = float(np.linalg.norm(orientation_error))

        pregrasp_target = cube_position + np.asarray(
            [0.0, 0.0, cfg.grasp_cube_half_size + cfg.pregrasp_clearance],
            dtype=np.float64,
        )
        pregrasp_pos_err = float(np.linalg.norm(pregrasp_target - eef_position))
        descend_pos_err = float(np.linalg.norm(cube_position - eef_position))
        pregrasp_pos_threshold = max(thresholds.xy_threshold, cfg.close_z_threshold)
        pregrasp_reached = pregrasp_pos_err <= pregrasp_pos_threshold
        orientation_reached = theta_err <= thresholds.theta_threshold
        reached_cube_z = abs(float(eef_position[2] - cube_position[2])) <= cfg.close_z_threshold

        orientation_delta = _bounded_orientation_delta(
            orientation_error,
            gain=cfg.orientation_gain,
            max_delta=cfg.max_orientation_delta,
        )
        if self._lift_target_pos is not None:
            phase = "lift"
            target_pos = self._lift_target_pos
            gripper_action = cfg.close_gripper_action
        elif self._phase == "grasp" or reached_cube_z:
            phase = "grasp"
            target_pos = eef_position.copy()
            gripper_action = cfg.close_gripper_action
            self._grasp_commanded = True
            self._grasp_steps += 1
            if self._grasp_steps >= cfg.grasp_hold_steps:
                self._lift_target_pos = eef_position + np.asarray([0.0, 0.0, cfg.lift_height], dtype=np.float64)
                self._phase = "lift"
            else:
                self._phase = "grasp"
        elif self._phase == "descend":
            phase = "descend"
            target_pos = cube_position.copy()
            gripper_action = cfg.open_gripper_action
            if descend_pos_err <= cfg.close_z_threshold:
                phase = "grasp"
                target_pos = eef_position.copy()
                gripper_action = cfg.close_gripper_action
                self._phase = "grasp"
                self._grasp_commanded = True
                self._grasp_steps = 1
        else:
            phase = "align_pregrasp"
            target_pos = pregrasp_target
            gripper_action = cfg.open_gripper_action
            self._phase = phase
            if pregrasp_reached and orientation_reached:
                phase = "descend"
                target_pos = cube_position.copy()
                self._phase = phase

        delta_pos = _bounded_position_delta(
            target_pos - eef_position,
            gain=cfg.position_gain,
            max_delta=cfg.max_delta_pos,
        )

        return GTPolicyAction(
            delta_pos=delta_pos,
            target_pos=target_pos,
            target_ori=target_ori,
            orientation_delta=orientation_delta,
            gripper_action=gripper_action,
            phase=phase,
            xy_err=xy_err,
            theta_err=theta_err,
        )

    def is_success(
        self,
        cube_pos: np.ndarray,
        reference_cube_z: float,
        min_lift_height: float | None = None,
    ) -> bool:
        threshold = self.config.lift_success_height if min_lift_height is None else min_lift_height
        return grasp_success(cube_pos, reference_cube_z=reference_cube_z, min_lift_height=threshold)

    def act_from_observation(
        self,
        observation: Mapping[str, Any],
        state: GTPolicyState | None = None,
    ) -> GTPolicyAction:
        if "eef_pos" not in observation or "eef_ori" not in observation:
            raise KeyError("GTPolicy requires observation['eef_pos'] and observation['eef_ori'] from sim / FK")
        return self.act(
            eef_pos=np.asarray(observation["eef_pos"], dtype=np.float64),
            eef_ori=np.asarray(observation["eef_ori"], dtype=np.float64),
            cube_pos=np.asarray(observation["cube_pos"], dtype=np.float64),
            state=state,
        )

    def act_from_env(
        self,
        env: Any,
        cube_pos: np.ndarray,
        state: GTPolicyState | None = None,
        site_name: str = "gripper0_right_grip_site",
    ) -> GTPolicyAction:
        eef_pos, eef_ori = eef_pose_from_env(env, site_name=site_name)
        return self.act(eef_pos=eef_pos, eef_ori=eef_ori, cube_pos=cube_pos, state=state)
