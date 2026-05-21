from __future__ import annotations
import numpy as np

from robosuite.utils.transform_utils import mat2quat, quat2axisangle, quat2mat


def _quat_wxyz_to_mat(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """wxyz to xyzw"""
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return quat2mat(quat_xyzw)

def _normalize_xy_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray([axis[0], axis[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError("Cannot build grasp orientation from a vertical cube axis")
    return axis / norm

def _select_top_down_orientation(current_eef_ori: np.ndarray, cube_orientation: np.ndarray) -> np.ndarray:
    """pick the top-down grasp orientation that is closest to the current EEF orientation"""
    # Columns are the desired EEF x, y, z axes in world coordinates. The EEF z
    # axis points down toward the cube; the two candidates align the gripper
    # with either cube face pair, using the cube yaw instead of world X / Y.
    cube_x = _normalize_xy_axis(cube_orientation[:, 0])
    cube_y = _normalize_xy_axis(cube_orientation[:, 1])
    down = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    candidates = []
    for eef_x in (cube_x, cube_y):
        eef_y = np.cross(down, eef_x)
        eef_y /= np.linalg.norm(eef_y)
        candidates.append(np.column_stack((eef_x, eef_y, down)))
    
    return min(candidates, key=lambda candidate: float(np.linalg.norm(_orientation_delta_axis_angle(candidate, current_eef_ori))))

def _orientation_delta_axis_angle(target_ori: np.ndarray, current_ori: np.ndarray) -> np.ndarray:
    delta_ori = np.dot(target_ori, current_ori.T)
    return quat2axisangle(mat2quat(delta_ori))

