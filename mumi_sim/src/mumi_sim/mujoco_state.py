from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Pose:
    name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraState:
    name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    fovy_degrees: float


@dataclass(frozen=True)
class GeomState:
    name: str
    geom_type: int
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    material_name: str | None
    mesh_position: tuple[float, float, float] | None = None
    mesh_quaternion_wxyz: tuple[float, float, float, float] | None = None


def _name_to_id(model: Any, kind: str, name: str) -> int:
    method = getattr(model, f"{kind}_name2id", None)
    if callable(method):
        return int(method(name))

    names = getattr(model, f"{kind}_names", None)
    if names is not None:
        decoded = [n.decode() if isinstance(n, bytes) else n for n in names]
        return decoded.index(name)

    raise AttributeError(f"Cannot resolve {kind} id for {name!r}")


def _id_to_name(model: Any, kind: str, idx: int) -> str | None:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        return method(idx)
    return None


def iter_body_names(sim: Any) -> list[str]:
    model = sim.model
    names: list[str] = []
    for idx in range(int(model.nbody)):
        name = _id_to_name(model, "body", idx)
        if name:
            names.append(name)
    return names


def get_body_pose(sim: Any, body_name: str) -> Pose:
    body_id = _name_to_id(sim.model, "body", body_name)
    pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    quat = np.asarray(sim.data.body_xquat[body_id], dtype=np.float64)
    return Pose(
        name=body_name,
        position=tuple(float(x) for x in pos),
        quaternion_wxyz=tuple(float(x) for x in quat),
    )


def get_camera_state(sim: Any, camera_name: str, fovy_degrees: float | None = None) -> CameraState:
    camera_id = _name_to_id(sim.model, "camera", camera_name)
    pos = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
    xmat = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    quat = _mat_to_quat_wxyz(xmat)
    fovy = float(np.asarray(sim.model.cam_fovy)[camera_id]) if fovy_degrees is None else fovy_degrees
    return CameraState(
        name=camera_name,
        position=tuple(float(x) for x in pos),
        quaternion_wxyz=tuple(float(x) for x in quat),
        fovy_degrees=fovy,
    )


def get_geom_state(sim: Any, geom_name: str) -> GeomState:
    geom_id = _name_to_id(sim.model, "geom", geom_name)
    pos = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
    xmat = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    quat = _mat_to_quat_wxyz(xmat)
    size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float64)
    rgba = np.asarray(sim.model.geom_rgba[geom_id], dtype=np.float64)
    material_name = _material_name(sim.model, int(sim.model.geom_matid[geom_id]))
    mesh_position = None
    mesh_quaternion = None
    geom_dataid = int(sim.model.geom_dataid[geom_id])
    if int(sim.model.geom_type[geom_id]) == 7 and geom_dataid >= 0:
        mesh_position_raw = np.asarray(sim.model.mesh_pos[geom_dataid], dtype=np.float64)
        mesh_quaternion_raw = np.asarray(sim.model.mesh_quat[geom_dataid], dtype=np.float64)
        mesh_position = tuple(float(x) for x in mesh_position_raw)
        mesh_quaternion = tuple(float(x) for x in mesh_quaternion_raw)
    return GeomState(
        name=geom_name,
        geom_type=int(sim.model.geom_type[geom_id]),
        position=tuple(float(x) for x in pos),
        quaternion_wxyz=tuple(float(x) for x in quat),
        size=tuple(float(x) for x in size),
        rgba=tuple(float(x) for x in rgba),
        material_name=material_name,
        mesh_position=mesh_position,
        mesh_quaternion_wxyz=mesh_quaternion,
    )


def collect_geom_states(sim: Any, names: Iterable[str]) -> dict[str, GeomState]:
    states: dict[str, GeomState] = {}
    for name in names:
        try:
            states[name] = get_geom_state(sim, name)
        except Exception:
            continue
    return states


def default_arena_geom_names() -> list[str]:
    return [
        "floor",
        "wall_leftcorner_visual",
        "wall_rightcorner_visual",
        "wall_left_visual",
        "wall_right_visual",
        "wall_rear_visual",
        "wall_front_visual",
        "table_visual",
        "table_leg1_visual",
        "table_leg2_visual",
        "table_leg3_visual",
        "table_leg4_visual",
        "fixed_mount0_torso_vis",
        "fixed_mount0_pedestal_vis",
    ]


def default_cube_visual_geom_name() -> str:
    return "cube_g0_vis"


def collect_body_poses(sim: Any, names: Iterable[str]) -> dict[str, Pose]:
    poses: dict[str, Pose] = {}
    for name in names:
        try:
            poses[name] = get_body_pose(sim, name)
        except Exception:
            continue
    return poses


def default_bridge_body_names(sim: Any) -> list[str]:
    names = iter_body_names(sim)
    selected = []
    for name in names:
        if name == "table" or name == "cube_main":
            selected.append(name)
        elif name == "robot0_right_hand":
            selected.append(name)
        elif name.startswith("robot0_link") or name.startswith("gripper0_"):
            selected.append(name)
    return selected


def body_metadata(sim: Any, body_name: str) -> dict[str, Any]:
    body_id = _name_to_id(sim.model, "body", body_name)
    geom_ids = np.where(np.asarray(sim.model.geom_bodyid) == body_id)[0]
    geom_names = []
    for geom_id in geom_ids:
        geom_name = _id_to_name(sim.model, "geom", int(geom_id))
        if geom_name:
            geom_names.append(geom_name)
    return {"body_id": int(body_id), "geom_names": geom_names}


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _material_name(model: Any, material_id: int) -> str | None:
    if material_id < 0:
        return None
    method = getattr(model, "mat_id2name", None)
    if callable(method):
        return method(material_id)
    return None
