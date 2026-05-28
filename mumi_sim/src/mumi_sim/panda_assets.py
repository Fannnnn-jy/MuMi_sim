"""本段代码用于加载数据位于 LumiGrasp/assets/robot_asset 中的机械臂数据"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class MeshSpec:
    body_name: str
    mesh_name: str
    mesh_path: Path
    material_name: str | None
    rgba: tuple[float, float, float, float] | None
    specular: float
    shininess: float
    local_position: tuple[float, float, float]
    local_quaternion_wxyz: tuple[float, float, float, float]


def load_panda_visual_meshes(asset_dir: str | Path) -> dict[str, list[MeshSpec]]:
    root_dir = Path(asset_dir)
    xml_path = root_dir / "panda.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing Panda XML: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    mesh_dir = _mesh_dir(root, root_dir)
    meshes = _parse_mesh_assets(root, mesh_dir)
    materials = _parse_materials(root)

    specs_by_body: dict[str, list[MeshSpec]] = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        return specs_by_body

    for body in worldbody.findall("body"):
        _collect_body_meshes(body, meshes, materials, specs_by_body)
    return specs_by_body


def _mesh_dir(root: ET.Element, root_dir: Path) -> Path:
    compiler = root.find("compiler")
    if compiler is None:
        return root_dir
    meshdir = compiler.get("meshdir")
    if not meshdir:
        return root_dir
    return root_dir / meshdir


def _parse_mesh_assets(root: ET.Element, mesh_dir: Path) -> dict[str, Path]:
    meshes: dict[str, Path] = {}
    asset = root.find("asset")
    if asset is None:
        return meshes

    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        mesh_name = mesh.get("name") or Path(file_attr).stem
        meshes[mesh_name] = mesh_dir / file_attr
    return meshes


def _parse_materials(root: ET.Element) -> dict[str, dict[str, object]]:
    materials: dict[str, dict[str, object]] = {}
    asset = root.find("asset")
    if asset is None:
        return materials

    for material in asset.findall("material"):
        name = material.get("name")
        rgba = material.get("rgba")
        if name and rgba:
            materials[name] = {
                "rgba": _parse_float_tuple(rgba, 4, default=(1.0, 1.0, 1.0, 1.0)),
                "specular": float(material.get("specular", "0.5")),
                "shininess": float(material.get("shininess", "0.25")),
            }
    return materials


def _collect_body_meshes(
    body: ET.Element,
    meshes: dict[str, Path],
    materials: dict[str, dict[str, object]],
    specs_by_body: dict[str, list[MeshSpec]],
) -> None:
    body_name = body.get("name")
    target_body = _map_panda_body_name(body_name)
    if target_body is not None:
        body_mesh_quat = _body_mesh_quat_correction(body)
        for geom in body.findall("geom"):
            if not _is_visual_geom(geom):
                continue
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in meshes:
                continue
            material_name = geom.get("material")
            material = materials.get(material_name) if material_name else None
            geom_pos = _parse_float_tuple(geom.get("pos"), 3, default=(0.0, 0.0, 0.0))
            geom_quat = _parse_float_tuple(geom.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
            spec = MeshSpec(
                body_name=target_body,
                mesh_name=mesh_name,
                mesh_path=meshes[mesh_name],
                material_name=material_name,
                rgba=material["rgba"] if material else None,
                specular=float(material["specular"]) if material else 0.5,
                shininess=float(material["shininess"]) if material else 0.25,
                local_position=_quat_rotate(body_mesh_quat, geom_pos),
                local_quaternion_wxyz=_quat_multiply(body_mesh_quat, geom_quat),
            )
            specs_by_body.setdefault(target_body, []).append(spec)

    for child in body.findall("body"):
        _collect_body_meshes(child, meshes, materials, specs_by_body)


def _map_panda_body_name(body_name: str | None) -> str | None:
    if body_name is None:
        return None
    if body_name.startswith("link") and body_name[4:].isdigit():
        return f"robot0_{body_name}"
    if body_name == "hand":
        return "robot0_right_hand"
    if body_name == "left_finger":
        return "gripper0_right_leftfinger"
    if body_name == "right_finger":
        return "gripper0_right_rightfinger"
    return None


def _is_visual_geom(geom: ET.Element) -> bool:
    cls = geom.get("class", "")
    if "collision" in cls:
        return False
    if "visual" in cls:
        return True
    return geom.get("material") is not None and geom.get("mesh") is not None


def _body_mesh_quat_correction(body: ET.Element) -> tuple[float, float, float, float]:
    # The Panda asset reuses the same finger mesh for both fingers and flips the
    # right finger at the body level. We drive meshes from robosuite body poses,
    # so this local visual correction must be retained for the imported mesh.
    if body.get("name") == "right_finger":
        return _parse_float_tuple(body.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
    return (1.0, 0.0, 0.0, 0.0)


def _quat_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    qvec = (0.0, vec[0], vec[1], vec[2])
    inv = (quat[0], -quat[1], -quat[2], -quat[3])
    rotated = _quat_multiply(_quat_multiply(quat, qvec), inv)
    return (rotated[1], rotated[2], rotated[3])


def _parse_float_tuple(
    value: str | None,
    expected_len: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    vals = tuple(float(x) for x in value.split())
    if len(vals) != expected_len:
        return default
    return vals
