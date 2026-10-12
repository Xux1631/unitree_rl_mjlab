#!/usr/bin/env python3
"""Rebuild the SolidWorks-exported URDF into an A2-interface-compatible URDF.

The SolidWorks export (`urdf/robot_a2.urdf`) has physically correct joint axes
but is not A2-interface compatible:

1. it is expressed in a **Y-up** frame (x = fore/aft with front at -x, y = up,
   z = lateral), while the A2 reference uses the ROS Z-up frame (x = forward,
   y = left, z = up);
2. it declares every axis as ``0 0 1`` in each child link's own frame instead of
   the reference's literal ``1 0 0`` / ``0 1 0``;
3. it has no ``*_foot`` links, no sensor links, no effort/velocity fields and
   joint limits of +-1.57 instead of the reference table in INTERFACE.md.

This script rewrites the link/joint frames so that every declared axis is
literally the reference axis while leaving the physical geometry, joint origins
and inertias untouched, then applies the reference limits, adds the foot and
sensor links, and emits the joints in the mandated ``RL -> FL -> RR -> FR`` /
``hip -> thigh -> calf`` order.

Run:  python tools/build_a2_urdf.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO / "src" / "assets" / "robots" / "quadruped_3320"
SRC_URDF = ASSET_DIR / "source" / "robot_solidworks_export.urdf"
SRC_ZIP = ASSET_DIR / "source" / "meshes_stl.zip"
DEFAULT_OUT = ASSET_DIR

ROBOT_NAME = "quadruped_3320"

LEGS = ("RL", "FL", "RR", "FR")  # mandated policy order
FAMILIES = ("hip", "thigh", "calf")

# Reference axis for each joint family (INTERFACE.md section 2).
REF_AXIS = {
    "hip": np.array([1.0, 0.0, 0.0]),
    "thigh": np.array([0.0, 1.0, 0.0]),
    "calf": np.array([0.0, 1.0, 0.0]),
}

# INTERFACE.md section 2: joint -> (lower, upper, effort, velocity)
REF_LIMITS = {
    "RL_hip_joint": (-1.01, 1.01, 120.0, 22.0),
    "RL_thigh_joint": (-1.56, 3.94, 120.0, 22.0),
    "RL_calf_joint": (-2.77, -0.54, 180.0, 14.6667),
    "FL_hip_joint": (-1.01, 1.01, 120.0, 22.0),
    "FL_thigh_joint": (-2.34, 3.15, 120.0, 22.0),
    "FL_calf_joint": (-2.77, -0.54, 180.0, 14.6667),
    "RR_hip_joint": (-1.01, 1.01, 120.0, 22.0),
    "RR_thigh_joint": (-1.56, 3.94, 120.0, 22.0),
    "RR_calf_joint": (-2.77, -0.54, 180.0, 14.6667),
    "FR_hip_joint": (-1.01, 1.01, 120.0, 22.0),
    "FR_thigh_joint": (-2.34, 3.15, 120.0, 22.0),
    "FR_calf_joint": (-2.77, -0.54, 180.0, 14.6667),
}

FOOT_RADIUS = 0.032  # A2 reference foot contact sphere
FOOT_DZ = 0.0  # sphere centre offset along the leg axis, metres
FOOT_MASS = 0.046

# A2 reference default stance (a2_constants.INIT_STATE) and the fraction of the
# straight leg length that A2 keeps at that stance.
SOLE_CACHE: dict = {}
FOOT_CACHE: dict = {}
A2_STANCE_THIGH = 0.9
A2_STANCE_CALF = -1.8
STAND_EXTENSION_RATIO = 0.598

# Global map: old (Y-up export) -> new (ROS Z-up).
#   new_x = -old_x (front becomes +x), new_y = old_z (left), new_z = old_y (up)
M_ROT = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Sensor link placement, in the *source* Y-up frame (metres), taken from the
# assembly part bounding boxes in parts.json.
SENSOR_JOINTS = [
    ("camera_joint", "base_link", "camera_link", np.array([-0.2961, 0.2075, 0.0])),
    ("front_lidar_joint", "base_link", "front_lidar_link", np.array([-0.28465, 0.15995, 0.0])),
    ("rear_lidar_joint", "front_lidar_link", "rear_lidar_link", np.array([0.0, 0.0, 0.0])),
]


def _floats(s, default="0 0 0"):
    """URDF attributes are occasionally empty/whitespace; default them."""
    if s is None or not str(s).strip():
        s = default
    return [float(v) for v in str(s).split()]


def rpy_to_R(rpy) -> np.ndarray:
    return Rot.from_euler("xyz", _floats(rpy)).as_matrix()


def R_to_rpy(R: np.ndarray):
    return Rot.from_matrix(R).as_euler("xyz")


def fmt(v) -> str:
    v = float(v)
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.9g}"


def fmt_vec(v) -> str:
    return " ".join(fmt(x) for x in v)


# --------------------------------------------------------------------------
# In-memory model
# --------------------------------------------------------------------------


@dataclass
class Geom:
    kind: str  # "mesh" | "box" | "sphere"
    p: np.ndarray
    R: np.ndarray
    params: dict
    tag: str  # "visual" | "collision"


@dataclass
class Link:
    name: str
    inertial: tuple | None = None  # (p, R, mass, I)
    geoms: list = field(default_factory=list)

    def geom_nodes(self, tag: str):
        return [g for g in self.geoms if g.tag == tag]


@dataclass
class Joint:
    name: str
    jtype: str  # "revolute" | "fixed"
    parent: str
    child: str
    p: np.ndarray
    R: np.ndarray
    axis: np.ndarray | None = None
    limits: tuple | None = None


# --------------------------------------------------------------------------
# Step 1: parse the SolidWorks export
# --------------------------------------------------------------------------


def parse_source(path: Path):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    links: dict[str, Link] = {}

    for le in root.findall("link"):
        name = le.get("name")
        if name == "world":
            continue
        lnk = Link(name=name)
        ine = le.find("inertial")
        if ine is not None:
            o = ine.find("origin")
            p = np.array(
                _floats(o.get("xyz") if o is not None else None)
            )
            R = rpy_to_R(o.get("rpy") if o is not None else None)
            mass = float(ine.find("mass").get("value"))
            i = ine.find("inertia")
            I = np.array(
                [
                    [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                    [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                    [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
                ]
            )
            lnk.inertial = (p, R, mass, I)

        for tag in ("visual", "collision"):
            for ge in le.findall(tag):
                o = ge.find("origin")
                p = np.array(
                    _floats(o.get("xyz") if o is not None else None)
                )
                R = rpy_to_R(o.get("rpy") if o is not None else None)
                g = ge.find("geometry")
                mesh = g.find("mesh")
                if mesh is not None:
                    scale = mesh.get("scale")
                    params = {
                        "file": Path(mesh.get("filename")).name,
                        "scale": [float(v) for v in scale.split()] if scale else [1.0, 1.0, 1.0],
                    }
                    lnk.geoms.append(Geom("mesh", p, R, params, tag))
                else:
                    child = list(g)[0]
                    if child.tag == "sphere":
                        params = {"radius": float(child.get("radius"))}
                    elif child.tag == "box":
                        params = {"size": _floats(child.get("size"))}
                    else:
                        raise NotImplementedError(child.tag)
                    lnk.geoms.append(Geom(child.tag, p, R, params, tag))
        links[name] = lnk

    joints: list[Joint] = []
    for je in root.findall("joint"):
        if je.get("type") == "floating":
            continue
        o = je.find("origin")
        p = np.array(_floats(o.get("xyz") if o is not None else None))
        R = rpy_to_R(o.get("rpy") if o is not None else None)
        ax = je.find("axis")
        axis = np.array(_floats(ax.get("xyz"))) if ax is not None else None
        joints.append(
            Joint(
                name=je.get("name"),
                jtype=je.get("type"),
                parent=je.find("parent").get("link"),
                child=je.find("child").get("link"),
                p=p,
                R=R,
                axis=axis,
            )
        )
    return links, joints


# --------------------------------------------------------------------------
# Foot contact point, computed in the ORIGINAL Y-up frames
# --------------------------------------------------------------------------


def foot_sole_points(links, joints, zip_path: Path):
    """Return per-leg sole point in that calf link's ORIGINAL frame."""
    import trimesh

    with zipfile.ZipFile(zip_path) as zf:
        blob = {}
        for leg in LEGS:
            g = links[f"{leg}_calf"].geom_nodes("visual")[0]
            blob[leg] = zf.read("meshes/" + g.params["file"])

    import io

    soles = {}
    for leg in LEGS:
        calf = links[f"{leg}_calf"]
        vis = calf.geom_nodes("visual")[0]
        mesh = trimesh.load(io.BytesIO(blob[leg]), file_type="stl", force="mesh")
        V = mesh.vertices * np.asarray(vis.params["scale"], dtype=float)
        V = (vis.R @ V.T).T + vis.p  # in the ORIGINAL calf frame
        T = fk(links, joints, {})[f"{leg}_calf"]
        Vw = (T[:3, :3] @ V.T).T + T[:3, 3]
        # Y-up export: the lowest y slab is the foot sole.
        ymin = Vw[:, 1].min()
        sel = Vw[:, 1] < ymin + 0.006
        soles[leg] = V[sel].mean(axis=0)
        # report the raw contact point too
        soles[leg + "_tip"] = V[int(np.argmin(Vw[:, 1]))]
    return soles


# --------------------------------------------------------------------------
# Forward kinematics on the in-memory model
# --------------------------------------------------------------------------


def fk(links, joints, q: dict, root="base_link"):
    children = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)
    out = {root: np.eye(4)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for j in children.get(parent, []):
            T = out[parent].copy()
            Tj = np.eye(4)
            Tj[:3, :3] = j.R
            Tj[:3, 3] = j.p
            T = T @ Tj
            if j.jtype == "revolute":
                Rq = np.eye(4)
                Rq[:3, :3] = Rot.from_rotvec(j.axis * q.get(j.name, 0.0)).as_matrix()
                T = T @ Rq
            out[j.child] = T
            stack.append(j.child)
    return out


# --------------------------------------------------------------------------
# Step 2: re-frame so every declared axis is literally the reference axis
# --------------------------------------------------------------------------


def accumulated_rotation(joints, root, link):
    """R of root -> link at q=0, using the current (pre-rewrite) frames."""
    by_child = {j.child: j for j in joints}
    ch = []
    cur = link
    while cur != root:
        j = by_child[cur]
        ch.append(j)
        cur = j.parent
    A = np.eye(3)
    for j in ch[::-1]:
        A = A @ j.R
    return A


def reframe(links, joints, root="base_link"):
    """Rewrite frames so declared axes equal REF_AXIS.

    Each link L gets an extra local rotation E_L; its new orientation is
    A'_L = M_ROT @ A_L @ E_L.  For actuated joints we require the physical axis
    A'_L @ a_ref to equal a_ref, which fixes E_L = align(a_ref -> A_L^T M^T a_ref).
    """
    children = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)

    E = {root: M_ROT.T}  # base_link ends up aligned with the ROS world

    def visit(link):
        for j in children.get(link, []):
            if j.jtype == "revolute":
                A_L = accumulated_rotation(joints, root, link) @ j.R
                fam = next(f for f in FAMILIES if f in j.name)
                a_ref = REF_AXIS[fam]
                target = A_L.T @ M_ROT.T @ a_ref
                E[j.child] = Rot.align_vectors(target[None, :], a_ref[None, :])[0].as_matrix()
            else:
                E[j.child] = np.eye(3)
            visit(j.child)

    visit(root)

    for name, lnk in links.items():
        El = E[name]
        if lnk.inertial is not None:
            p, R, mass, I = lnk.inertial
            lnk.inertial = (El.T @ p, El.T @ R, mass, El.T @ I @ El)
        for g in lnk.geoms:
            g.p = El.T @ g.p
            g.R = El.T @ g.R

    for j in joints:
        Ep = E[j.parent]
        j.R = Ep.T @ j.R @ E[j.child]
        j.p = Ep.T @ j.p
        if j.axis is not None:
            fam = next(f for f in FAMILIES if f in j.name)
            j.axis = REF_AXIS[fam].copy()

    return E


# --------------------------------------------------------------------------
# Step 3: add foot + sensor links and set the reference limits
# --------------------------------------------------------------------------


def add_aux_links(links, joints, E, soles):
    for leg in LEGS:
        sl = E[f"{leg}_calf"].T @ soles[leg]  # sole in the NEW calf frame
        up = -sl / np.linalg.norm(sl)
        centre = sl + (FOOT_RADIUS + FOOT_DZ) * up

        foot = Link(name=f"{leg}_foot")
        I = np.eye(3) * (0.4 * FOOT_MASS * FOOT_RADIUS**2)
        foot.inertial = (np.zeros(3), np.eye(3), FOOT_MASS, I)
        foot.geoms.append(
            Geom("sphere", np.zeros(3), np.eye(3), {"radius": FOOT_RADIUS}, "collision")
        )
        links[foot.name] = foot
        FOOT_CACHE[leg] = centre
        joints.append(
            Joint(
                name=f"{leg}_foot_joint",
                jtype="fixed",
                parent=f"{leg}_calf",
                child=f"{leg}_foot",
                p=centre,
                R=np.eye(3),
            )
        )

    for jname, parent, child, p_old in SENSOR_JOINTS:
        links[child] = Link(name=child)
        joints.append(
            Joint(
                name=jname, jtype="fixed", parent=parent, child=child, p=M_ROT @ p_old, R=np.eye(3)
            )
        )


def apply_limits(joints):
    for j in joints:
        if j.jtype == "revolute":
            j.limits = REF_LIMITS[j.name]


# --------------------------------------------------------------------------
# Step 4: joint zero offsets so q=0 reproduces the A2 neutral pose
# --------------------------------------------------------------------------


def solve_offsets(links, joints, soles):
    """Per-leg joint zero offsets.

    The SolidWorks assembly is NOT modelled in a symmetric stance: the FL leg
    sits in a visibly different pose from the other three (its foot is ~8 cm
    higher).  Using the as-modelled pose as the joint zero would therefore give
    an asymmetric robot.  Instead, for every leg we

      1. solve for the raw joint angles that place the foot directly under the
         thigh joint at A2's default standing extension, then
      2. shift that leg's zeros so those angles become exactly the A2 reference
         default stance (hip -+0.1, thigh 0.9, calf -1.8).

    After the shift the same joint command produces the same symmetric pose on
    every leg, and the reference limits keep their meaning (q_calf = 0 is the
    fully extended knee, exactly as in A2, which is why -0.54 is the upper stop).
    """
    from scipy.optimize import least_squares

    T0 = fk(links, joints, {})
    offsets, report = {}, []

    # The CAD's legs are not all the same length (the front thighs are ~7.5 mm
    # shorter than the rear ones), so scale the stance drop off the MEAN leg and
    # use one common absolute foot height.  Every foot then lands level.
    geom = {}
    for leg in LEGS:
        pt = FOOT_CACHE[leg]
        tj = T0[f"{leg}_thigh"][:3, 3]
        knee = T0[f"{leg}_calf"][:3, 3]
        foot = T0[f"{leg}_calf"][:3, :3] @ pt + T0[f"{leg}_calf"][:3, 3]
        geom[leg] = (pt, float(np.linalg.norm(knee - tj)), float(np.linalg.norm(foot - knee)))
    mean_drop = STAND_EXTENSION_RATIO * float(
        np.mean([lt + lc for _pt, lt, lc in geom.values()])
    )
    hip_z = float(T0[f"{LEGS[0]}_hip"][2, 3])
    target_z = hip_z - mean_drop

    for leg in LEGS:
        names = [f"{leg}_{f}_joint" for f in FAMILIES]
        pt, lt, lc = geom[leg]

        def resid(x, leg=leg, names=names, pt=pt, tz=target_z):
            T = fk(links, joints, dict(zip(names, x)))
            tj = T[f"{leg}_thigh"][:3, 3]
            foot = T[f"{leg}_calf"][:3, :3] @ pt + T[f"{leg}_calf"][:3, 3]
            return [foot[0] - tj[0], foot[1] - tj[1], foot[2] - tz]

        sol = least_squares(resid, [0.0, 0.0, 0.0], method="lm", max_nfev=40000)
        hip_sign = -0.1 if leg in ("RL", "FL") else 0.1
        want = np.array([hip_sign, A2_STANCE_THIGH, A2_STANCE_CALF])
        offsets.update({n: float(t - q) for n, t, q in zip(names, want, sol.x)})
        report.append((leg, lt, lc, sol.x, want, float(np.abs(np.asarray(sol.fun)).max())))
    return offsets, report


def apply_offsets(joints, offsets):
    """new_q = old_q + theta0  <=>  insert Rot(axis, -theta0) after the origin."""
    for j in joints:
        if j.jtype != "revolute" or j.name not in offsets:
            continue
        th = offsets[j.name]
        j.R = j.R @ Rot.from_rotvec(j.axis * -th).as_matrix()


# --------------------------------------------------------------------------
# Serialise
# --------------------------------------------------------------------------

ORDER = (
    [f"{leg}_{f}_joint" for leg in LEGS for f in FAMILIES]
    + [f"{leg}_foot_joint" for leg in LEGS]
    + ["camera_joint", "front_lidar_joint", "rear_lidar_joint"]
)


def serialize(links, joints) -> str:
    by_name = {j.name: j for j in joints}
    missing = [n for n in ORDER if n not in by_name]
    assert not missing, f"missing joints: {missing}"

    lines = [
        f'<robot name="{ROBOT_NAME}">',
        "  <mujoco>",
        '    <compiler meshdir="../meshes" discardvisual="false" angle="radian" '
        'balanceinertia="true"/>',
        "  </mujoco>",
    ]

    for lnk in links.values():
        lines.append(f'  <link name="{lnk.name}">')
        if lnk.inertial is not None:
            p, R, mass, I = lnk.inertial
            lines += [
                "    <inertial>",
                f'      <origin xyz="{fmt_vec(p)}" rpy="{fmt_vec(R_to_rpy(R))}" />',
                f'      <mass value="{fmt(mass)}" />',
                f'      <inertia ixx="{fmt(I[0, 0])}" ixy="{fmt(I[0, 1])}" ixz="{fmt(I[0, 2])}"'
                f' iyy="{fmt(I[1, 1])}" iyz="{fmt(I[1, 2])}" izz="{fmt(I[2, 2])}" />',
                "    </inertial>",
            ]
        for tag in ("visual", "collision"):
            for g in lnk.geom_nodes(tag):
                lines.append(f"    <{tag}>")
                lines.append(f'      <origin xyz="{fmt_vec(g.p)}" rpy="{fmt_vec(R_to_rpy(g.R))}" />')
                lines.append("      <geometry>")
                if g.kind == "mesh":
                    lines.append(
                        f'        <mesh filename="{g.params["file"]}" '
                        f'scale="{fmt_vec(g.params["scale"])}" />'
                    )
                elif g.kind == "box":
                    lines.append(f'        <box size="{fmt_vec(g.params["size"])}" />')
                elif g.kind == "sphere":
                    lines.append(f'        <sphere radius="{fmt(g.params["radius"])}" />')
                elif g.kind == "cylinder":
                    lines.append(
                        f'        <cylinder radius="{fmt(g.params["radius"])}" '
                        f'length="{fmt(g.params["length"])}" />'
                    )
                else:
                    raise NotImplementedError(g.kind)
                lines.append("      </geometry>")
                lines.append(f"    </{tag}>")
        lines.append("  </link>")

    for jname in ORDER:
        j = by_name[jname]
        lines.append("")
        lines.append(f'  <joint name="{j.name}" type="{j.jtype}">')
        if j.jtype == "revolute":
            lo, hi, eff, vel = j.limits
            lines.append(
                f'    <limit lower="{fmt(lo)}" upper="{fmt(hi)}" '
                f'effort="{fmt(eff)}" velocity="{fmt(vel)}" />'
            )
        lines.append(f'    <parent link="{j.parent}" />')
        lines.append(f'    <child link="{j.child}" />')
        lines.append(f'    <origin xyz="{fmt_vec(j.p)}" rpy="{fmt_vec(R_to_rpy(j.R))}" />')
        if j.jtype == "revolute":
            lines.append(f'    <axis xyz="{fmt_vec(j.axis)}" />')
        lines.append("  </joint>")

    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def copy_meshes(out_dir: Path, zip_path: Path):
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.startswith("meshes/") and n.endswith(".stl")]
        for n in names:
            target = mesh_dir / Path(n).name
            if not target.exists():
                with zf.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return mesh_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source", type=Path, default=SRC_URDF)
    ap.add_argument("--zip", type=Path, default=SRC_ZIP)
    ap.add_argument("--no-offsets", action="store_true")
    ap.add_argument("--keep-mesh-collisions", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out
    (out_dir / "urdf").mkdir(parents=True, exist_ok=True)
    mesh_dir = copy_meshes(out_dir, args.zip)
    dec = decimate_meshes(mesh_dir)
    for nm, before, after in dec:
        print(f"decimated {nm}: {before} -> {after} faces")
    print(f"meshes  -> {mesh_dir} ({len(list(mesh_dir.glob('*.stl')))} files)")

    links, joints = parse_source(args.source)
    print(f"parsed  : {len(links)} links / {len(joints)} joints")

    soles = foot_sole_points(links, joints, args.zip)
    E = reframe(links, joints)
    soi = {k: E[k + "_calf"].T @ soles[k] for k in LEGS}
    SOLE_CACHE.update(soi)
    add_aux_links(links, joints, E, soles)
    if not args.keep_mesh_collisions:
        for nm, before, desc in simplify_collisions(links, joints, mesh_dir):
            print(f"collision {nm}: {before} mesh geoms -> {desc}")
    apply_limits(joints)

    offsets = {}
    if not args.no_offsets:
        offsets, report = solve_offsets(links, joints, soi)
        for leg, lt, lc, q, want, res in report:
            print(
                f"  {leg}: Lt={lt:.4f} Lc={lc:.4f} | stance raw="
                f"{np.degrees(q[0]):7.2f} {np.degrees(q[1]):7.2f} {np.degrees(q[2]):7.2f} deg"
                f" -> A2 {np.degrees(want[0]):6.2f} {np.degrees(want[1]):6.2f}"
                f" {np.degrees(want[2]):7.2f} deg | ik residual={res * 1000:5.2f} mm"
            )
        apply_offsets(joints, offsets)

    text = serialize(links, joints)
    target = out_dir / "urdf" / f"{ROBOT_NAME}.urdf"
    target.write_text(text)
    print(f"wrote   -> {target}  ({len(text)} bytes)")

    mjcf = serialize_mjcf(links, joints)
    mjcf_target = out_dir / "xmls" / f"{ROBOT_NAME}.xml"
    mjcf_target.parent.mkdir(parents=True, exist_ok=True)
    mjcf_target.write_text(mjcf)
    print(f"wrote   -> {mjcf_target}  ({len(mjcf)} bytes)")

    if offsets:
        (out_dir / "joint_zero_offsets.json").write_text(
            json.dumps({k: float(v) for k, v in offsets.items()}, indent=2) + "\n"
        )




# --------------------------------------------------------------------------
# Step 5: mesh decimation and A2-style primitive collisions
# --------------------------------------------------------------------------

MAX_FACES = 150_000  # MuJoCo hard limit is 200k; leave headroom


def decimate_meshes(mesh_dir: Path):
    """Vertex-cluster decimation for any mesh above MuJoCo's face limit."""
    import trimesh

    report = []
    for stl in sorted(mesh_dir.glob("*.stl")):
        n = max(0, (stl.stat().st_size - 84) // 50)
        if n <= MAX_FACES:
            continue
        mesh = trimesh.load(str(stl), process=False)
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
        lo = V.min(0)
        ext = np.maximum(V.max(0) - lo, 1e-9)
        def cluster(grid):
            cell = ext / grid
            idx = np.floor((V - lo) / cell).astype(np.int64)
            key = (idx[:, 0] * (grid + 2) + idx[:, 1]) * (grid + 2) + idx[:, 2]
            uniq, inv = np.unique(key, return_inverse=True)
            rep = np.zeros((len(uniq), 3))
            cnt = np.zeros(len(uniq))
            np.add.at(rep, inv, V)
            np.add.at(cnt, inv, 1)
            rep /= cnt[:, None]
            Fn = inv[F]
            keep = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
            Fn = Fn[keep]
            _, ui = np.unique(np.sort(Fn, axis=1), axis=0, return_index=True)
            return rep, Fn[np.sort(ui)]

        # Finest grid whose decimated face count still fits under MAX_FACES.
        lo_g, hi_g, best = 4, 4096, None
        while lo_g < hi_g:
            mid = (lo_g + hi_g + 1) // 2
            cand = cluster(mid)
            if len(cand[1]) <= MAX_FACES:
                best = cand
                lo_g = mid
            else:
                hi_g = mid - 1
        if best is None:
            best = cluster(4)
        rep, Fn = best
        trimesh.Trimesh(vertices=rep, faces=Fn, process=False).export(str(stl))
        report.append((stl.name, n, len(Fn)))
    return report


def _mesh_bounds(mesh_dir: Path, g: Geom):
    import trimesh

    m = trimesh.load(str(mesh_dir / g.params["file"]), process=False)
    V = np.asarray(m.vertices, float) * np.asarray(g.params["scale"], float)
    lo, hi = V.min(0), V.max(0)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    return (g.R @ corners.T).T + g.p


def _mesh_vertices(mesh_dir: Path, g: Geom):
    import trimesh

    m = trimesh.load(str(mesh_dir / g.params["file"]), process=False)
    V = np.asarray(m.vertices, float) * np.asarray(g.params["scale"], float)
    return (g.R @ V.T).T + g.p


def _union_aabb(mesh_dir: Path, geoms, pct=0.0):
    """AABB of the union of the real vertices; `pct` trims outliers.

    A2 deliberately uses a compact collision box (0.24 m) for a much larger
    visual body, so we trim the extremities (mast, lidar, arm) as well.
    """
    pts = np.vstack([_mesh_vertices(mesh_dir, g) for g in geoms])
    if pct <= 0.0:
        return pts.min(0), pts.max(0)
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100.0 - pct, axis=0)
    return lo, hi


def simplify_collisions(links, joints, mesh_dir: Path):
    """Replace the exported per-part mesh collisions with A2-style primitives.

    The SolidWorks export turned every visual part into a collision part, giving
    190 collision geoms (158 of them on base_link alone).  The A2 reference uses
    about ten simple primitives, so we mirror that: one box for the body and each
    hip housing, and one cylinder per thigh/calf segment.
    """
    report = []
    for name, lnk in links.items():
        coll = lnk.geom_nodes("collision")
        if not coll or all(g.kind == "sphere" for g in coll):
            continue  # the new foot links keep their single contact sphere

        if name == "base_link" or name.endswith("_hip"):
            lo, hi = _union_aabb(mesh_dir, coll, pct=3.0 if name == "base_link" else 0.0)
            if name == "base_link":
                # A2 uses a compact 0.24 m collision box for a much larger visual
                # body; keep the proxy small so the chassis does not snag on steps.
                c, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * 0.6
                lo, hi = c - half, c + half
            lnk.geoms = [g for g in lnk.geoms if g.tag != "collision"]
            lnk.geoms.append(
                Geom(
                    "box",
                    (lo + hi) / 2.0,
                    np.eye(3),
                    {"size": list(hi - lo)},
                    "collision",
                )
            )
            report.append((name, len(coll), f"box size={np.round(hi - lo, 4)}"))
            continue

        leg, fam = name.split("_", 1)
        if fam not in ("thigh", "calf"):
            continue

        # Segment start/end in this link's own frame.
        T = fk(links, joints, {})
        if fam == "thigh":
            world = T[f"{leg}_calf"][:3, 3] - T[name][:3, 3]
            end = T[name][:3, :3].T @ world
            # The front thigh joints sit only ~59 mm from the centre line, i.e.
            # inside the body box, so a full-length thigh cylinder would always
            # overlap it.  Start the proxy ~30% down the segment instead.
            start = end * 0.30
        else:
            # Stop well short of the foot contact sphere.  At the crouched
            # default stance the shin axis is tilted ~50-60 deg from vertical, so
            # a cylinder reaching the sphere would have its distal rim dip below
            # the sphere and become the ground contact instead of the foot.
            end = FOOT_CACHE[leg] * 0.72
            start = np.zeros(3)

        seg = np.asarray(end) - np.asarray(start)
        length = float(np.linalg.norm(seg))
        direction = seg / length

        pts = np.vstack([_mesh_vertices(mesh_dir, g) for g in coll])
        rel = pts - np.asarray(start)
        axial = rel @ direction
        perp = rel - np.outer(axial, direction)
        radius = min(float(np.percentile(np.linalg.norm(perp, axis=1), 97.0)), 0.05)

        z = np.array([0.0, 0.0, 1.0])
        R = Rot.align_vectors(direction[None, :], z[None, :])[0].as_matrix()
        lnk.geoms = [g for g in lnk.geoms if g.tag != "collision"]
        lnk.geoms.append(
            Geom(
                "cylinder",
                np.asarray(start) + seg / 2.0,
                R,
                {"radius": radius, "length": max(length, 1e-4)},
                "collision",
            )
        )
        report.append((name, len(coll), f"cyl r={radius:.4f} l={length:.4f}"))
    return report


# --------------------------------------------------------------------------
# Step 6: emit the MJCF used for training
# --------------------------------------------------------------------------


def serialize_mjcf(links, joints) -> str:
    """Emit an MJCF mirroring the official a2.xml structure.

    MuJoCo's *URDF* importer re-parents the children of a jointless root link to
    the world, losing the root's mass, so the URDF alone is not usable for
    simulation.  The reference package handles this the same way: a2.urdf has no
    floating joint and a2.xml carries ``floating_base_joint``.  We mirror that.
    """
    children = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)

    mesh_assets = {}
    for lnk in links.values():
        for g in lnk.geoms:
            if g.kind == "mesh":
                mesh_assets.setdefault(g.params["file"], g.params["scale"])

    out = [
        f'<mujoco model="{ROBOT_NAME}">',
        '  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>',
        "",
        "  <default>",
        f'    <default class="{ROBOT_NAME}">',
        '      <default class="visual">',
        '        <geom type="mesh" contype="0" conaffinity="0" density="0" group="2"/>',
        "      </default>",
        '      <default class="collision">',
        '        <geom priority="1" condim="6" group="3"/>',
        "      </default>",
        '      <site rgba="1 0 0 1" group="5"/>',
        "    </default>",
        "  </default>",
        "",
        "  <asset>",
    ]
    for fname in sorted(mesh_assets):
        nm = Path(fname).stem
        out.append(
            f'    <mesh name="{nm}" file="{fname}" '
            f'scale="{fmt_vec(mesh_assets[fname])}"/>'
        )
    out += ["  </asset>", "", "  <worldbody>"]

    def geom_line(g: Geom, name: str | None, cls: str) -> str:
        attrs = []
        if name:
            attrs.append(f'name="{name}"')
        attrs.append(f'class="{cls}"')
        if g.kind == "mesh":
            attrs.append(f'mesh="{Path(g.params["file"]).stem}"')
        elif g.kind == "box":
            attrs.append(f'type="box" size="{fmt_vec(np.asarray(g.params["size"]) / 2.0)}"')
        elif g.kind == "sphere":
            attrs.append(f'type="sphere" size="{fmt(g.params["radius"])}"')
        elif g.kind == "cylinder":
            attrs.append(
                f'type="cylinder" size="{fmt(g.params["radius"])} '
                f'{fmt(g.params["length"] / 2.0)}"'
            )
        q = Rot.from_matrix(g.R).as_quat()  # x y z w
        attrs.append(f'pos="{fmt_vec(g.p)}" quat="{fmt_vec([q[3], q[0], q[1], q[2]])}"')
        return "        <geom " + " ".join(attrs) + "/>"

    def emit_body(name: str, indent: int, pos, quat, extra_joint: str | None):
        pad = "  " * indent
        lnk = links[name]
        head = f'{pad}<body name="{name}" pos="{fmt_vec(pos)}" quat="{fmt_vec(quat)}">'
        lines = [head]
        if extra_joint:
            lines.append(f"{pad}  {extra_joint}")
        if lnk.inertial is not None:
            p, R, mass, I = lnk.inertial
            q = Rot.from_matrix(R).as_quat()
            lines.append(
                f'{pad}  <inertial pos="{fmt_vec(p)}" '
                f'quat="{fmt_vec([q[3], q[0], q[1], q[2]])}" mass="{fmt(mass)}" '
                f'diaginertia="{fmt_vec(np.linalg.eigvalsh(I))}"/>'
            )
        for g in lnk.geoms:
            if g.kind == "mesh":
                lines.append(geom_line(g, None, "visual"))
            else:
                lines.append(geom_line(g, f"{name}_collision", "collision"))
        if name == "base_link":
            lines.append(f'{pad}  <site name="imu" pos="0 0 0"/>')
        if name.endswith("_foot"):
            leg = name.split("_")[0]
            lines.append(
                f'{pad}  <site name="{leg}" pos="0 0 0" type="sphere" size="{fmt(FOOT_RADIUS)}"/>'
            )
        for j in children.get(name, []):
            q = Rot.from_matrix(j.R).as_quat()
            if j.jtype == "revolute":
                lo, hi, _eff, _vel = j.limits
                js = (
                    f'<joint name="{j.name}" pos="0 0 0" axis="{fmt_vec(j.axis)}" '
                    f'range="{fmt(lo)} {fmt(hi)}"/>'
                )
            else:
                js = None
            lines += emit_body(
                j.child, indent + 1, j.p, [q[3], q[0], q[1], q[2]], js
            )
        lines.append(f"{pad}</body>")
        return lines

    out += emit_body(
        "base_link",
        2,
        np.zeros(3),
        [1.0, 0.0, 0.0, 0.0],
        '<joint name="floating_base_joint" type="free" limited="false" '
        'actuatorfrclimited="false"/>',
    )
    out += [
        "  </worldbody>",
        "  <sensor>",
        '    <gyro name="imu_ang_vel" site="imu"/>',
        '    <velocimeter name="imu_lin_vel" site="imu"/>',
        '    <accelerometer name="imu_lin_acc" site="imu"/>',
        '    <subtreeangmom name="root_angmom" body="base_link"/>',
        "  </sensor>",
        "</mujoco>",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
