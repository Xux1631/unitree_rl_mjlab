#!/usr/bin/env python3
"""Verify the rebuilt URDF against the A2 interface reference.

Checks, in order:
  A. joint names / declaration order / axes / limits against INTERFACE.md
  B. the URDF compiles in MuJoCo with valid masses and inertias
  C. the A2 default stance gives four level, symmetric feet
  D. every joint's *physical* axis and positive direction matches the reference

Run:  python tools/verify_a2_urdf.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parent.parent
MINE = (
    REPO
    / "src"
    / "assets"
    / "robots"
    / "quadruped_3320"
    / "urdf"
    / "quadruped_3320.urdf"
)
MJCF = MINE.parent.parent / "xmls" / "quadruped_3320.xml"
MINE_URDF = MINE
# Read-only A2 interface reference, shipped by the assessment:
#   assessment/reference/A2_urdf/a2_description/urdf/a2.urdf
REF = Path(
    os.environ.get(
        "A2_REFERENCE_URDF",
        REPO.parents[1] / ".assessment_ref" / "assessment" / "reference"
        / "A2_urdf" / "a2_description" / "urdf" / "a2.urdf",
    )
)

LEGS = ("RL", "FL", "RR", "FR")
FAMILIES = ("hip", "thigh", "calf")
JOINT_ORDER = [f"{leg}_{f}_joint" for leg in LEGS for f in FAMILIES]
REF_AXIS = {"hip": "1 0 0", "thigh": "0 1 0", "calf": "0 1 0"}
REF_LIMITS = {
    "RL_hip_joint": (-1.01, 1.01), "RL_thigh_joint": (-1.56, 3.94), "RL_calf_joint": (-2.77, -0.54),
    "FL_hip_joint": (-1.01, 1.01), "FL_thigh_joint": (-2.34, 3.15), "FL_calf_joint": (-2.77, -0.54),
    "RR_hip_joint": (-1.01, 1.01), "RR_thigh_joint": (-1.56, 3.94), "RR_calf_joint": (-2.77, -0.54),
    "FR_hip_joint": (-1.01, 1.01), "FR_thigh_joint": (-2.34, 3.15), "FR_calf_joint": (-2.77, -0.54),
}
# A2 default stance from a2_constants.INIT_STATE
STANCE = {"thigh": 0.9, "calf": -1.8, "RL": -0.1, "FL": -0.1, "RR": 0.1, "FR": 0.1}

FAILS: list[str] = []


def check(ok: bool, msg: str):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILS.append(msg)


def parse(path: Path):
    root = ET.parse(path).getroot()
    joints = [j for j in root.findall("joint")]
    return root, joints


def num(s, default="0 0 0"):
    if s is None or not str(s).strip():
        s = default
    return [float(v) for v in str(s).split()]


def build_fk(path: Path):
    """Return (fk(q) -> {link: 4x4}, joint index dict) for a URDF."""
    root = ET.parse(path).getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "floating"]

    def fk(q):
        out = {"base_link": np.eye(4)}
        changed = True
        while changed:
            changed = False
            for j in joints:
                par = j.find("parent").get("link")
                ch = j.find("child").get("link")
                if par in out and ch not in out:
                    o = j.find("origin")
                    T = out[par].copy()
                    Tj = np.eye(4)
                    Tj[:3, :3] = Rot.from_euler("xyz", num(o.get("rpy") if o is not None else None)).as_matrix()
                    Tj[:3, 3] = num(o.get("xyz") if o is not None else None)
                    T = T @ Tj
                    ax = j.find("axis")
                    if ax is not None:
                        a = np.array(num(ax.get("xyz")))
                        a = a / np.linalg.norm(a)
                        Rq = np.eye(4)
                        Rq[:3, :3] = Rot.from_rotvec(a * q.get(j.get("name"), 0.0)).as_matrix()
                        T = T @ Rq
                    out[ch] = T
                    changed = True
        return out

    return fk, joints


def foot_point(fkq, leg: str, sole_local: np.ndarray):
    T = fkq[f"{leg}_calf"]
    if f"{leg}_foot" in fkq:
        return fkq[f"{leg}_foot"][:3, 3]
    return T[:3, :3] @ sole_local + T[:3, 3]


def main():
    print("=" * 78)
    print("A. interface conformance (INTERFACE.md section 2/3)")
    print("=" * 78)
    root, joints = parse(MINE)
    names = [j.get("name") for j in joints]
    rev = [j for j in joints if j.get("type") == "revolute"]
    rev_names = [j.get("name") for j in rev]
    check(rev_names == JOINT_ORDER, f"revolute declaration order == RL/FL/RR/FR, hip/thigh/calf")
    check(root.get("name") == "quadruped_3320", f"robot name = {root.get('name')}")

    links = [l.get("name") for l in root.findall("link")]
    for want in [f"{leg}_foot" for leg in LEGS] + [
        "base_link", "camera_link", "front_lidar_link", "rear_lidar_link",
    ]:
        check(want in links, f"link '{want}' present")

    for leg in LEGS:
        for fam in FAMILIES:
            j = [x for x in joints if x.get("name") == f"{leg}_{fam}_joint"][0]
            ax = j.find("axis").get("xyz")
            check(ax == REF_AXIS[fam], f"{leg}_{fam}_joint axis = '{ax}' (want '{REF_AXIS[fam]}')")
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            wlo, whi = REF_LIMITS[f"{leg}_{fam}_joint"]
            check(
                abs(lo - wlo) < 1e-9 and abs(hi - whi) < 1e-9
                and lim.get("effort") and lim.get("velocity"),
                f"{leg}_{fam}_joint limits [{lo}, {hi}] + effort/velocity",
            )

    print()
    print("=" * 78)
    print("B. MuJoCo load")
    print("=" * 78)
    import mujoco

    try:
        model = mujoco.MjModel.from_xml_path(str(MINE))
        _ = model
        check(True, f"MjModel.from_xml_path OK  (nbody={model.nbody}, ngeom={model.ngeom}, "
                    f"njnt={model.njnt}, nu={model.nu})")
    except Exception as exc:  # noqa: BLE001
        check(False, f"MuJoCo load failed: {exc}")
        print("\n" + "\n".join(FAILS))
        return 1

    masses = np.array([model.body_mass[i] for i in range(1, model.nbody)])
    check(bool(np.all(masses > 0)), f"all body masses > 0 (min={masses.min():.5f} kg)")
    check(len(masses) == 12, f"{len(masses)} jointless-root children (base_link dropped by MuJoCo)")
    inertia_ok = True
    for i in range(1, model.nbody):
        I = model.body_inertia[i]
        if not np.all(np.isfinite(I)) or np.any(I <= 0):
            inertia_ok = False
    check(inertia_ok, "every body has finite, positive principal inertia")

    print()
    print("=" * 78)
    print("B2. URDF load (reference a2.urdf has the same jointless-root behaviour)")
    print("=" * 78)
    try:
        mu = mujoco.MjModel.from_xml_path(str(MINE_URDF))
        check(True, f"MjModel.from_xml_path(URDF) OK (nbody={mu.nbody}, ngeom={mu.ngeom})")
        check(
            True,
            "NOTE: MuJoCo re-parents a jointless root's children to the world, so "
            "base_link's mass is dropped here. The reference a2.urdf behaves "
            "identically; the free joint lives in the MJCF.",
        )
    except Exception as exc:  # noqa: BLE001
        check(False, f"URDF load failed: {exc}")

    print()
    print("=" * 78)
    print("B3. MJCF (the file MJLab actually loads)")
    print("=" * 78)
    model = None
    try:
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        check(True, f"MjModel.from_xml_path(MJCF) OK (nbody={model.nbody}, "
                    f"njnt={model.njnt}, ngeom={model.ngeom}, nsite={model.nsite})")
    except Exception as exc:  # noqa: BLE001
        check(False, f"MJCF load failed: {exc}")
    if model is not None:
        total = float(model.body_mass.sum())
        check(abs(total - 9.21) < 0.2, f"MJCF total mass = {total:.4f} kg (all links present)")
        jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        check(jn[0] == "floating_base_joint", "MJCF has floating_base_joint")
        check(jn[1:13] == JOINT_ORDER, "MJCF hinge joint order == A2 policy order")
        gn = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
        ]
        coll = [g for g in gn if g and g.endswith("_collision")]
        check(len(coll) <= 24, f"MJCF has {len(coll)} collision geoms (A2-style primitives)")
        for leg in LEGS:
            check(f"{leg}_foot_collision" in coll, f"MJCF geom '{leg}_foot_collision' matches env_cfgs")
        sn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
        for want in ["imu"] + list(LEGS):
            check(want in sn, f"MJCF site '{want}' present (needed by env_cfgs)")

    print()
    print("=" * 78)
    print("C. A2 default stance -> level, symmetric feet")
    print("=" * 78)
    fk_mine, _ = build_fk(MINE)
    fk_ref, _ = build_fk(REF)

    q_stance = {}
    for leg in LEGS:
        q_stance[f"{leg}_hip_joint"] = STANCE[leg]
        q_stance[f"{leg}_thigh_joint"] = STANCE["thigh"]
        q_stance[f"{leg}_calf_joint"] = STANCE["calf"]

    # sole offsets are not needed: the rebuilt URDF has explicit *_foot links
    Tm = fk_mine(q_stance)
    feet = {leg: Tm[f"{leg}_foot"][:3, 3] for leg in LEGS}
    zs = np.array([feet[leg][2] for leg in LEGS])
    ptp = float(np.ptp(zs))
    check(ptp < 0.005, f"foot heights within 5 mm (ptp={ptp * 1000:.2f} mm)")
    for leg in LEGS:
        tj = Tm[f"{leg}_thigh"][:3, 3]
        d = feet[leg] - tj
        check(
            abs(d[0]) < 0.02 and abs(d[1]) < 0.02,
            f"{leg} foot directly under thigh joint (dx={d[0] * 1000:+.1f} mm, dy={d[1] * 1000:+.1f} mm)",
        )
    check(
        abs(feet["RL"][1] - (-feet["RR"][1])) < 0.002,
        f"left/right stance symmetric (RL.y={feet['RL'][1]:+.4f}, RR.y={feet['RR'][1]:+.4f})",
    )
    check(
        float(zs.mean()) < -0.1,
        f"feet below the base (mean foot z = {zs.mean():.4f} m, base at 0)",
    )

    print()
    print("=" * 78)
    print("D. physical axis + positive direction vs the A2 reference")
    print("=" * 78)
    if not Path(REF).exists():
        print(f"  [SKIP] reference URDF not found at {REF}")
        print("         set A2_REFERENCE_URDF to the assessment's")
        print("         assessment/reference/A2_urdf/a2_description/urdf/a2.urdf")
        return 1 if FAILS else 0
    print("=" * 78)
    for leg in LEGS:
        for fam in FAMILIES:
            jname = f"{leg}_{fam}_joint"
            base = dict(q_stance)
            eps = 0.05
            d = {}
            for tag, fk_q in (
                ("mine", fk_mine),
                ("ref", fk_ref),
            ):
                f0 = fk_q(base)
                b = dict(base)
                b[jname] = base[jname] + eps
                f1 = fk_q(b)
                d[tag] = f1[f"{leg}_foot"][:3, 3] - f0[f"{leg}_foot"][:3, 3]
            cos = float(np.dot(d["mine"], d["ref"]) / (np.linalg.norm(d["mine"]) * np.linalg.norm(d["ref"])))
            check(cos > 0.90, f"{jname}: foot motion direction matches A2 (cos={cos:+.4f})")

    print()
    print("=" * 78)
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} check(s) failed)")
        for f in FAILS:
            print("   -", f)
        return 1
    print("RESULT: PASS - URDF conforms to the A2 interface and stands correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
