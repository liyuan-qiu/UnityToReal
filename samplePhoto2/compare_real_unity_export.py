"""Compare unitysamplephoto2/unity_camera_quat_export.csv vs camera_pose_unity_real_photos.csv."""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALC = HERE / "camera_pose_unity_real_photos.csv"
EXP = HERE.parent / "unitysamplephoto2" / "unity_camera_quat_export.csv"


def q4(r, prefix):
    return (
        float(r[f"{prefix}_x"]),
        float(r[f"{prefix}_y"]),
        float(r[f"{prefix}_z"]),
        float(r[f"{prefix}_w"]),
    )


def qdot(a, b):
    return abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])


def ang(d):
    d = min(1.0, max(0.0, d))
    return math.degrees(2.0 * math.acos(d))


def eul_err(a, b):
    out = []
    for x, y in zip(a, b):
        d = abs(x - y) % 360.0
        out.append(min(d, 360.0 - d))
    return out


def main() -> None:
    calc = list(csv.DictReader(CALC.open(encoding="utf-8-sig")))
    exp = list(csv.DictReader(EXP.open(encoding="utf-8-sig")))
    n = min(len(calc), len(exp))
    print(f"calc: {CALC.name}  ({len(calc)} rows)")
    print(f"exp : {EXP}  ({len(exp)} rows)")
    print(f"applied_via={exp[0].get('applied_via')}  export quat_dot_abs[0]={exp[0].get('quat_dot_abs')}")
    print()
    print(
        f"{'#':>2} {'photo':>10}  {'|q_calc.q_live|':>14}  {'ang':>7}  "
        f"{'|eul dx|':>8} {'|dy|':>6} {'|dz|':>6}  "
        f"{'|csv.q_live|':>12}  {'pos_err_mm':>10}"
    )

    qd, ae, pe = [], [], []
    for i in range(n):
        c, e = calc[i], exp[i]
        qc = q4(c, "unity_quat")
        ql = (
            float(e["unity_quat_x"]),
            float(e["unity_quat_y"]),
            float(e["unity_quat_z"]),
            float(e["unity_quat_w"]),
        )
        qcsv = (
            float(e["csv_quat_x"]),
            float(e["csv_quat_y"]),
            float(e["csv_quat_z"]),
            float(e["csv_quat_w"]),
        )
        d = qdot(qc, ql)
        qd.append(d)
        ec = (float(c["unity_rot_x"]), float(c["unity_rot_y"]), float(c["unity_rot_z"]))
        ee = (float(e["euler_x"]), float(e["euler_y"]), float(e["euler_z"]))
        de = eul_err(ec, ee)
        ae.append(de)
        pc = (
            float(c["unity_pos_x"]),
            float(c["unity_pos_y"]),
            float(c["unity_pos_z"]),
        )
        pe_i = math.sqrt(
            (pc[0] - float(e["pos_x"])) ** 2
            + (pc[1] - float(e["pos_y"])) ** 2
            + (pc[2] - float(e["pos_z"])) ** 2
        ) * 1000.0
        pe.append(pe_i)
        name = Path(c.get("image_file", c.get("photo", "?"))).stem
        print(
            f"{i+1:2d} {name:>10}  {d:14.6f}  {ang(d):7.4f}  "
            f"{de[0]:8.3f} {de[1]:6.3f} {de[2]:6.3f}  "
            f"{qdot(qcsv, ql):12.6f}  {pe_i:10.4f}"
        )

    print()
    print(f"mean |q_calc · q_live| = {sum(qd)/len(qd):.6f}")
    print(f"mean angle            = {sum(ang(d) for d in qd)/len(qd):.4f} deg")
    print(
        "mean |euler calc-live| x/y/z = "
        f"{sum(a[0] for a in ae)/len(ae):.4f} / "
        f"{sum(a[1] for a in ae)/len(ae):.4f} / "
        f"{sum(a[2] for a in ae)/len(ae):.4f} deg"
    )
    print(f"mean |pos| err        = {sum(pe)/len(pe):.4f} mm")


if __name__ == "__main__":
    main()
