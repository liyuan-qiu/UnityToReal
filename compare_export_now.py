import csv
import math
from pathlib import Path

calc = list(csv.DictReader(Path("camera_pose_unity_cam2tag_face.csv").open(encoding="utf-8-sig")))
exp = list(csv.DictReader(Path("unity_camera_quat_export.csv").open(encoding="utf-8-sig")))


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


print("applied_via:", exp[0]["applied_via"], "  export quat_dot_abs:", exp[0]["quat_dot_abs"])
print()
print(
    f"{'#':>2}  {'|q_calc.q_live|':>14}  {'ang_deg':>8}  "
    f"{'eul_live_x':>10} {'eul_live_y':>10} {'eul_live_z':>10}  "
    f"{'|dx|':>6} {'|dy|':>6} {'|dz|':>6}"
)

qd, ae = [], []
for i, (c, e) in enumerate(zip(calc, exp), 1):
    qc = q4(c, "unity_quat")
    ql = (
        float(e["unity_quat_x"]),
        float(e["unity_quat_y"]),
        float(e["unity_quat_z"]),
        float(e["unity_quat_w"]),
    )
    d = qdot(qc, ql)
    qd.append(d)
    ec = (float(c["unity_rot_x"]), float(c["unity_rot_y"]), float(c["unity_rot_z"]))
    ee = (float(e["euler_x"]), float(e["euler_y"]), float(e["euler_z"]))
    de = eul_err(ec, ee)
    ae.append(de)
    print(
        f"{i:2d}  {d:14.6f}  {ang(d):8.4f}  "
        f"{ee[0]:10.4f} {ee[1]:10.4f} {ee[2]:10.4f}  "
        f"{de[0]:6.3f} {de[1]:6.3f} {de[2]:6.3f}"
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
print()
print("Also: export.csv_quat vs calc.unity_quat (should match input):")
for i, (c, e) in enumerate(zip(calc, exp), 1):
    qc = q4(c, "unity_quat")
    qcsv = (
        float(e["csv_quat_x"]),
        float(e["csv_quat_y"]),
        float(e["csv_quat_z"]),
        float(e["csv_quat_w"]),
    )
    print(f"  {i:2d} |q·q|={qdot(qc, qcsv):.6f}")
