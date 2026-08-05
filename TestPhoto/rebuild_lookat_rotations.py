"""
Rebuild Unity rotations so each camera LOOKS AT the tag/baseline point.
Also keep converted euler for comparison.
Fixes the common issue: CV euler conversion may leave object outside view
even when 'forward·to_tag' is roughly correct (up-axis / order ambiguity).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import convert_pose_to_unity as c

CSV_IN = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
CSV_OUT = Path(__file__).with_name("camera_pose_unity_lookat.csv")
# also try updating main csv if not locked
CSV_OUT_MAIN = Path(__file__).with_name("camera_pose_unity.csv")

BASELINE = (0.03160001, -2.2834, 12.5992)


def q_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def look_rotation(forward, up=(0.0, 1.0, 0.0)):
    """Unity-like LookRotation: local +Z -> forward, try keep +Y near up."""
    fx, fy, fz = forward
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn

    ux, uy, uz = up
    # right = up x forward
    rx = uy * fz - uz * fy
    ry = uz * fx - ux * fz
    rz = ux * fy - uy * fx
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rn < 1e-8:
        # forward parallel up; pick another up
        ux, uy, uz = 0.0, 0.0, 1.0
        rx = uy * fz - uz * fy
        ry = uz * fx - ux * fz
        rz = ux * fy - uy * fx
        rn = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn

    # recomputed up = forward x right
    ux = fy * rz - fz * ry
    uy = fz * rx - fx * rz
    uz = fx * ry - fy * rx

    # rotation matrix columns = right, up, forward (Unity)
    # convert matrix to quaternion
    m00, m01, m02 = rx, ux, fx
    m10, m11, m12 = ry, uy, fy
    m20, m21, m22 = rz, uz, fz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return q_normalize((qx, qy, qz, qw))


def cam_pry_to_unity_q(pitch, roll, yaw):
    pitch_q = c.q_angle_axis(pitch, (1.0, 0.0, 0.0))
    roll_q = c.q_angle_axis(roll, (0.0, 0.0, 1.0))
    yaw_q = c.q_angle_axis(yaw, (0.0, 1.0, 0.0))
    ext_q = c.q_mul(yaw_q, c.q_mul(roll_q, pitch_q))
    m2_q = c.q_axis_flip_x_neg_y_z(ext_q)
    x180 = c.q_angle_axis(180.0, (1.0, 0.0, 0.0))
    return c.q_mul(x180, m2_q)


def main():
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    out_rows = []
    print("NOTE: Unity Camera near clip should be <= 0.01 (tag distance ~0.04..0.09 m)")
    print()
    for row in rows:
        pitch = float(row["camera_pitch_deg"])
        roll = float(row["camera_roll_deg"])
        yaw = float(row["camera_yaw_deg"])
        xmm = float(row["camera_x_mm"])
        ymm = float(row["camera_y_mm"])
        zmm = float(row["camera_z_mm"])

        cam = (
            BASELINE[0] - xmm / 1000.0,
            BASELINE[1] + ymm / 1000.0,
            BASELINE[2] + zmm / 1000.0,
        )
        # direction camera -> tag
        to_tag = (
            BASELINE[0] - cam[0],
            BASELINE[1] - cam[1],
            BASELINE[2] - cam[2],
        )
        dist = math.sqrt(to_tag[0] ** 2 + to_tag[1] ** 2 + to_tag[2] ** 2)

        q_cv = cam_pry_to_unity_q(pitch, roll, yaw)
        e_cv = c.q_to_unity_euler(q_cv)

        q_look = look_rotation(to_tag, up=(0.0, 1.0, 0.0))
        e_look = c.q_to_unity_euler(q_look)

        # also look with up from converted rotation (preserve measured roll)
        # up_cv = R_cv * (0,1,0)
        def qrot(q, v):
            x, y, z, w = q
            vx, vy, vz = v
            tx = 2 * (y * vz - z * vy)
            ty = 2 * (z * vx - x * vz)
            tz = 2 * (x * vy - y * vx)
            return (
                vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx),
            )

        up_cv = qrot(q_cv, (0.0, 1.0, 0.0))
        q_look_roll = look_rotation(to_tag, up=up_cv)
        e_look_roll = c.q_to_unity_euler(q_look_roll)

        name = Path(row["image_file"]).name
        # Prefer look-at with measured up (sees tag + keeps CV roll)
        rx, ry, rz = e_look_roll

        out_rows.append(
            {
                "image_file": name,
                "tag_id": row["tag_id"],
                "csv_x_mm": xmm,
                "csv_y_mm": ymm,
                "csv_z_mm": zmm,
                "csv_roll_deg": roll,
                "csv_pitch_deg": pitch,
                "csv_yaw_deg": yaw,
                "unity_pos_x": f"{cam[0]:.9f}",
                "unity_pos_y": f"{cam[1]:.9f}",
                "unity_pos_z": f"{cam[2]:.9f}",
                "unity_rot_x": f"{rx:.6f}",
                "unity_rot_y": f"{ry:.6f}",
                "unity_rot_z": f"{rz:.6f}",
                "dist_to_tag_m": f"{dist:.6f}",
                "rot_cv_x180m2": f"{e_cv[0]:.4f},{e_cv[1]:.4f},{e_cv[2]:.4f}",
                "rot_lookat_yup": f"{e_look[0]:.4f},{e_look[1]:.4f},{e_look[2]:.4f}",
                "rot_mode": "lookat_keep_cv_up",
            }
        )
        print(
            f"{name:20s} dist={dist:.4f}m  "
            f"LOOK=({rx:6.2f},{ry:7.2f},{rz:6.2f})  "
            f"CV=({e_cv[0]:6.2f},{e_cv[1]:7.2f},{e_cv[2]:6.2f})"
        )

    def write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {path}")

    write_csv(CSV_OUT)
    try:
        write_csv(CSV_OUT_MAIN)
    except PermissionError:
        print(f"(skipped locked file {CSV_OUT_MAIN.name})")
    print("Use unity_rot_* columns (LookAt tag, keep CV up-axis).")
    print("If still invisible: set Camera Near Clip = 0.001")


if __name__ == "__main__":
    main()
