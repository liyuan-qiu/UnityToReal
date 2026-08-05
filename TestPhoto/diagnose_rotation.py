"""Diagnose which rotation conversion makes Unity camera look toward tag/origin."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import convert_pose_to_unity as c

CSV = Path(__file__).with_name("camera_pose_relative_to_tag.csv")


def q_rotate_vec(q, v):
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


def q_inv(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def build_q(pitch, roll, yaw, mode: str):
    pq = c.q_angle_axis(pitch, (1, 0, 0))
    rq = c.q_angle_axis(roll, (0, 0, 1))
    yq = c.q_angle_axis(yaw, (0, 1, 0))
    ext = c.q_mul(yq, c.q_mul(rq, pq))  # extrinsic P->R->Y
    m2 = c.q_axis_flip_x_neg_y_z(ext)

    if mode == "raw":
        return ext
    if mode == "m1_negyaw":
        yq2 = c.q_angle_axis(-yaw, (0, 1, 0))
        return c.q_mul(yq2, c.q_mul(rq, pq))
    if mode == "m2":
        return m2
    if mode == "m2_x180":
        return c.q_mul(m2, c.q_angle_axis(180, (1, 0, 0)))
    if mode == "x180_m2":
        return c.q_mul(c.q_angle_axis(180, (1, 0, 0)), m2)
    if mode == "m2_y180":
        return c.q_mul(m2, c.q_angle_axis(180, (0, 1, 0)))
    if mode == "y180_m2":
        return c.q_mul(c.q_angle_axis(180, (0, 1, 0)), m2)
    if mode == "inv_ext_m2":
        return c.q_axis_flip_x_neg_y_z(q_inv(ext))
    if mode == "inv_m2":
        return q_inv(m2)
    if mode == "opencv_cam":
        # common OpenCV->Unity camera: S * R * S then * 180X
        return c.q_mul(m2, c.q_angle_axis(180, (1, 0, 0)))
    raise ValueError(mode)


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    modes = [
        "raw",
        "m1_negyaw",
        "m2",
        "m2_x180",
        "x180_m2",
        "m2_y180",
        "y180_m2",
        "inv_ext_m2",
        "inv_m2",
    ]
    print("dot(camera_forward, direction_to_tag); +1 = looking at tag (using unity pos = -x,y,z mm)")
    print()
    for mode in modes:
        dots = []
        for row in rows:
            pitch = float(row["camera_pitch_deg"])
            roll = float(row["camera_roll_deg"])
            yaw = float(row["camera_yaw_deg"])
            x = float(row["camera_x_mm"])
            y = float(row["camera_y_mm"])
            z = float(row["camera_z_mm"])
            # unity-mapped position (mm space ok for direction)
            px, py, pz = -x, y, z
            n = math.sqrt(px * px + py * py + pz * pz) + 1e-12
            to_tag = (-px / n, -py / n, -pz / n)
            q = build_q(pitch, roll, yaw, mode)
            fwd = q_rotate_vec(q, (0, 0, 1))  # Unity camera forward
            dots.append(fwd[0] * to_tag[0] + fwd[1] * to_tag[1] + fwd[2] * to_tag[2])
        mean = sum(dots) / len(dots)
        print(f"{mode:12s}  mean={mean:+.3f}  min={min(dots):+.3f}  max={max(dots):+.3f}")

    print("\nPer-image detail for best candidates:")
    for mode in ["m2", "m2_x180", "x180_m2", "y180_m2", "inv_m2"]:
        print(f"\n=== {mode} ===")
        for row in rows[:5]:
            pitch = float(row["camera_pitch_deg"])
            roll = float(row["camera_roll_deg"])
            yaw = float(row["camera_yaw_deg"])
            x = float(row["camera_x_mm"])
            y = float(row["camera_y_mm"])
            z = float(row["camera_z_mm"])
            px, py, pz = -x, y, z
            n = math.sqrt(px * px + py * py + pz * pz)
            to_tag = (-px / n, -py / n, -pz / n)
            q = build_q(pitch, roll, yaw, mode)
            fwd = q_rotate_vec(q, (0, 0, 1))
            d = fwd[0] * to_tag[0] + fwd[1] * to_tag[1] + fwd[2] * to_tag[2]
            name = Path(row["image_file"]).name
            e = c.q_to_unity_euler(q)
            print(
                f"  {name:20s} fwd=({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f}) "
                f"to_tag=({to_tag[0]:+.2f},{to_tag[1]:+.2f},{to_tag[2]:+.2f}) "
                f"dot={d:+.3f} euler=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})"
            )


if __name__ == "__main__":
    main()
