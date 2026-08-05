"""Check whether baseline/tag point falls inside each camera frustum."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import convert_pose_to_unity as c

CSV = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
BASELINE = (0.03160001, -2.2834, 12.5992)
HFOV = 70.5934
ASPECT = 1080 / 720
LENS_SHIFT_X = 0.112535
LENS_SHIFT_Y = 0.000522


def q_rotate(q, v):
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


def build_q(pitch, roll, yaw, mode: str):
    pq = c.q_angle_axis(pitch, (1, 0, 0))
    rq = c.q_angle_axis(roll, (0, 0, 1))
    yq = c.q_angle_axis(yaw, (0, 1, 0))
    ext = c.q_mul(yq, c.q_mul(rq, pq))
    m2 = c.q_axis_flip_x_neg_y_z(ext)
    x180 = c.q_angle_axis(180, (1, 0, 0))
    y180 = c.q_angle_axis(180, (0, 1, 0))
    if mode == "current_x180_m2":
        return c.q_mul(x180, m2)
    if mode == "m2_x180":
        return c.q_mul(m2, x180)
    if mode == "m2_only":
        return m2
    if mode == "y180_m2":
        return c.q_mul(y180, m2)
    raise ValueError(mode)


def world_to_cam(q, cam_pos, world_pt):
    # p_cam = R^T * (p_world - cam_pos)  for Unity (R maps cam->world)
    # Unity rotation quaternion rotates local->world, so world_to_local = R^{-1}
    dx = world_pt[0] - cam_pos[0]
    dy = world_pt[1] - cam_pos[1]
    dz = world_pt[2] - cam_pos[2]
    qi = (-q[0], -q[1], -q[2], q[3])
    return q_rotate(qi, (dx, dy, dz))


def in_fov(pc, hfov_deg, aspect, shift_x, shift_y):
    x, y, z = pc
    if z <= 1e-6:
        return False, "behind/near", None
    # Unity perspective with horizontal FOV + lens shift (approx):
    # NDC-ish: x' = (x/z)/tan(hfov/2) - 2*shift_x ? 
    # Unity lens shift moves the projection center.
    # From Unity docs / common: ray offset so principal point shifts.
    # Approximate: compare x/z to frustum edges shifted by lensShift * tan(fov/2)*2 / something
    # Simpler OpenCV-like: 
    # u = fx*X/Z + cx  with fx = W/(2tan(hfov/2)), cx = W/2 + shift_x*W
    half_w = math.tan(math.radians(hfov_deg) / 2)
    vfov = 2 * math.degrees(math.atan(half_w / aspect))
    half_h = math.tan(math.radians(vfov) / 2)
    # normalized image plane coords with principal point offset
    # xn = X/Z, yn = Y/Z; visible if |xn - sx| < half_w and |yn - sy| < half_h
    # where sx = 2*lensShiftX * half_w ? Unity: lensShift is fraction of sensor.
    # Sensor NDC: x_ndc from -1..1 maps to sensor. Lens shift X adds offset in sensor units.
    # Unity: "The lens shift offsets the projection... X of 0.5 shifts by half sensor width"
    sx = shift_x * 2 * half_w  # shift in same units as X/Z
    sy = shift_y * 2 * half_h
    xn, yn = x / z, y / z
    ok_x = abs(xn - sx) <= half_w * 1.001
    ok_y = abs(yn - sy) <= half_h * 1.001
    info = {
        "z": z,
        "xn": xn,
        "yn": yn,
        "sx": sx,
        "sy": sy,
        "half_w": half_w,
        "half_h": half_h,
        "ok_x": ok_x,
        "ok_y": ok_y,
    }
    return ok_x and ok_y, "ok" if (ok_x and ok_y) else "out", info


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    modes = ["current_x180_m2", "m2_x180", "m2_only", "y180_m2"]
    for mode in modes:
        print(f"\n======== {mode} ========")
        for row in rows[:5]:
            name = Path(row["image_file"]).name
            pitch = float(row["camera_pitch_deg"])
            roll = float(row["camera_roll_deg"])
            yaw = float(row["camera_yaw_deg"])
            xmm = float(row["camera_x_mm"])
            ymm = float(row["camera_y_mm"])
            zmm = float(row["camera_z_mm"])
            cam = (
                BASELINE[0] - xmm / 1000,
                BASELINE[1] + ymm / 1000,
                BASELINE[2] + zmm / 1000,
            )
            q = build_q(pitch, roll, yaw, mode)
            e = c.q_to_unity_euler(q)
            pc = world_to_cam(q, cam, BASELINE)
            ok, reason, info = in_fov(pc, HFOV, ASPECT, LENS_SHIFT_X, LENS_SHIFT_Y)
            fwd = q_rotate(q, (0, 0, 1))
            print(
                f"{name:20s} rot=({e[0]:6.2f},{e[1]:7.2f},{e[2]:6.2f}) "
                f"tag_in_cam=({pc[0]:+.4f},{pc[1]:+.4f},{pc[2]:+.4f}) "
                f"fwd=({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f}) "
                f"FOV={'IN' if ok else 'OUT':3s} {reason}"
            )
            if info and not ok:
                print(
                    f"  xn={info['xn']:+.3f} (center_shift={info['sx']:+.3f}, half={info['half_w']:.3f}) "
                    f"yn={info['yn']:+.3f} (shift={info['sy']:+.3f}, half={info['half_h']:.3f}) "
                    f"okx={info['ok_x']} oky={info['ok_y']}"
                )


if __name__ == "__main__":
    main()
