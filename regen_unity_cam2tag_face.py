"""
CSV angles = CAM -> TAG (Extrinsic XYZ: Rx(roll)->Ry(pitch)->Rz(yaw))
CSV xyz    = camera position in tag frame (mm)

  R_cam2tag = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  forward_real = R_cam2tag @ [0,0,1]          # camera face (no transpose)
  S = diag(-1,1,1)
  t_unity = original + S @ t_cam
  R_unity = S @ R_cam2tag @ S
  unity euler = Quaternion.Euler convention (ZXY)

Writes camera_pose_unity_cam2tag_face.csv and a Real RH / Unity LH plot.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

CSV_IN = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
CSV_OUT = Path(__file__).with_name("camera_pose_unity_cam2tag_face.csv")
OUT_PNG = Path(__file__).with_name("real_vs_unity_cam2tag_face.png")

ORIGINAL = np.array([0.03160001, -2.3284, 12.5992], dtype=float)
S = np.diag([-1.0, 1.0, 1.0])


def Rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def extrinsic_xyz_cam2tag(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """CAM->TAG Extrinsic XYZ: Rx(roll)->Ry(pitch)->Rz(yaw)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def unity_euler_zxy(ex: float, ey: float, ez: float) -> np.ndarray:
    return Ry(ey) @ Rx(ex) @ Rz(ez)


def wrap360(deg: float) -> float:
    d = deg % 360.0
    return d + 360.0 if d < 0 else d


def mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def quat_to_unity_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = q
    singularity = 0.499999
    sqx, sqy, sqz, sqw = x * x, y * y, z * z, w * w
    unit = sqx + sqy + sqz + sqw
    test = x * w - y * z
    if test > singularity * unit:
        pitch, yaw, roll = math.pi / 2, 2 * math.atan2(y, x), 0.0
    elif test < -singularity * unit:
        pitch, yaw, roll = -math.pi / 2, -2 * math.atan2(y, x), 0.0
    else:
        pitch = math.asin(2 * test / unit)
        yaw = math.atan2(2 * (w * y + z * x), sqw - sqx - sqy + sqz)
        roll = math.atan2(2 * (w * z + y * x), sqw - sqx + sqy - sqz)
    return (wrap360(math.degrees(pitch)), wrap360(math.degrees(yaw)), wrap360(math.degrees(roll)))


def to_lh_display(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=float).copy()
    q[..., 0] = -q[..., 0]
    return q


def set_equal(ax, pts, pad=0.01):
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    r = float(np.max(np.linalg.norm(pts - c, axis=1))) + pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    return r


def main() -> None:
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    out = []
    cams_r, fwds_r, c2_r = [], [], []
    cams_u, tags_u, fwds_u, c2_u = [], [], [], []
    labels, dots_r, dots_u, match = [], [], [], []

    print("CAM->TAG Extrinsic XYZ; face = R@[0,0,1]; Unity = S@R@S, t = original+S@t")
    print(f"original = {ORIGINAL.tolist()}")
    print()

    for i, row in enumerate(rows, start=1):
        name = Path(row["image_file"]).name
        roll = float(row["camera_roll_deg"])
        pitch = float(row["camera_pitch_deg"])
        yaw = float(row["camera_yaw_deg"])
        t = np.array(
            [float(row["camera_x_mm"]), float(row["camera_y_mm"]), float(row["camera_z_mm"])],
            dtype=float,
        ) / 1000.0

        R = extrinsic_xyz_cam2tag(roll, pitch, yaw)
        fwd_r = R @ np.array([0.0, 0.0, 1.0])
        fwd_r = fwd_r / (np.linalg.norm(fwd_r) + 1e-12)
        c2 = -t
        c2n = c2 / (np.linalg.norm(c2) + 1e-12)
        dr = float(np.dot(fwd_r, c2n))

        t_u = ORIGINAL + (S @ t)
        R_u = S @ R @ S
        q = mat_to_quat(R_u)  # Unity (x,y,z,w)
        # Prefer w >= 0 so q and -q don't flip randomly vs Unity
        if q[3] < 0:
            q = (-q[0], -q[1], -q[2], -q[3])
        eu = quat_to_unity_euler(q)
        fwd_u = R_u[:, 2]
        fwd_u = fwd_u / (np.linalg.norm(fwd_u) + 1e-12)
        c2_u_v = ORIGINAL - t_u
        c2_u_n = c2_u_v / (np.linalg.norm(c2_u_v) + 1e-12)
        du = float(np.dot(fwd_u, c2_u_n))
        err = float(np.linalg.norm(S @ fwd_r - fwd_u))

        out.append(
            {
                "image_file": name,
                "tag_id": row["tag_id"],
                "csv_x_mm": float(row["camera_x_mm"]),
                "csv_y_mm": float(row["camera_y_mm"]),
                "csv_z_mm": float(row["camera_z_mm"]),
                "csv_roll_deg": roll,
                "csv_pitch_deg": pitch,
                "csv_yaw_deg": yaw,
                "original_x": float(ORIGINAL[0]),
                "original_y": float(ORIGINAL[1]),
                "original_z": float(ORIGINAL[2]),
                "unity_pos_x": f"{t_u[0]:.9f}",
                "unity_pos_y": f"{t_u[1]:.9f}",
                "unity_pos_z": f"{t_u[2]:.9f}",
                "unity_cam2tag_x": f"{c2_u_v[0]:.9f}",
                "unity_cam2tag_y": f"{c2_u_v[1]:.9f}",
                "unity_cam2tag_z": f"{c2_u_v[2]:.9f}",
                "unity_rot_x": f"{eu[0]:.6f}",
                "unity_rot_y": f"{eu[1]:.6f}",
                "unity_rot_z": f"{eu[2]:.6f}",
                "unity_quat_x": f"{q[0]:.9f}",
                "unity_quat_y": f"{q[1]:.9f}",
                "unity_quat_z": f"{q[2]:.9f}",
                "unity_quat_w": f"{q[3]:.9f}",
                "look_dot_real": f"{dr:.4f}",
                "look_dot_unity": f"{du:.4f}",
                "rot_mode": "CAM->TAG ExtXYZ Rx(roll)->Ry(pitch)->Rz(yaw); face=R@z; R_u=S@R@S; quat from R_u",
            }
        )
        cams_r.append(t)
        fwds_r.append(fwd_r)
        c2_r.append(c2)
        cams_u.append(t_u)
        tags_u.append(ORIGINAL.copy())
        fwds_u.append(fwd_u)
        c2_u.append(c2_u_v)
        labels.append(str(i))
        dots_r.append(dr)
        dots_u.append(du)
        match.append(err)
        print(
            f"{i:2d} {name:20s}  real_dot={dr:+.3f}  unity_dot={du:+.3f}  "
            f"|S@fwd-U|={err:.1e}  euler=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f})  "
            f"quat=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {CSV_OUT}")
    print(f"mean look_dot real={np.mean(dots_r):+.3f}  unity={np.mean(dots_u):+.3f}")
    print(f"mean |S@fwd_real - fwd_unity|={np.mean(match):.2e}")

    cams_r = np.asarray(cams_r)
    fwds_r = np.asarray(fwds_r)
    c2_r = np.asarray(c2_r)
    cams_u = np.asarray(cams_u)
    tags_u = np.asarray(tags_u)
    fwds_u = np.asarray(fwds_u)
    c2_u = np.asarray(c2_u)
    fwd_len = 0.035

    fig = plt.figure(figsize=(16.5, 8.0))
    ax_r = fig.add_subplot(121, projection="3d")
    ax_u = fig.add_subplot(122, projection="3d")

    # ---- REAL RH ----
    ax_r.scatter(0, 0, 0, c="k", s=100, depthshade=False, zorder=6)
    ax_r.text(0.002, 0.002, 0.002, " TAG", fontsize=10, fontweight="bold")
    for i in range(len(rows)):
        ax_r.scatter(*cams_r[i], c="#ff7f0e", s=50, depthshade=False)
        ax_r.plot(
            [cams_r[i, 0], 0], [cams_r[i, 1], 0], [cams_r[i, 2], 0],
            color="0.5", ls="--", lw=1.0, alpha=0.7,
        )
        tip = cams_r[i] + fwd_len * fwds_r[i]
        ax_r.quiver(
            cams_r[i, 0], cams_r[i, 1], cams_r[i, 2],
            tip[0] - cams_r[i, 0], tip[1] - cams_r[i, 1], tip[2] - cams_r[i, 2],
            color="#d62728", arrow_length_ratio=0.18, lw=1.8,
        )
        ax_r.text(*cams_r[i], f" {labels[i]}", fontsize=8)
    for v, col, name in [
        (np.array([0.04, 0, 0]), "#d62728", "+X"),
        (np.array([0, 0.04, 0]), "#2ca02c", "+Y"),
        (np.array([0, 0, 0.04]), "#1f77b4", "+Z"),
    ]:
        ax_r.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2.4)
        ax_r.text(*v, f" {name}", color=col, fontsize=9, fontweight="bold")
    ax_r.plot([], [], color="0.5", ls="--", label="geometric cam2tag")
    ax_r.plot([], [], color="#d62728", lw=2, label="face = R@[0,0,1] (CAM→TAG)")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.set_title(
        "REAL right-handed (tag@0)\n"
        "CAM→TAG ExtXYZ: R=Rz(yaw)@Ry(pitch)@Rx(roll)\n"
        f"red = camera face   mean face·cam2tag={np.mean(dots_r):+.3f}",
        fontsize=10,
    )
    ax_r.set_xlabel("X"); ax_r.set_ylabel("Y"); ax_r.set_zlabel("Z")
    set_equal(ax_r, np.vstack([np.zeros(3), cams_r]), pad=0.012)
    ax_r.view_init(elev=22, azim=-60)
    ax_r.text2D(
        0.02, 0.02, "RIGHT-HANDED",
        transform=ax_r.transAxes, fontsize=10, fontweight="bold", color="#2a9d8f",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    # ---- UNITY LH display ----
    tag0 = tags_u[0]
    tag_d = to_lh_display(tag0)
    cams_d = np.array([to_lh_display(p) for p in cams_u])
    c2_d = np.array([to_lh_display(v) for v in c2_u])
    fwds_d = np.array([to_lh_display(v) for v in fwds_u])
    ax_u.scatter(*tag_d, c="k", s=100, depthshade=False, zorder=6)
    ax_u.text(tag_d[0] + 0.002, tag_d[1] + 0.002, tag_d[2] + 0.002, " TAG", fontsize=10, fontweight="bold")
    for i in range(len(rows)):
        ax_u.scatter(*cams_d[i], c="#17becf", s=50, depthshade=False)
        ax_u.plot(
            [cams_d[i, 0], cams_d[i, 0] + c2_d[i, 0]],
            [cams_d[i, 1], cams_d[i, 1] + c2_d[i, 1]],
            [cams_d[i, 2], cams_d[i, 2] + c2_d[i, 2]],
            color="0.5", ls="--", lw=1.0, alpha=0.7,
        )
        tip = cams_d[i] + fwd_len * fwds_d[i]
        ax_u.quiver(
            cams_d[i, 0], cams_d[i, 1], cams_d[i, 2],
            tip[0] - cams_d[i, 0], tip[1] - cams_d[i, 1], tip[2] - cams_d[i, 2],
            color="#1f77b4", arrow_length_ratio=0.18, lw=1.8,
        )
        ax_u.text(*cams_d[i], f" {labels[i]}", fontsize=8)
    g = 0.04
    for v, col, name in [
        (np.array([g, 0, 0]), "#d62728", "+X"),
        (np.array([0, g, 0]), "#2ca02c", "+Y"),
        (np.array([0, 0, g]), "#1f77b4", "+Z"),
    ]:
        tip = tag_d + to_lh_display(v)
        ax_u.plot([tag_d[0], tip[0]], [tag_d[1], tip[1]], [tag_d[2], tip[2]], color=col, lw=2.4)
        ax_u.text(*tip, f" {name}", color=col, fontsize=9, fontweight="bold")
    ax_u.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{-v:.2f}"))
    ax_u.plot([], [], color="0.5", ls="--", label="geometric cam2tag")
    ax_u.plot([], [], color="#1f77b4", lw=2, label="face from Unity euler / R_u")
    ax_u.legend(loc="upper left", fontsize=8)
    ax_u.set_title(
        "UNITY left-handed view\n"
        "t_u=original+S@t   R_u=S@R@S   euler=ZXY\n"
        f"blue = camera face   mean face·cam2tag={np.mean(dots_u):+.3f}\n"
        f"|S@fwd_real − fwd_unity| mean={np.mean(match):.2e}",
        fontsize=10,
    )
    ax_u.set_xlabel("Unity X (LH)")
    ax_u.set_ylabel("Unity Y")
    ax_u.set_zlabel("Unity Z")
    set_equal(ax_u, np.vstack([tag_d.reshape(1, 3), cams_d]), pad=0.012)
    ax_u.view_init(elev=22, azim=-60)
    ax_u.text2D(
        0.02, 0.02, "LEFT-HANDED (display)",
        transform=ax_u.transAxes, fontsize=10, fontweight="bold", color="#c44e52",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    fig.suptitle(
        "CAM→TAG face (R@[0,0,1])  |  Real RH vs Unity LH   "
        "CSV: camera_pose_unity_cam2tag_face.csv",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.05)
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
