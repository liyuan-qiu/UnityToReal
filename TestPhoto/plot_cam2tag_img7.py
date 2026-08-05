"""
CamCoordTest_7: WORLD vs UNITY cam2tag + camera axes on ONE plot.
Tag aligned to origin so positions/directions are directly comparable.
Uses camera_pose_unity_cam2tag.csv only.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV = Path(__file__).with_name("camera_pose_unity_cam2tag.csv")
OUT = Path(__file__).with_name("pose_vector_compare") / "CamCoordTest_7_cam2tag_world_vs_unity.png"


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


def extrinsic_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """WORLD Extrinsic: Roll(Z) -> Pitch(X) -> Yaw(Y)."""
    return Ry(yaw) @ Rx(pitch) @ Rz(roll)


def unity_euler_zxy(ex: float, ey: float, ez: float) -> np.ndarray:
    """Unity Quaternion.Euler(x,y,z): Z then X then Y. R = Ry(y) @ Rx(x) @ Rz(z)."""
    return Ry(ey) @ Rx(ex) @ Rz(ez)


def draw_frame(ax, origin, R, scale, style, label_prefix, axis_alpha=1.0):
    """style: 'solid' WORLD, 'dashed' UNITY. Colors still RGB for X/Y/Z."""
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    names = ("X", "Y", "Z")
    ls = "-" if style == "solid" else "--"
    lw = 2.8 if style == "solid" else 2.2
    for i, (col, name) in enumerate(zip(colors, names)):
        v = origin + scale * R[:, i]
        ax.plot(
            [origin[0], v[0]],
            [origin[1], v[1]],
            [origin[2], v[2]],
            color=col,
            ls=ls,
            lw=lw,
            alpha=axis_alpha,
        )
        ax.text(
            v[0], v[1], v[2],
            f" {label_prefix}{name}",
            color=col,
            fontsize=8,
            fontweight="bold",
            alpha=axis_alpha,
        )


def set_equal(ax, pts, pad=0.01):
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    r = np.max(np.linalg.norm(pts - c, axis=1)) + pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    row = next(r for r in rows if r["image_file"] == "CamCoordTest_7.jpg")

    # WORLD: tag at 0
    cam_w = np.array(
        [float(row["cam_in_tag_x_m"]), float(row["cam_in_tag_y_m"]), float(row["cam_in_tag_z_m"])],
        dtype=float,
    )
    tag = np.zeros(3)
    roll = float(row["csv_roll_deg"])
    pitch = float(row["csv_pitch_deg"])
    yaw = float(row["csv_yaw_deg"])
    R_w = extrinsic_rpy(roll, pitch, yaw)

    # UNITY: shift so tag -> origin (same plot space)
    cam_u_abs = np.array(
        [float(row["unity_pos_x"]), float(row["unity_pos_y"]), float(row["unity_pos_z"])],
        dtype=float,
    )
    tag_u_abs = np.array(
        [float(row["original_x"]), float(row["original_y"]), float(row["original_z"])],
        dtype=float,
    )
    cam_u = cam_u_abs - tag_u_abs  # relative to tag
    ux = float(row["unity_rot_x"])
    uy = float(row["unity_rot_y"])
    uz = float(row["unity_rot_z"])
    R_u = unity_euler_zxy(ux, uy, uz)

    to_w = tag - cam_w
    to_u = tag - cam_u
    nw = np.linalg.norm(to_w) + 1e-12
    nu = np.linalg.norm(to_u) + 1e-12
    fw, fu = R_w[:, 2], R_u[:, 2]
    dw = float(np.dot(fw, to_w / nw))
    du = float(np.dot(fu, to_u / nu))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    scale = 0.028

    # shared tag
    ax.scatter(*tag, c="k", s=90, depthshade=False, zorder=6)
    ax.text(0.002, 0.002, 0.002, " TAG", color="k", fontsize=10, fontweight="bold")

    # WORLD camera + cam2tag (solid)
    ax.scatter(*cam_w, c="#ff7f0e", s=70, depthshade=False, zorder=5, label="WORLD cam")
    ax.quiver(
        cam_w[0], cam_w[1], cam_w[2],
        to_w[0], to_w[1], to_w[2],
        color="#ff7f0e", arrow_length_ratio=0.12, lw=2.4, alpha=0.95,
    )
    ax.text(*(cam_w + 0.5 * to_w), " W cam2tag", color="#ff7f0e", fontsize=9)
    draw_frame(ax, cam_w, R_w, scale, "solid", "W")

    # UNITY camera + cam2tag (dashed), tag-aligned
    ax.scatter(*cam_u, c="#17becf", s=70, depthshade=False, zorder=5, label="UNITY cam (tag-aligned)")
    ax.quiver(
        cam_u[0], cam_u[1], cam_u[2],
        to_u[0], to_u[1], to_u[2],
        color="#17becf", arrow_length_ratio=0.12, lw=2.2, alpha=0.95,
    )
    ax.text(*(cam_u + 0.5 * to_u), " U cam2tag", color="#17becf", fontsize=9)
    draw_frame(ax, cam_u, R_u, scale, "dashed", "U")

    # unit-direction overlay from tag for easy angle compare
    ax.quiver(
        0, 0, 0, *(to_w / nw * 0.05),
        color="#ff7f0e", arrow_length_ratio=0.15, lw=1.5, alpha=0.55,
    )
    ax.quiver(
        0, 0, 0, *(to_u / nu * 0.05),
        color="#17becf", arrow_length_ratio=0.15, lw=1.5, alpha=0.55,
    )

    # legend proxies
    ax.plot([], [], color="#ff7f0e", ls="-", lw=2.5, label="WORLD solid (Extrinsic RPY)")
    ax.plot([], [], color="#17becf", ls="--", lw=2.2, label="UNITY dashed (Euler ZXY)")
    ax.plot([], [], color="#d62728", lw=2, label="axis X")
    ax.plot([], [], color="#2ca02c", lw=2, label="axis Y")
    ax.plot([], [], color="#1f77b4", lw=2, label="axis Z (forward)")

    ax.set_title(
        "CamCoordTest_7 — WORLD vs UNITY on one plot (tag @ origin)\n"
        f"WORLD Extrinsic RPY=({roll:.2f},{pitch:.2f},{yaw:.2f})  fwd·cam2tag={dw:+.3f}\n"
        f"UNITY Euler (x,y,z)=({ux:.2f},{uy:.2f},{uz:.2f})  fwd·cam2tag={du:+.3f}",
        fontsize=11,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    set_equal(ax, np.vstack([tag, cam_w, cam_u, cam_w + scale * R_w, cam_u + scale * R_u]), pad=0.015)
    ax.view_init(elev=22, azim=-55)

    fig.text(
        0.5,
        0.015,
        "Orange solid = WORLD (right-handed Extrinsic Roll→Pitch→Yaw).  "
        "Cyan dashed = UNITY (left-handed coords, Euler ZXY), tag shifted to origin.\n"
        f"W cam={cam_w.tolist()}  cam2tag={to_w.tolist()}\n"
        f"U cam(rel)={cam_u.tolist()}  cam2tag={to_u.tolist()}   "
        f"| abs Unity cam={cam_u_abs.tolist()} tag={tag_u_abs.tolist()}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    print(f"Wrote {OUT}")
    print(f"WORLD cam={cam_w} cam2tag={to_w} fwd·={dw:+.3f}")
    print(f"UNITY cam(rel)={cam_u} cam2tag={to_u} fwd·={du:+.3f}")


if __name__ == "__main__":
    main()
