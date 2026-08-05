"""
Plot camera_pose_relative_to_tag.csv in REAL (right-handed) tag frame.

Extrinsic XYZ (fixed axes): Rx(roll) -> Ry(pitch) -> Rz(yaw)
  R_tag2cam = Rz(yaw) @ Ry(pitch) @ Rx(roll)

Shows:
  - TAG @ origin, CAM positions
  - gray dashed = geometric CAM -> TAG  (cam2tag by position)
  - red arrow   = R_tag2cam @ [0,0,1]     (tag2cam-axis sense if R is TAG->CAM)
  - blue arrow  = R_tag2cam.T @ [0,0,1]   (camera +Z in tag = optical forward)

Dot products tell which matches cam2tag.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
OUT = Path(__file__).with_name("real_extrinsic_xyz_cam_vectors.png")


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


def extrinsic_xyz_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic XYZ: X=roll, Y=pitch, Z=yaw. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def set_equal(ax, pts, pad=0.01):
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    r = float(np.max(np.linalg.norm(pts - c, axis=1))) + pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    return r


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    cams, c2tags, fwd_t2c, fwd_cam, dots_t2c, dots_cam = [], [], [], [], [], []
    labels = []

    print("Extrinsic XYZ: R_tag2cam = Rz(yaw)@Ry(pitch)@Rx(roll)")
    print("gray = geometric cam2tag; red = R@[0,0,1]; blue = R.T@[0,0,1] (cam +Z)")
    print()

    for i, row in enumerate(rows, start=1):
        cam = np.array(
            [float(row["camera_x_mm"]), float(row["camera_y_mm"]), float(row["camera_z_mm"])],
            dtype=float,
        ) / 1000.0
        roll = float(row["camera_roll_deg"])
        pitch = float(row["camera_pitch_deg"])
        yaw = float(row["camera_yaw_deg"])
        R = extrinsic_xyz_rpy(roll, pitch, yaw)  # interpret as TAG->CAM
        c2 = -cam
        c2n = c2 / (np.linalg.norm(c2) + 1e-12)
        # If R is TAG->CAM: columns of R.T = camera axes in tag
        v_red = R @ np.array([0.0, 0.0, 1.0])       # tag Z expressed? or R applied to +Z
        v_blue = R.T @ np.array([0.0, 0.0, 1.0])    # camera local +Z in tag
        v_red /= np.linalg.norm(v_red) + 1e-12
        v_blue /= np.linalg.norm(v_blue) + 1e-12
        d_red = float(np.dot(v_red, c2n))
        d_blue = float(np.dot(v_blue, c2n))

        cams.append(cam)
        c2tags.append(c2)
        fwd_t2c.append(v_red)
        fwd_cam.append(v_blue)
        dots_t2c.append(d_red)
        dots_cam.append(d_blue)
        labels.append(str(i))
        name = Path(row["image_file"]).name
        print(
            f"{i:2d} {name:20s}  "
            f"dot(R@z, cam2tag)={d_red:+.3f}  "
            f"dot(R.T@z, cam2tag)={d_blue:+.3f}"
        )

    cams = np.asarray(cams)
    c2tags = np.asarray(c2tags)
    fwd_t2c = np.asarray(fwd_t2c)
    fwd_cam = np.asarray(fwd_cam)
    fwd_len = 0.035

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    tag = np.zeros(3)
    ax.scatter(*tag, c="k", s=110, depthshade=False, zorder=6)
    ax.text(0.002, 0.002, 0.002, " TAG", fontsize=11, fontweight="bold")

    for i in range(len(rows)):
        ax.scatter(*cams[i], c="#ff7f0e", s=55, depthshade=False, zorder=5)
        # geometric cam2tag
        ax.plot(
            [cams[i, 0], 0], [cams[i, 1], 0], [cams[i, 2], 0],
            color="0.45", ls="--", lw=1.1, alpha=0.75,
        )
        # R @ +Z
        tip_r = cams[i] + fwd_len * fwd_t2c[i]
        ax.quiver(
            cams[i, 0], cams[i, 1], cams[i, 2],
            tip_r[0] - cams[i, 0], tip_r[1] - cams[i, 1], tip_r[2] - cams[i, 2],
            color="#d62728", arrow_length_ratio=0.18, lw=1.5, alpha=0.9,
        )
        # R.T @ +Z (camera forward if angles are TAG->CAM)
        tip_b = cams[i] + fwd_len * fwd_cam[i]
        ax.quiver(
            cams[i, 0], cams[i, 1], cams[i, 2],
            tip_b[0] - cams[i, 0], tip_b[1] - cams[i, 1], tip_b[2] - cams[i, 2],
            color="#1f77b4", arrow_length_ratio=0.18, lw=1.8, alpha=0.95,
        )
        ax.text(*cams[i], f" {labels[i]}", fontsize=8)

    ax.scatter([], [], c="#ff7f0e", s=40, label="CAM (CSV xyz in tag)")
    ax.plot([], [], color="0.45", ls="--", lw=1.5, label="geometric cam2tag (CAM→TAG)")
    ax.plot([], [], color="#d62728", lw=2, label="R@[0,0,1]  (if R=TAG→CAM: tag-Z in cam / raw)")
    ax.plot([], [], color="#1f77b4", lw=2, label="Rᵀ@[0,0,1] (camera +Z in tag)")
    ax.legend(loc="upper left", fontsize=8)

    # axis triad
    g = 0.04
    for v, col, name in [
        (np.array([g, 0, 0]), "#d62728", "+X"),
        (np.array([0, g, 0]), "#2ca02c", "+Y"),
        (np.array([0, 0, g]), "#1f77b4", "+Z"),
    ]:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2.2)
        ax.text(*v, f" {name}", color=col, fontsize=9, fontweight="bold")

    mean_r = float(np.mean(dots_t2c))
    mean_b = float(np.mean(dots_cam))
    ax.set_title(
        "REAL tag frame — Extrinsic XYZ  Rx(roll)→Ry(pitch)→Rz(yaw)\n"
        f"R = Rz(yaw)@Ry(pitch)@Rx(roll)\n"
        f"mean dot(R@z, cam2tag)={mean_r:+.3f}   "
        f"mean dot(Rᵀ@z, cam2tag)={mean_b:+.3f}\n"
        f"{'BLUE ≈ cam2tag → angles act as TAG→CAM, blue is camera look (cam→tag)' if mean_b > 0.5 else ''}"
        f"{'RED ≈ cam2tag → use R@z as look' if mean_r > 0.5 and mean_b <= 0.5 else ''}"
        f"{'BLUE ≈ −cam2tag → blue is tag2cam / looking away' if mean_b < -0.5 else ''}",
        fontsize=10,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_equal(ax, np.vstack([tag, cams]), pad=0.015)
    ax.view_init(elev=22, azim=-60)
    ax.text2D(
        0.02, 0.02, "RIGHT-HANDED (Real)",
        transform=ax.transAxes, fontsize=10, fontweight="bold", color="#2a9d8f",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    print()
    print(f"Wrote {OUT}")
    print(f"mean dot(R  @z, cam2tag) = {mean_r:+.3f}")
    print(f"mean dot(Rᵀ @z, cam2tag) = {mean_b:+.3f}")
    if mean_b > 0.5:
        print("=> BLUE (Rᵀ@[0,0,1]) aligns with cam2tag: camera looks toward TAG (cam→tag).")
        print("   CSV angles behave as TAG→CAM; optical forward = Rᵀ z.")
    elif mean_b < -0.5:
        print("=> BLUE aligns with TAG→CAM (away from tag): looking opposite to cam2tag.")
    if mean_r > 0.5:
        print("=> RED (R@[0,0,1]) aligns with cam2tag.")
    elif mean_r < -0.5:
        print("=> RED aligns opposite to cam2tag.")


if __name__ == "__main__":
    main()
