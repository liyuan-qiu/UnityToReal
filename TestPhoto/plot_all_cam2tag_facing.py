"""
REAL (right-handed): tag, cam, cam2tag vector.
UNITY (left-handed view): tag, cam, then from cam the forward vector
  from Unity Euler (Quaternion.Euler / ZXY order).

Data: camera_pose_unity_facing_cam2tag.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

CSV = Path(__file__).with_name("camera_pose_unity_facing_xyz.csv")
OUT = Path(__file__).with_name("pose_vector_compare") / "all_cam2tag_real_vs_unity.png"


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


def unity_euler_matrix(ex: float, ey: float, ez: float) -> np.ndarray:
    """Unity Quaternion.Euler(x,y,z): apply Z, then X, then Y. R = Ry(y)@Rx(x)@Rz(z)."""
    return Ry(ey) @ Rx(ex) @ Rz(ez)


def extrinsic_xyz_matrix(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """
    Extrinsic XYZ (fixed axes): X → Y → Z
      first Rx(roll), then Ry(yaw), then Rz(pitch)
      R = Rz(pitch) @ Ry(yaw) @ Rx(roll)
    CSV angles used as: roll→X, yaw→Y, pitch→Z (per user).
    """
    return Rz(pitch) @ Ry(yaw) @ Rx(roll)


def to_lh_display(p: np.ndarray) -> np.ndarray:
    """Embed LH numeric coords into matplotlib RH canvas by mirroring X for display only."""
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


def draw_axis_triad(ax, origin, scale, left_handed_display: bool):
    """RGB = +X/+Y/+Z. If left_handed_display, embed with X mirrored."""
    origin = np.asarray(origin, float)
    for vec, col, name in [
        (np.array([scale, 0.0, 0.0]), "#d62728", "+X"),
        (np.array([0.0, scale, 0.0]), "#2ca02c", "+Y"),
        (np.array([0.0, 0.0, scale]), "#1f77b4", "+Z"),
    ]:
        d = to_lh_display(vec) if left_handed_display else vec
        tip = origin + d
        ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], color=col, lw=2.6)
        ax.text(*tip, f" {name}", color=col, fontsize=9, fontweight="bold")


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ---- gather ----
    cams_r, c2_r, fwds_r, dots_r = [], [], [], []
    cams_u, tags_u, c2_u, fwds_u, eulers_u = [], [], [], [], []
    labels, looks = [], []

    for i, row in enumerate(rows, start=1):
        # REAL RHS: camera in tag frame
        cam_r = np.array(
            [float(row["csv_x_mm"]), float(row["csv_y_mm"]), float(row["csv_z_mm"])],
            dtype=float,
        ) / 1000.0
        to_r = -cam_r  # geometric cam → tag (for reference only)
        to_r_n = to_r / (np.linalg.norm(to_r) + 1e-12)

        # Extrinsic XYZ: Rx(roll) → Ry(yaw) → Rz(pitch); invert → cam axes in tag
        roll = float(row["csv_roll_deg"])
        pitch = float(row["csv_pitch_deg"])
        yaw = float(row["csv_yaw_deg"])
        R_tag_to_cam = extrinsic_xyz_matrix(roll, yaw, pitch)
        R_cam_in_tag = R_tag_to_cam.T
        fwd_r = R_cam_in_tag @ np.array([0.0, 0.0, 1.0])  # optical +Z in REAL
        dot_r = float(np.dot(fwd_r, to_r_n))

        # UNITY: CSV absolute pose + Euler facing
        tag_u = np.array(
            [float(row["original_x"]), float(row["original_y"]), float(row["original_z"])],
            dtype=float,
        )
        cam_u = np.array(
            [float(row["unity_pos_x"]), float(row["unity_pos_y"]), float(row["unity_pos_z"])],
            dtype=float,
        )
        to_u = np.array(
            [float(row["unity_cam2tag_x"]), float(row["unity_cam2tag_y"]), float(row["unity_cam2tag_z"])],
            dtype=float,
        )
        eu = (
            float(row["unity_rot_x"]),
            float(row["unity_rot_y"]),
            float(row["unity_rot_z"]),
        )
        R_u = unity_euler_matrix(*eu)
        fwd_u = R_u @ np.array([0.0, 0.0, 1.0])  # Unity camera forward = local +Z

        cams_r.append(cam_r)
        c2_r.append(to_r)
        fwds_r.append(fwd_r)
        dots_r.append(dot_r)
        cams_u.append(cam_u)
        tags_u.append(tag_u)
        c2_u.append(to_u)
        fwds_u.append(fwd_u)
        eulers_u.append(eu)
        labels.append(str(i))
        looks.append(float(row["look_dot_cam2tag"]))

    cams_r = np.asarray(cams_r)
    c2_r = np.asarray(c2_r)
    fwds_r = np.asarray(fwds_r)
    cams_u = np.asarray(cams_u)
    tags_u = np.asarray(tags_u)
    c2_u = np.asarray(c2_u)
    fwds_u = np.asarray(fwds_u)

    # scale for orientation arrows (meters)
    fwd_len = 0.035

    fig = plt.figure(figsize=(16.5, 8.0))
    ax_r = fig.add_subplot(121, projection="3d")
    ax_u = fig.add_subplot(122, projection="3d")

    # ========== REAL: RH — tag, cam, geometric link + Extrinsic forward ==========
    tag_r = np.zeros(3)
    ax_r.scatter(*tag_r, c="k", s=100, depthshade=False, zorder=6, label="TAG")
    ax_r.text(0.002, 0.002, 0.002, " TAG", fontsize=10, fontweight="bold")

    for i in range(len(rows)):
        ax_r.scatter(*cams_r[i], c="#ff7f0e", s=50, depthshade=False, zorder=5)
        # geometric CAM→TAG (reference only — always hits TAG by construction)
        ax_r.plot(
            [cams_r[i, 0], tag_r[0]],
            [cams_r[i, 1], tag_r[1]],
            [cams_r[i, 2], tag_r[2]],
            color="0.55", ls="--", lw=1.0, alpha=0.65,
        )
        # orientation from Extrinsic XYZ (this is the real check)
        tip = cams_r[i] + fwd_len * fwds_r[i]
        ax_r.quiver(
            cams_r[i, 0], cams_r[i, 1], cams_r[i, 2],
            tip[0] - cams_r[i, 0], tip[1] - cams_r[i, 1], tip[2] - cams_r[i, 2],
            color="#d62728", arrow_length_ratio=0.18, lw=1.8, alpha=0.95,
        )
        ax_r.text(*cams_r[i], f" {labels[i]}", fontsize=8, color="0.25")

    ax_r.scatter([], [], c="#ff7f0e", s=40, label="CAM")
    ax_r.plot([], [], color="0.55", ls="--", lw=1.5, label="CAM–TAG line (geometry)")
    ax_r.plot([], [], color="#d62728", lw=2.2, label="forward Extrinsic XYZ")
    r_span = set_equal(ax_r, np.vstack([tag_r, cams_r]), pad=0.012)
    draw_axis_triad(ax_r, tag_r, 0.5 * r_span, left_handed_display=False)
    ax_r.text2D(
        0.02, 0.02, "RIGHT-HANDED (Real)",
        transform=ax_r.transAxes, fontsize=10, fontweight="bold", color="#2a9d8f",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax_r.set_title(
        "REAL (right-handed)\n"
        "1) TAG @0  2) CAM = CSV xyz/1000\n"
        "3) red = Extrinsic XYZ: Rx(roll)→Ry(yaw)→Rz(pitch)\n"
        f"   R=Rz(pitch)@Ry(yaw)@Rx(roll), then Rᵀ@[0,0,1]\n"
        f"   mean fwd·cam2tag={np.mean(dots_r):+.3f}   gray=geometric CAM–TAG",
        fontsize=10,
    )
    ax_r.set_xlabel("X"); ax_r.set_ylabel("Y"); ax_r.set_zlabel("Z")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.view_init(elev=22, azim=-60)

    # ========== UNITY: LH view — tag, cam, forward from Unity Euler ==========
    tag0 = tags_u[0]
    tag_d = to_lh_display(tag0)
    cams_d = np.array([to_lh_display(p) for p in cams_u])
    # cam2tag in display: tip direction also LH-embedded
    c2_d = np.array([to_lh_display(v) for v in c2_u])
    fwds_d = np.array([to_lh_display(v) for v in fwds_u])

    ax_u.scatter(*tag_d, c="k", s=100, depthshade=False, zorder=6, label="TAG")
    ax_u.text(tag_d[0] + 0.002, tag_d[1] + 0.002, tag_d[2] + 0.002, " TAG", fontsize=10, fontweight="bold")

    for i in range(len(rows)):
        ax_u.scatter(*cams_d[i], c="#17becf", s=50, depthshade=False, zorder=5)
        # 1) cam2tag (thin dashed) — position relation
        ax_u.plot(
            [cams_d[i, 0], cams_d[i, 0] + c2_d[i, 0]],
            [cams_d[i, 1], cams_d[i, 1] + c2_d[i, 1]],
            [cams_d[i, 2], cams_d[i, 2] + c2_d[i, 2]],
            color="0.45", ls="--", lw=1.0, alpha=0.7,
        )
        # 2) forward from Unity Euler (main vector user asked for)
        tip = cams_d[i] + fwd_len * fwds_d[i]
        ax_u.quiver(
            cams_d[i, 0], cams_d[i, 1], cams_d[i, 2],
            tip[0] - cams_d[i, 0], tip[1] - cams_d[i, 1], tip[2] - cams_d[i, 2],
            color="#1f77b4", arrow_length_ratio=0.18, lw=1.8, alpha=0.95,
        )
        ax_u.text(*cams_d[i], f" {labels[i]}", fontsize=8, color="0.25")

    ax_u.scatter([], [], c="#17becf", s=40, label="CAM (unity_pos)")
    ax_u.plot([], [], color="0.45", ls="--", lw=1.5, label="cam2tag (CSV)")
    ax_u.plot([], [], color="#1f77b4", lw=2.2, label="forward from Unity Euler")
    u_span = set_equal(ax_u, np.vstack([tag_d.reshape(1, 3), cams_d]), pad=0.012)
    draw_axis_triad(ax_u, tag_d, 0.5 * u_span, left_handed_display=True)
    ax_u.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{-v:.2f}"))
    ax_u.text2D(
        0.02, 0.02, "LEFT-HANDED (Unity view)",
        transform=ax_u.transAxes, fontsize=10, fontweight="bold", color="#c44e52",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax_u.set_title(
        "UNITY (left-handed display)\n"
        "1) TAG = original_*   2) CAM = unity_pos_*\n"
        "3) blue = forward from Euler ZXY (local +Z)\n"
        "   gray dashed = cam2tag   mean look_dot="
        f"{np.mean(looks):+.3f}",
        fontsize=10,
    )
    ax_u.set_xlabel("Unity X (LH)")
    ax_u.set_ylabel("Unity Y")
    ax_u.set_zlabel("Unity Z")
    ax_u.legend(loc="upper left", fontsize=8)
    ax_u.view_init(elev=22, azim=-60)

    fig.suptitle(
        "REAL Extrinsic XYZ vs UNITY (from same XYZ + S=diag(-1,1,1))\n"
        "Unity forward.x = −Real forward.x by design (handedness). "
        "After S they match (see camera_pose_unity_facing_xyz.csv).",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.05)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    # Overlay in shared frame: REAL after S vs UNITY tag-relative (should coincide)
    S = np.diag([-1.0, 1.0, 1.0])
    fig2 = plt.figure(figsize=(9.5, 8))
    ax = fig2.add_subplot(111, projection="3d")
    ax.scatter(0, 0, 0, c="k", s=90, depthshade=False)
    ax.text(0.002, 0.002, 0.002, " TAG", fontsize=10, fontweight="bold")
    errs = []
    for i in range(len(rows)):
        cam_s = S @ cams_r[i]
        fwd_s = S @ fwds_r[i]
        cam_ur = cams_u[i] - tags_u[i]
        ax.scatter(*cam_s, c="#ff7f0e", s=50, depthshade=False, alpha=0.9)
        ax.scatter(*cam_ur, c="#17becf", s=30, depthshade=False, alpha=0.9)
        tip_s = cam_s + fwd_len * fwd_s
        tip_u = cam_ur + fwd_len * fwds_u[i]
        ax.plot([cam_s[0], tip_s[0]], [cam_s[1], tip_s[1]], [cam_s[2], tip_s[2]], color="#ff7f0e", lw=2.0)
        ax.plot([cam_ur[0], tip_u[0]], [cam_ur[1], tip_u[1]], [cam_ur[2], tip_u[2]], color="#17becf", lw=1.6, ls="--")
        ax.text(*cam_ur, f" {labels[i]}", fontsize=8)
        errs.append(float(np.linalg.norm(fwd_s / (np.linalg.norm(fwd_s) + 1e-12) - fwds_u[i] / (np.linalg.norm(fwds_u[i]) + 1e-12))))
    ax.plot([], [], color="#ff7f0e", lw=2, label="REAL after S (fwd)")
    ax.plot([], [], color="#17becf", ls="--", lw=2, label="UNITY fwd")
    ax.legend(loc="upper left")
    ax.set_title(
        f"Shared frame check: S@REAL_fwd vs UNITY_fwd\n"
        f"mean |dir err|={np.mean(errs):.2e}  (0 => X flip is only S, then identical)",
        fontsize=11,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_equal(ax, np.vstack([np.zeros(3), np.array([S @ c for c in cams_r])]), pad=0.012)
    ax.view_init(elev=22, azim=-60)
    out2 = OUT.with_name("all_cam2tag_overlay_S_match.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=160)
    plt.close(fig2)

    print(f"Wrote {OUT}")
    print(f"Wrote {out2}")
    print("REAL Extrinsic XYZ: R=Rz(pitch)@Ry(yaw)@Rx(roll); fwd=R.T@[0,0,1]")
    print("UNITY: same XYZ -> S@R@S (no Rx180); blue=Unity Euler forward")
    print(f"REAL mean fwd·cam2tag={np.mean(dots_r):+.3f}   UNITY mean look_dot={np.mean(looks):+.3f}")
    print(f"overlay mean |S@REAL_fwd - UNITY_fwd|={np.mean(errs):.2e}")
    for i, eu in enumerate(eulers_u):
        print(
            f"  {i+1:2d} REAL_dot={dots_r[i]:+.3f}  "
            f"UNITY euler=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f}) look={looks[i]:+.3f}"
        )


if __name__ == "__main__":
    main()
