"""
Regenerate Unity poses with X-flip similarity, then plot WORLD(mapped) vs UNITY
cam2tag for CamCoordTest_7 on one axes.

  S = diag(-1, 1, 1)
  t_unity_rel = S @ t_cam_in_tag
  R_unity     = S @ R_cam_in_tag @ S

WORLD is also drawn after the same S so both share one comparable frame (tag @ 0).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import convert_pose_to_unity as c

CSV_IN = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
CSV_OUT = Path(__file__).with_name("camera_pose_unity_S.csv")
OUT_PNG = Path(__file__).with_name("pose_vector_compare") / "CamCoordTest_7_cam2tag_after_S.png"

S = c.S
ORIGINAL = c.ORIGINAL


def convert_one(roll: float, pitch: float, yaw: float, x_mm: float, y_mm: float, z_mm: float):
    R_tag_to_cam = c.extrinsic_rpy_matrix(roll, pitch, yaw)
    t_cam_w = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0  # cam pos in tag/world
    R_cam_w = R_tag_to_cam.T  # camera axes in tag/world

    t_cam_u_rel = S @ t_cam_w
    R_unity = S @ R_cam_w @ S
    t_unity = ORIGINAL + t_cam_u_rel
    e_unity = c.quat_to_unity_euler(c.mat_to_quat(R_unity))

    cam2tag_w = -t_cam_w
    cam2tag_u = -t_cam_u_rel  # tag at origin in relative frame
    # mapped world (apply S) for overlay check
    t_cam_w_S = S @ t_cam_w
    cam2tag_w_S = S @ cam2tag_w
    R_w_S = S @ R_cam_w @ S

    fwd_u = R_unity[:, 2]
    to_u_n = cam2tag_u / (np.linalg.norm(cam2tag_u) + 1e-12)
    look = float(np.dot(fwd_u, to_u_n))

    return {
        "t_cam_w": t_cam_w,
        "R_cam_w": R_cam_w,
        "cam2tag_w": cam2tag_w,
        "t_cam_w_S": t_cam_w_S,
        "R_w_S": R_w_S,
        "cam2tag_w_S": cam2tag_w_S,
        "t_cam_u_rel": t_cam_u_rel,
        "R_unity": R_unity,
        "cam2tag_u": cam2tag_u,
        "t_unity": t_unity,
        "e_unity": e_unity,
        "look_dot": look,
    }


def draw_frame(ax, origin, R, scale, ls, prefix, alpha=1.0):
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for i, (col, name) in enumerate(zip(colors, ("X", "Y", "Z"))):
        v = origin + scale * R[:, i]
        ax.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]],
                color=col, ls=ls, lw=2.6, alpha=alpha)
        ax.text(v[0], v[1], v[2], f" {prefix}{name}", color=col, fontsize=8, fontweight="bold")


def set_equal(ax, pts, pad=0.012):
    pts = np.asarray(pts, float)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    out_rows = []
    row7 = None
    r7 = None

    print("S = diag(-1,1,1);  t_u = S@t_w;  R_u = S@R_w@S")
    print(f"original (tag in Unity) = {ORIGINAL.tolist()}\n")

    for row in rows:
        name = Path(row["image_file"]).name
        roll = float(row["camera_roll_deg"])
        pitch = float(row["camera_pitch_deg"])
        yaw = float(row["camera_yaw_deg"])
        x_mm = float(row["camera_x_mm"])
        y_mm = float(row["camera_y_mm"])
        z_mm = float(row["camera_z_mm"])
        r = convert_one(roll, pitch, yaw, x_mm, y_mm, z_mm)
        tu, eu = r["t_unity"], r["e_unity"]
        tw, tws = r["t_cam_w"], r["t_cam_w_S"]
        c2w, c2ws, c2u = r["cam2tag_w"], r["cam2tag_w_S"], r["cam2tag_u"]
        pos_err = float(np.linalg.norm(r["t_cam_u_rel"] - tws))
        c2_err = float(np.linalg.norm(c2u - c2ws))

        out_rows.append(
            {
                "image_file": name,
                "tag_id": row["tag_id"],
                "csv_x_mm": x_mm,
                "csv_y_mm": y_mm,
                "csv_z_mm": z_mm,
                "csv_roll_deg": roll,
                "csv_pitch_deg": pitch,
                "csv_yaw_deg": yaw,
                "world_cam_x": f"{tw[0]:.9f}",
                "world_cam_y": f"{tw[1]:.9f}",
                "world_cam_z": f"{tw[2]:.9f}",
                "world_cam2tag_x": f"{c2w[0]:.9f}",
                "world_cam2tag_y": f"{c2w[1]:.9f}",
                "world_cam2tag_z": f"{c2w[2]:.9f}",
                "world_S_cam_x": f"{tws[0]:.9f}",
                "world_S_cam_y": f"{tws[1]:.9f}",
                "world_S_cam_z": f"{tws[2]:.9f}",
                "world_S_cam2tag_x": f"{c2ws[0]:.9f}",
                "world_S_cam2tag_y": f"{c2ws[1]:.9f}",
                "world_S_cam2tag_z": f"{c2ws[2]:.9f}",
                "original_x": float(ORIGINAL[0]),
                "original_y": float(ORIGINAL[1]),
                "original_z": float(ORIGINAL[2]),
                "unity_pos_x": f"{tu[0]:.9f}",
                "unity_pos_y": f"{tu[1]:.9f}",
                "unity_pos_z": f"{tu[2]:.9f}",
                "unity_cam2tag_x": f"{c2u[0]:.9f}",
                "unity_cam2tag_y": f"{c2u[1]:.9f}",
                "unity_cam2tag_z": f"{c2u[2]:.9f}",
                "unity_rot_x": f"{eu[0]:.6f}",
                "unity_rot_y": f"{eu[1]:.6f}",
                "unity_rot_z": f"{eu[2]:.6f}",
                "look_dot_to_tag": f"{r['look_dot']:.4f}",
                "pos_match_err": f"{pos_err:.3e}",
                "cam2tag_match_err": f"{c2_err:.3e}",
                "rot_mode": "invert(tag->cam); t_u=S@t; R_u=S@R@S",
            }
        )
        print(
            f"{name:20s}  Upos=({tu[0]:.4f},{tu[1]:.4f},{tu[2]:.4f})  "
            f"Urot=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f})  "
            f"cam2tag_err={c2_err:.1e}  look={r['look_dot']:+.3f}"
        )
        if name == "CamCoordTest_7.jpg":
            row7, r7 = row, r

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {CSV_OUT}")

    # ---- single-plot comparison for image 7 ----
    assert r7 is not None
    tag = np.zeros(3)
    cam_ws = r7["t_cam_w_S"]
    cam_u = r7["t_cam_u_rel"]
    to_ws = r7["cam2tag_w_S"]
    to_u = r7["cam2tag_u"]
    R_ws = r7["R_w_S"]
    R_u = r7["R_unity"]
    eu = r7["e_unity"]

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    scale = 0.028

    ax.scatter(*tag, c="k", s=90, depthshade=False, zorder=6)
    ax.text(0.002, 0.002, 0.002, " TAG", color="k", fontsize=10, fontweight="bold")

    # WORLD after S (solid orange)
    ax.scatter(*cam_ws, c="#ff7f0e", s=80, depthshade=False, zorder=5, label="WORLD after S (cam)")
    ax.quiver(cam_ws[0], cam_ws[1], cam_ws[2], to_ws[0], to_ws[1], to_ws[2],
              color="#ff7f0e", arrow_length_ratio=0.12, lw=2.6)
    ax.text(*(cam_ws + 0.45 * to_ws), " W_S cam2tag", color="#ff7f0e", fontsize=9)
    draw_frame(ax, cam_ws, R_ws, scale, "-", "W")

    # UNITY (dashed cyan) — should overlap WORLD after S
    ax.scatter(*cam_u, c="#17becf", s=55, depthshade=False, zorder=5, label="UNITY new (cam, tag-rel)")
    ax.quiver(cam_u[0], cam_u[1], cam_u[2], to_u[0], to_u[1], to_u[2],
              color="#17becf", arrow_length_ratio=0.12, lw=2.0, alpha=0.85)
    ax.text(*(cam_u + 0.55 * to_u + np.array([0, 0.004, 0])), " U cam2tag", color="#17becf", fontsize=9)
    draw_frame(ax, cam_u, R_u, scale, "--", "U", alpha=0.85)

    ax.plot([], [], color="#ff7f0e", ls="-", lw=2.5, label="WORLD after S  (solid)")
    ax.plot([], [], color="#17becf", ls="--", lw=2.2, label="UNITY S@R@S     (dashed)")
    ax.plot([], [], color="#d62728", lw=2, label="X")
    ax.plot([], [], color="#2ca02c", lw=2, label="Y")
    ax.plot([], [], color="#1f77b4", lw=2, label="Z forward")

    pos_err = float(np.linalg.norm(cam_u - cam_ws))
    c2_err = float(np.linalg.norm(to_u - to_ws))
    ax.set_title(
        "CamCoordTest_7 — after S: WORLD(mapped) vs new UNITY (one frame, tag@0)\n"
        f"t_u = S@t_w ,  R_u = S@R_w@S\n"
        f"Unity euler=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f})  "
        f"|cam pos err|={pos_err:.2e}  |cam2tag err|={c2_err:.2e}  "
        f"look_dot={r7['look_dot']:+.3f}",
        fontsize=11,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    set_equal(ax, np.vstack([tag, cam_ws, cam_u, cam_ws + scale * R_ws, cam_u + scale * R_u]))
    ax.view_init(elev=22, azim=-55)

    tw = r7["t_cam_w"]
    fig.text(
        0.5, 0.015,
        "Orange solid = WORLD pose after S.  Cyan dashed = new UNITY (should coincide).\n"
        f"raw WORLD cam={tw.tolist()}  cam2tag={r7['cam2tag_w'].tolist()}\n"
        f"WORLD after S cam={cam_ws.tolist()}  cam2tag={to_ws.tolist()}\n"
        f"UNITY tag-rel  cam={cam_u.tolist()}  cam2tag={to_u.tolist()}   "
        f"abs Unity pos={(ORIGINAL + cam_u).tolist()}",
        ha="center", va="bottom", fontsize=8.5, family="monospace",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
