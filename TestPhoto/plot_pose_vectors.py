"""Update plot script for tag->cam invert conversion."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import convert_pose_to_unity as c

OUT_DIR = Path(__file__).with_name("pose_vector_compare")


def axes_from_R(R):
    return R[:, 0], R[:, 1], R[:, 2]


def draw_frame(ax, origin, R, scale, prefix):
    o = np.asarray(origin, float)
    for vec, col, name in zip(axes_from_R(R), ("r", "g", "b"), ("X", "Y", "Z")):
        v = o + scale * vec
        ax.plot([o[0], v[0]], [o[1], v[1]], [o[2], v[2]], color=col, lw=2.5)
        ax.text(v[0], v[1], v[2], f" {prefix}{name}", color=col, fontsize=8)


def set_equal(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def quat_str(q):
    return f"({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(c.CSV_IN.open(encoding="utf-8-sig")))
    lines = []
    dots_w, dots_u = [], []

    for row in rows:
        name = Path(row["image_file"]).name
        r = c.convert_one(
            float(row["camera_roll_deg"]),
            float(row["camera_pitch_deg"]),
            float(row["camera_yaw_deg"]),
            float(row["camera_x_mm"]),
            float(row["camera_y_mm"]),
            float(row["camera_z_mm"]),
        )
        R_tag = r["R_cam_in_tag"]  # camera in tag/world after invert
        R_u = r["R_unity"]
        t_tag = r["t_cam_in_tag"]
        t_u = r["t_unity"]
        qw = c.mat_to_quat(R_tag)
        qu = c.mat_to_quat(R_u)

        tag_w = np.zeros(3)
        tag_u = c.ORIGINAL.copy()
        fw = R_tag[:, 2]
        fu = R_u[:, 2]
        to_w = (tag_w - t_tag) / (np.linalg.norm(tag_w - t_tag) + 1e-12)
        to_u = (tag_u - t_u) / (np.linalg.norm(tag_u - t_u) + 1e-12)
        dw, du = float(np.dot(fw, to_w)), float(np.dot(fu, to_u))
        dots_w.append(dw)
        dots_u.append(du)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        scale = 0.04

        ax1.set_title(f"WORLD (cam in tag) after invert\n{name}")
        draw_frame(ax1, t_tag, R_tag, scale, "W")
        ax1.scatter(*tag_w, c="k", s=40)
        ax1.scatter(*t_tag, c="m", s=30)
        ax1.plot([t_tag[0], 0], [t_tag[1], 0], [t_tag[2], 0], "k--", lw=1.5)
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
        set_equal(ax1, t_tag, 0.08)

        ax2.set_title(f"UNITY S=diag(-1,1,1)\n{name}")
        draw_frame(ax2, t_u, R_u, scale, "U")
        ax2.scatter(*tag_u, c="k", s=40)
        ax2.scatter(*t_u, c="m", s=30)
        ax2.plot([t_u[0], tag_u[0]], [t_u[1], tag_u[1]], [t_u[2], tag_u[2]], "k--", lw=1.5)
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
        set_equal(ax2, t_u, 0.08)

        fig.text(
            0.02,
            0.02,
            f"CSV = TAG->CAM (Extrinsic RPY), then invert to CAM in TAG, then S for Unity\n"
            f"quat cam_in_tag {quat_str(qw)}\n"
            f"quat unity      {quat_str(qu)}\n"
            f"forward·to_tag: world={dw:+.3f}  unity={du:+.3f}   (want +1)\n"
            f"RGB=X/Y/Z(forward)  black dashed=to tag",
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        out = OUT_DIR / f"{Path(name).stem}_vectors.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        lines.append(
            f"{name}\n  quat_cam_in_tag {quat_str(qw)}\n  quat_unity {quat_str(qu)}\n"
            f"  look_dot world={dw:+.3f} unity={du:+.3f}\n"
        )
        print(f"Wrote {out.name}  unity_dot={du:+.3f}")

    # overview
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, title, dots in zip(
        axes,
        ["WORLD cam-in-tag forward vs to-tag (X-Z)", "UNITY forward vs to-tag (X-Z)"],
        [dots_w, dots_u],
    ):
        # recompute for quiver
        pass
    # simpler overview from stored - regenerate quickly
    fw_list_w, tw_list_w, fw_list_u, tw_list_u = [], [], [], []
    for row in rows:
        r = c.convert_one(
            float(row["camera_roll_deg"]),
            float(row["camera_pitch_deg"]),
            float(row["camera_yaw_deg"]),
            float(row["camera_x_mm"]),
            float(row["camera_y_mm"]),
            float(row["camera_z_mm"]),
        )
        fw_list_w.append(r["R_cam_in_tag"][:, 2])
        tw_list_w.append(-r["t_cam_in_tag"] / (np.linalg.norm(r["t_cam_in_tag"]) + 1e-12))
        fw_list_u.append(r["R_unity"][:, 2])
        to = c.ORIGINAL - r["t_unity"]
        tw_list_u.append(to / (np.linalg.norm(to) + 1e-12))

    for ax, title, fws, tws, dots in [
        (axes[0], "WORLD after invert", fw_list_w, tw_list_w, dots_w),
        (axes[1], "UNITY S R S", fw_list_u, tw_list_u, dots_u),
    ]:
        for i, (f, t) in enumerate(zip(fws, tws)):
            ax.arrow(0, 0, f[0], f[2], head_width=0.05, color="b", alpha=0.35, length_includes_head=True)
            ax.arrow(0, 0, t[0], t[2], head_width=0.05, color="k", alpha=0.25, length_includes_head=True)
            ax.text(f[0] * 1.05, f[2] * 1.05, str(i + 1), color="b", fontsize=7)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_title(title)
        ax.text(0.02, 0.98, f"mean dot={np.mean(dots):+.2f}", transform=ax.transAxes, va="top")
        ax.plot([], [], "b-", label="forward"); ax.plot([], [], "k-", label="to tag"); ax.legend(fontsize=8)

    fig.suptitle("After invert(tag->cam): blue forward should match black to-tag")
    fig.tight_layout()
    overview = OUT_DIR / "overview_forward_vs_tag.png"
    fig.savefig(overview, dpi=140)
    plt.close(fig)

    report = OUT_DIR / "quat_vector_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Overview {overview}")
    print(f"mean unity look_dot = {np.mean(dots_u):+.3f}")


if __name__ == "__main__":
    main()
