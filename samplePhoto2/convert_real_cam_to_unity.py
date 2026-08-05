"""
REAL photo poses (NOT tag) -> Unity.

Axis map (RH real -> LH Unity):
  real X -> unity X
  real Y -> unity -Y
  real Z -> unity Z
  S = diag(1, -1, 1)

Input CSV: coordinate2_filled.csv
  CamX/Y/Z     : camera position relative to origin (mm)
  CamRoll/Pitch/Yaw : CAM orientation, Extrinsic XYZ
      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
      face = R @ [0,0,1]

Unity:
  original = (0.03160001, -2.2834, 12.5992)
  t_unity  = original + S @ t_cam
  R_unity  = S @ R @ S

Outputs (all under this folder):
  camera_pose_unity_real_photos.csv
  real_cam_vectors.png
  real_vs_unity_real_photos.png

Do NOT import or reuse tag conversion scripts.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "coordinate2_filled.csv"
CSV_OUT = HERE / "camera_pose_unity_real_photos.csv"
PNG_REAL = HERE / "real_cam_vectors.png"
PNG_COMPARE = HERE / "real_vs_unity_real_photos.png"

ORIGINAL = np.array([0.03160001, -2.2834, 12.5992], dtype=float)
S = np.diag([1.0, -1.0, 1.0])  # real (x,y,z) -> unity (x,-y,z)


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


def extrinsic_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic XYZ: Rx(roll) -> Ry(pitch) -> Rz(yaw)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


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
    return (
        wrap360(math.degrees(pitch)),
        wrap360(math.degrees(yaw)),
        wrap360(math.degrees(roll)),
    )


def to_lh_display(p: np.ndarray) -> np.ndarray:
    """Flip displayed X so matplotlib RH axes look like Unity LH (X flipped view)."""
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


def load_rows() -> list[dict]:
    with CSV_IN.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def convert_all(rows: list[dict]) -> list[dict]:
    out = []
    print("REAL photos (not tag): ExtXYZ Roll->Pitch->Yaw; face=R@[0,0,1]")
    print(f"S = diag(1,-1,1)  original = {ORIGINAL.tolist()}")
    print()
    for row in rows:
        name = row["photo#"].strip()
        roll = float(row["CamRoll"])
        pitch = float(row["CamPitch"])
        yaw = float(row["CamYaw"])
        t = np.array(
            [float(row["CamX"]), float(row["CamY"]), float(row["CamZ"])],
            dtype=float,
        ) / 1000.0

        R = extrinsic_xyz(roll, pitch, yaw)
        face_r = R @ np.array([0.0, 0.0, 1.0])
        face_r = face_r / (np.linalg.norm(face_r) + 1e-12)

        t_u = ORIGINAL + (S @ t)
        R_u = S @ R @ S
        q = mat_to_quat(R_u)
        if q[3] < 0:
            q = (-q[0], -q[1], -q[2], -q[3])
        eu = quat_to_unity_euler(q)
        face_u = R_u[:, 2]
        face_u = face_u / (np.linalg.norm(face_u) + 1e-12)
        match = float(np.linalg.norm(S @ face_r - face_u))

        # direction from cam toward origin (geometric), for reference
        to_origin = -t
        to_n = to_origin / (np.linalg.norm(to_origin) + 1e-12)
        look_dot = float(np.dot(face_r, to_n))

        out.append(
            {
                # image_file: for PoseCsvAutoCapture.cs (same role as tag CSV)
                "image_file": f"{name}.jpg",
                "photo": name,
                "csv_CamX_mm": float(row["CamX"]),
                "csv_CamY_mm": float(row["CamY"]),
                "csv_CamZ_mm": float(row["CamZ"]),
                "csv_CamRoll_deg": roll,
                "csv_CamPitch_deg": pitch,
                "csv_CamYaw_deg": yaw,
                "LinearX": row.get("LinearX", ""),
                "LinearY": row.get("LinearY", ""),
                "LinearZ": row.get("LinearZ", ""),
                "original_x": float(ORIGINAL[0]),
                "original_y": float(ORIGINAL[1]),
                "original_z": float(ORIGINAL[2]),
                "unity_pos_x": f"{t_u[0]:.9f}",
                "unity_pos_y": f"{t_u[1]:.9f}",
                "unity_pos_z": f"{t_u[2]:.9f}",
                "unity_rot_x": f"{eu[0]:.6f}",
                "unity_rot_y": f"{eu[1]:.6f}",
                "unity_rot_z": f"{eu[2]:.6f}",
                "unity_quat_x": f"{q[0]:.9f}",
                "unity_quat_y": f"{q[1]:.9f}",
                "unity_quat_z": f"{q[2]:.9f}",
                "unity_quat_w": f"{q[3]:.9f}",
                "face_real_x": f"{face_r[0]:.6f}",
                "face_real_y": f"{face_r[1]:.6f}",
                "face_real_z": f"{face_r[2]:.6f}",
                "face_unity_x": f"{face_u[0]:.6f}",
                "face_unity_y": f"{face_u[1]:.6f}",
                "face_unity_z": f"{face_u[2]:.6f}",
                "look_dot_face_to_origin": f"{look_dot:.4f}",
                "face_S_match_err": f"{match:.2e}",
                "rot_mode": "REAL ExtXYZ Rx(roll)->Ry(pitch)->Rz(yaw); face=R@z; S=diag(1,-1,1); R_u=S@R@S",
            }
        )
        print(
            f"{name:10s}  pos_u=({t_u[0]:.4f},{t_u[1]:.4f},{t_u[2]:.4f})  "
            f"euler=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f})  "
            f"look_dot={look_dot:+.3f}  |S@f-U|={match:.1e}"
        )
    return out


def plot_real(rows_out: list[dict]) -> None:
    cams, faces, labels = [], [], []
    for r in rows_out:
        cams.append(
            np.array(
                [r["csv_CamX_mm"], r["csv_CamY_mm"], r["csv_CamZ_mm"]],
                dtype=float,
            )
            / 1000.0
        )
        faces.append(
            np.array(
                [float(r["face_real_x"]), float(r["face_real_y"]), float(r["face_real_z"])],
                dtype=float,
            )
        )
        labels.append(str(r["photo"]))
    cams = np.asarray(cams)
    faces = np.asarray(faces)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], c="k", s=80, depthshade=False, label="origin")
    ax.text(0, 0, 0, "  ORIGIN", fontsize=9)
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="#ff7f0e", s=50, depthshade=False, label="CAM")

    arrow_len = 0.025
    for i, (p, f, lab) in enumerate(zip(cams, faces, labels)):
        ax.plot([p[0], 0], [p[1], 0], [p[2], 0], color="0.7", ls="--", lw=0.8)
        ax.quiver(
            p[0], p[1], p[2],
            f[0] * arrow_len, f[1] * arrow_len, f[2] * arrow_len,
            color="#1f77b4", arrow_length_ratio=0.25, linewidth=1.5,
        )
        ax.text(p[0], p[1], p[2], f" {lab}", fontsize=8)

    # axes gizmo
    g = 0.02
    ax.quiver(0, 0, 0, g, 0, 0, color="r", arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, g, 0, color="g", arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, g, color="b", arrow_length_ratio=0.15)
    ax.text(g, 0, 0, "+X", color="r", fontsize=8)
    ax.text(0, g, 0, "+Y", color="g", fontsize=8)
    ax.text(0, 0, g, "+Z", color="b", fontsize=8)

    set_equal(ax, np.vstack([cams, np.zeros((1, 3))]), pad=0.015)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        "REAL RH — photo cams from coordinate2_filled.csv\n"
        "ExtXYZ: R=Rz(yaw)@Ry(pitch)@Rx(roll)   blue = cam face R@[0,0,1]\n"
        "dashed = cam → origin"
    )
    ax.text2D(0.02, 0.02, "RIGHT-HANDED (Real)", transform=ax.transAxes, color="green", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG_REAL, dpi=160)
    plt.close(fig)
    print(f"Wrote {PNG_REAL}")


def plot_real_vs_unity(rows_out: list[dict]) -> None:
    """
    Same LH display trick as tag regen_unity_cam2tag_face.py:
      matplotlib is RH; negate plotted X and relabel ticks with -v
      so the view reads as Unity left-handed.

    Left: REAL after S at origin  (S@t, S@face) — same relative geometry as Unity.
    Right: Unity absolute pose with LH display.
    Shapes should match (S already applied on the left).
    """
    cams_r, faces_r = [], []
    cams_u, faces_u = [], []
    labels, dots, match = [], [], []

    for r in rows_out:
        t = np.array(
            [r["csv_CamX_mm"], r["csv_CamY_mm"], r["csv_CamZ_mm"]],
            dtype=float,
        ) / 1000.0
        fr = np.array(
            [float(r["face_real_x"]), float(r["face_real_y"]), float(r["face_real_z"])],
            dtype=float,
        )
        tu = np.array(
            [float(r["unity_pos_x"]), float(r["unity_pos_y"]), float(r["unity_pos_z"])],
            dtype=float,
        )
        fu = np.array(
            [float(r["face_unity_x"]), float(r["face_unity_y"]), float(r["face_unity_z"])],
            dtype=float,
        )
        # comparable frame: apply S on real (Unity relative = S @ t)
        cams_r.append(S @ t)
        faces_r.append(S @ fr)
        cams_u.append(tu)
        faces_u.append(fu)
        labels.append(str(r["photo"]))
        dots.append(float(r["look_dot_face_to_origin"]))
        match.append(float(r["face_S_match_err"]))

    cams_r = np.asarray(cams_r)
    faces_r = np.asarray(faces_r)
    cams_u = np.asarray(cams_u)
    faces_u = np.asarray(faces_u)
    origin_u = ORIGINAL.copy()

    # relative Unity (for shape-match check vs left)
    cams_u_rel = cams_u - origin_u
    rel_err = float(np.max(np.linalg.norm(cams_r - cams_u_rel, axis=1)))

    fig = plt.figure(figsize=(16.5, 8.0))
    ax_r = fig.add_subplot(121, projection="3d")
    ax_u = fig.add_subplot(122, projection="3d")
    fwd_len = 0.025
    g = 0.04

    # ---- LEFT: REAL after S (Unity-axis numbers), origin@0 ----
    ax_r.scatter(0, 0, 0, c="k", s=100, depthshade=False, zorder=6)
    ax_r.text(0.002, 0.002, 0.002, " ORIGIN", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        ax_r.scatter(*cams_r[i], c="#ff7f0e", s=50, depthshade=False)
        ax_r.plot(
            [cams_r[i, 0], 0], [cams_r[i, 1], 0], [cams_r[i, 2], 0],
            color="0.5", ls="--", lw=1.0, alpha=0.7,
        )
        tip = cams_r[i] + fwd_len * faces_r[i]
        ax_r.quiver(
            cams_r[i, 0], cams_r[i, 1], cams_r[i, 2],
            tip[0] - cams_r[i, 0], tip[1] - cams_r[i, 1], tip[2] - cams_r[i, 2],
            color="#d62728", arrow_length_ratio=0.18, lw=1.8,
        )
        ax_r.text(*cams_r[i], f" {labels[i]}", fontsize=8)
    for v, col, name in [
        (np.array([g, 0, 0]), "#d62728", "+X"),
        (np.array([0, g, 0]), "#2ca02c", "+Y"),
        (np.array([0, 0, g]), "#1f77b4", "+Z"),
    ]:
        ax_r.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2.4)
        ax_r.text(*v, f" {name}", color=col, fontsize=9, fontweight="bold")
    ax_r.plot([], [], color="0.5", ls="--", label="cam -> origin")
    ax_r.plot([], [], color="#d62728", lw=2, label="face after S (= Unity face)")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.set_title(
        "REAL after S @ origin  (same numbers as Unity relative)\n"
        "S=diag(1,-1,1): real(x,y,z)->(x,-y,z)\n"
        f"red = S@face   mean face·toOrigin(raw)={np.mean(dots):+.3f}\n"
        f"|S@t - (t_unity-original)| max={rel_err:.2e}",
        fontsize=10,
    )
    ax_r.set_xlabel("X (Unity-axis)")
    ax_r.set_ylabel("Y (Unity-axis)")
    ax_r.set_zlabel("Z (Unity-axis)")
    set_equal(ax_r, np.vstack([np.zeros(3), cams_r]), pad=0.012)
    ax_r.view_init(elev=22, azim=-60)
    ax_r.text2D(
        0.02, 0.02, "REAL after S\n(compare shape →)",
        transform=ax_r.transAxes, fontsize=10, fontweight="bold", color="#2a9d8f",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    # ---- RIGHT: UNITY LH display (same method as tag plot) ----
    origin_d = to_lh_display(origin_u)
    cams_d = np.array([to_lh_display(p) for p in cams_u])
    faces_d = np.array([to_lh_display(v) for v in faces_u])
    # geometric cam -> origin in Unity
    to_o_u = origin_u - cams_u
    to_o_d = np.array([to_lh_display(v) for v in to_o_u])

    ax_u.scatter(*origin_d, c="k", s=100, depthshade=False, zorder=6)
    ax_u.text(
        origin_d[0] + 0.002, origin_d[1] + 0.002, origin_d[2] + 0.002,
        " ORIGIN", fontsize=10, fontweight="bold",
    )
    for i in range(len(labels)):
        ax_u.scatter(*cams_d[i], c="#17becf", s=50, depthshade=False)
        ax_u.plot(
            [cams_d[i, 0], cams_d[i, 0] + to_o_d[i, 0]],
            [cams_d[i, 1], cams_d[i, 1] + to_o_d[i, 1]],
            [cams_d[i, 2], cams_d[i, 2] + to_o_d[i, 2]],
            color="0.5", ls="--", lw=1.0, alpha=0.7,
        )
        tip = cams_d[i] + fwd_len * faces_d[i]
        ax_u.quiver(
            cams_d[i, 0], cams_d[i, 1], cams_d[i, 2],
            tip[0] - cams_d[i, 0], tip[1] - cams_d[i, 1], tip[2] - cams_d[i, 2],
            color="#1f77b4", arrow_length_ratio=0.18, lw=1.8,
        )
        ax_u.text(*cams_d[i], f" {labels[i]}", fontsize=8)

    # LH axis gizmo: plot Unity +X/+Y/+Z with to_lh_display (X flips in view)
    for v, col, name in [
        (np.array([g, 0.0, 0.0]), "#d62728", "+X"),
        (np.array([0.0, g, 0.0]), "#2ca02c", "+Y"),
        (np.array([0.0, 0.0, g]), "#1f77b4", "+Z"),
    ]:
        tip = origin_d + to_lh_display(v)
        ax_u.plot(
            [origin_d[0], tip[0]], [origin_d[1], tip[1]], [origin_d[2], tip[2]],
            color=col, lw=2.4,
        )
        ax_u.text(*tip, f" {name}", color=col, fontsize=9, fontweight="bold")

    ax_u.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{-v:.3f}"))
    ax_u.plot([], [], color="0.5", ls="--", label="cam -> origin")
    ax_u.plot([], [], color="#1f77b4", lw=2, label="face from R_u")
    ax_u.legend(loc="upper left", fontsize=8)
    ax_u.set_title(
        "UNITY left-handed view\n"
        "t_u=original+S@t   R_u=S@R@S   euler=ZXY\n"
        f"blue = camera face   |S@face_real - face_u| mean={np.mean(match):.2e}",
        fontsize=10,
    )
    ax_u.set_xlabel("Unity X (LH)")
    ax_u.set_ylabel("Unity Y")
    ax_u.set_zlabel("Unity Z")
    set_equal(ax_u, np.vstack([origin_d.reshape(1, 3), cams_d]), pad=0.012)
    ax_u.view_init(elev=22, azim=-60)
    ax_u.text2D(
        0.02, 0.02, "LEFT-HANDED (display)",
        transform=ax_u.transAxes, fontsize=10, fontweight="bold", color="#c44e52",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    fig.suptitle(
        "REAL photo cams vs Unity LH  |  CSV: camera_pose_unity_real_photos.csv\n"
        "Axis map: real(x,y,z)->unity(x,-y,z)   S=diag(1,-1,1)   "
        "original=(0.0316, -2.2834, 12.5992)\n"
        "Left already applies S so cam layout should match Unity relative pose on the right.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.05)
    fig.savefig(PNG_COMPARE, dpi=160)
    plt.close(fig)
    print(f"Wrote {PNG_COMPARE}")
    print(f"  shape match |S@t - (t_u-original)| max = {rel_err:.2e}")


def main() -> None:
    rows = load_rows()
    out = convert_all(rows)
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {CSV_OUT}")

    plot_real(out)
    plot_real_vs_unity(out)
    print("Done (real-photo pipeline; separate from tag scripts).")


if __name__ == "__main__":
    main()
