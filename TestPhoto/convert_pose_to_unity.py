"""
CSV -> Unity pose

CSV meaning (per user):
  - Angles roll/pitch/yaw = TAG -> CAMERA  (Extrinsic Z->X->Y)
  - Vector xyz            = CAM -> TAG     (camera position in tag/world frame)

So invert ROTATION only; keep translation as-is:

  R_tag_to_cam = Extrinsic Ry(yaw) @ Rx(pitch) @ Rz(roll)
  R_cam_in_tag = R_tag_to_cam.T
  t_cam_in_tag = (csv_x, csv_y, csv_z) / 1000     # already cam2tag

Then World/tag -> Unity (X flipped):
  S = diag(-1, 1, 1)
  R_unity = S @ R_cam_in_tag @ S
  t_unity = original + S @ t_cam_in_tag

original = [0.03160001, -2.2834-0.045, 12.5992]
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

CSV_IN = Path(__file__).with_name("camera_pose_relative_to_tag.csv")
CSV_OUT = Path(__file__).with_name("camera_pose_unity_cam2tag.csv")

ORIGINAL = np.array([0.03160001, -2.2834 - 0.045, 12.5992], dtype=float)
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


def extrinsic_rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic Roll(Z)->Pitch(X)->Yaw(Y). Used for TAG->CAMERA rotation."""
    return Ry(yaw) @ Rx(pitch) @ Rz(roll)


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


def convert_one(roll: float, pitch: float, yaw: float, x_mm: float, y_mm: float, z_mm: float):
    # Angles: TAG -> CAMERA ; Vector: CAM -> TAG (camera position in tag frame)
    R_tag_to_cam = extrinsic_rpy_matrix(roll, pitch, yaw)
    t_cam_in_tag = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0

    # Invert rotation only
    R_cam_in_tag = R_tag_to_cam.T

    # TAG/world -> Unity
    R_unity = S @ R_cam_in_tag @ S
    R_facecam = Rx(180.0) @ R_unity  # camera +Z currently faces away from tag
    t_unity = ORIGINAL + (S @ t_cam_in_tag)

    e_cam_tag = quat_to_unity_euler(mat_to_quat(R_cam_in_tag))
    e_unity = quat_to_unity_euler(mat_to_quat(R_unity))
    e_facecam = quat_to_unity_euler(mat_to_quat(R_facecam))

    def look_dot_of(R):
        fwd = R @ np.array([0.0, 0.0, 1.0])
        to_tag = ORIGINAL - t_unity
        to_tag = to_tag / (float(np.linalg.norm(to_tag)) + 1e-12)
        return float(np.dot(fwd, to_tag))

    return {
        "t_cam_in_tag": t_cam_in_tag,
        "t_unity": t_unity,
        "R_tag_to_cam": R_tag_to_cam,
        "R_cam_in_tag": R_cam_in_tag,
        "R_unity": R_unity,
        "R_facecam": R_facecam,
        "e_cam_tag": e_cam_tag,
        "e_unity": e_unity,
        "e_facecam": e_facecam,
        "look_dot": look_dot_of(R_unity),
        "look_dot_facecam": look_dot_of(R_facecam),
    }


def main() -> None:
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    print("CSV angles = TAG->CAM (invert R); CSV xyz = CAM->TAG (use as-is).")
    print("Then Unity: S=diag(-1,1,1), original =", ORIGINAL.tolist())
    print()

    out = []
    dots = []
    dots_f = []
    for row in rows:
        roll = float(row["camera_roll_deg"])
        pitch = float(row["camera_pitch_deg"])
        yaw = float(row["camera_yaw_deg"])
        x_mm = float(row["camera_x_mm"])
        y_mm = float(row["camera_y_mm"])
        z_mm = float(row["camera_z_mm"])
        name = Path(row["image_file"]).name

        r = convert_one(roll, pitch, yaw, x_mm, y_mm, z_mm)
        dots.append(r["look_dot"])
        dots_f.append(r["look_dot_facecam"])
        tu = r["t_unity"]
        eu = r["e_unity"]
        ef = r["e_facecam"]
        tc = r["t_cam_in_tag"]

        out.append(
            {
                "image_file": name,
                "tag_id": row["tag_id"],
                "csv_x_mm": x_mm,
                "csv_y_mm": y_mm,
                "csv_z_mm": z_mm,
                "csv_roll_deg": roll,
                "csv_pitch_deg": pitch,
                "csv_yaw_deg": yaw,
                "cam_in_tag_x_m": f"{tc[0]:.9f}",
                "cam_in_tag_y_m": f"{tc[1]:.9f}",
                "cam_in_tag_z_m": f"{tc[2]:.9f}",
                "original_x": float(ORIGINAL[0]),
                "original_y": float(ORIGINAL[1]),
                "original_z": float(ORIGINAL[2]),
                "unity_pos_x": f"{tu[0]:.9f}",
                "unity_pos_y": f"{tu[1]:.9f}",
                "unity_pos_z": f"{tu[2]:.9f}",
                "unity_rot_x": f"{eu[0]:.6f}",
                "unity_rot_y": f"{eu[1]:.6f}",
                "unity_rot_z": f"{eu[2]:.6f}",
                "look_dot_to_tag": f"{r['look_dot']:.4f}",
                "unity_rot_facecam_x": f"{ef[0]:.6f}",
                "unity_rot_facecam_y": f"{ef[1]:.6f}",
                "unity_rot_facecam_z": f"{ef[2]:.6f}",
                "look_dot_facecam": f"{r['look_dot_facecam']:.4f}",
                "rot_mode": "invert(R tag->cam); t=cam2tag; S(-x,y,z); facecam=Rx180",
            }
        )
        print(
            f"{name:20s}  "
            f"pos=({tu[0]:.6f},{tu[1]:.6f},{tu[2]:.6f})  "
            f"rot=({eu[0]:.2f},{eu[1]:.2f},{eu[2]:.2f}) d={r['look_dot']:+.2f}  "
            f"facecam=({ef[0]:.2f},{ef[1]:.2f},{ef[2]:.2f}) d={r['look_dot_facecam']:+.2f}"
        )

    print(
        f"\nmean look_dot strict={sum(dots)/len(dots):+.3f}  "
        f"facecam={sum(dots_f)/len(dots_f):+.3f}"
    )

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
