"""
Convert CamPitch / CamRoll / CamYaw (Extrinsic Pitch -> Roll -> Yaw)
to Unity eulerAngles (x, y, z) and write back into the CSV.

Method 2 — similarity transform for axis map (x, y, z) -> (x, -y, z):
    S = diag(1, -1, 1)
    R_unity = S * R_ext * S
    quaternion (x, y, z, w) -> (x, -y, z, -w)

    Quaternion ext = yawQ * rollQ * pitchQ;  // original CamPitch/Roll/Yaw
    Quaternion unity = new Quaternion(ext.x, -ext.y, ext.z, -ext.w);
    Vector3 unityEuler = unity.eulerAngles;
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

CSV_PATH = Path(__file__).with_name("coordinate2_filled.csv")

COL_PITCH = "CamPitch"
COL_ROLL = "CamRoll"
COL_YAW = "CamYaw"
COL_UX = "unity x roation"
COL_UY = "unity y rotation "
COL_UZ = "unity z rotation"


def q_angle_axis(angle_deg: float, axis: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Unity Quaternion.AngleAxis — returns (x, y, z, w)."""
    ax, ay, az = axis
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return (ax * s, ay * s, az * s, math.cos(half))


def q_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Quaternion multiply a * b (Unity order)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def q_to_unity_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """
    Match Unity Quaternion.eulerAngles (degrees, typically in [0, 360)).
    Based on Unity's Internal_ToEulerRad + MakePositive (ZXY convention).
    """
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return (0.0, 0.0, 0.0)
    x, y, z, w = x / n, y / n, z / n, w / n

    singularity = 0.499999
    sqx, sqy, sqz, sqw = x * x, y * y, z * z, w * w
    unit = sqx + sqy + sqz + sqw
    test = x * w - y * z

    if test > singularity * unit:
        pitch = math.pi / 2
        yaw = 2 * math.atan2(y, x)
        roll = 0.0
    elif test < -singularity * unit:
        pitch = -math.pi / 2
        yaw = -2 * math.atan2(y, x)
        roll = 0.0
    else:
        pitch = math.asin(2 * test / unit)
        yaw = math.atan2(2 * (w * y + z * x), sqw - sqx - sqy + sqz)
        roll = math.atan2(2 * (w * z + y * x), sqw - sqx + sqy - sqz)

    def make_positive(rad: float) -> float:
        deg = math.degrees(rad) % 360.0
        if deg < 0:
            deg += 360.0
        return deg

    # Unity Vector3: (x=pitch, y=yaw, z=roll) in eulerAngles naming
    return (make_positive(pitch), make_positive(yaw), make_positive(roll))


def q_axis_flip_x_neg_y_z(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """R' = S R S with S=diag(1,-1,1): (x,y,z,w) -> (x,-y,z,-w)."""
    x, y, z, w = q
    return (x, -y, z, -w)


def cam_pry_to_unity_euler(pitch: float, roll: float, yaw: float) -> tuple[float, float, float]:
    """Extrinsic Pitch -> Roll -> Yaw, then (x,-y,z) similarity on quaternion."""
    pitch_q = q_angle_axis(pitch, (1.0, 0.0, 0.0))  # Vector3.right
    roll_q = q_angle_axis(roll, (0.0, 0.0, 1.0))  # Vector3.forward
    yaw_q = q_angle_axis(yaw, (0.0, 1.0, 0.0))  # Vector3.up
    ext_q = q_mul(yaw_q, q_mul(roll_q, pitch_q))  # yaw * roll * pitch
    unity_q = q_axis_flip_x_neg_y_z(ext_q)
    return q_to_unity_euler(unity_q)


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in (COL_UX, COL_UY, COL_UZ):
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        pitch = float(row[COL_PITCH])
        roll = float(row[COL_ROLL])
        yaw = float(row[COL_YAW])
        ux, uy, uz = cam_pry_to_unity_euler(pitch, roll, yaw)
        row[COL_UX] = f"{ux:.6f}"
        row[COL_UY] = f"{uy:.6f}"
        row[COL_UZ] = f"{uz:.6f}"
        print(
            f"{row.get('photo#', '?'):>8}  "
            f"PRY=({pitch:.6f}, {roll:.6f}, {yaw:.6f})  "
            f"-> Unity XYZ=({ux:.6f}, {uy:.6f}, {uz:.6f})"
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {CSV_PATH}")


if __name__ == "__main__":
    main()
