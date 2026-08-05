"""
Compare CSV unity_quat_* vs Unity-exported camera quaternions.

Usage:
  python compare_unity_quaternions.py
  python compare_unity_quaternions.py --export recording/camera_pose_unity_cam2tag_face/unity_camera_quat_export.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

CSV_CALC = Path(__file__).with_name("camera_pose_unity_cam2tag_face.csv")
DEFAULT_EXPORT = Path(__file__).with_name("recording") / "camera_pose_unity_cam2tag_face" / "unity_camera_quat_export.csv"


def qdot_abs(a, b) -> float:
    return abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])


def angle_deg(dot_abs: float) -> float:
    d = min(1.0, max(0.0, dot_abs))
    return math.degrees(2.0 * math.acos(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", type=Path, default=CSV_CALC)
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = ap.parse_args()

    if not args.calc.is_file():
        raise SystemExit(f"Missing calc CSV: {args.calc}")
    if not args.export.is_file():
        raise SystemExit(
            f"Missing Unity export: {args.export}\n"
            "Run PoseCsvAutoCapture in Unity first; it writes unity_camera_quat_export.csv"
        )

    calc = {
        Path(r["image_file"]).name: (
            float(r["unity_quat_x"]),
            float(r["unity_quat_y"]),
            float(r["unity_quat_z"]),
            float(r["unity_quat_w"]),
        )
        for r in csv.DictReader(args.calc.open(encoding="utf-8-sig"))
    }
    exp_rows = list(csv.DictReader(args.export.open(encoding="utf-8-sig")))

    print(f"calc:   {args.calc}")
    print(f"export: {args.export}")
    print()
    print(f"{'image':22s}  {'|q·q|':>8s}  {'ang(deg)':>8s}  export_quat")
    dots = []
    for r in exp_rows:
        name = Path(r["image_file"]).name
        if name not in calc:
            print(f"{name:22s}  MISSING in calc CSV")
            continue
        qe = (
            float(r["unity_quat_x"]),
            float(r["unity_quat_y"]),
            float(r["unity_quat_z"]),
            float(r["unity_quat_w"]),
        )
        # prefer export's live columns if present
        if "unity_quat_x" in r and "csv_quat_x" in r:
            qe = (
                float(r["unity_quat_x"]),
                float(r["unity_quat_y"]),
                float(r["unity_quat_z"]),
                float(r["unity_quat_w"]),
            )
        qc = calc[name]
        if qc[3] < 0:
            qc = (-qc[0], -qc[1], -qc[2], -qc[3])
        if qe[3] < 0:
            qe = (-qe[0], -qe[1], -qe[2], -qe[3])
        d = qdot_abs(qc, qe)
        dots.append(d)
        print(
            f"{name:22s}  {d:8.6f}  {angle_deg(d):8.4f}  "
            f"({qe[0]:+.5f},{qe[1]:+.5f},{qe[2]:+.5f},{qe[3]:+.5f})"
        )

    if dots:
        print()
        print(f"mean |q·q| = {sum(dots)/len(dots):.6f}   mean ang = {sum(angle_deg(d) for d in dots)/len(dots):.4f} deg")
        print("OK if |q·q| ≈ 1 (q and -q are the same rotation).")


if __name__ == "__main__":
    main()
