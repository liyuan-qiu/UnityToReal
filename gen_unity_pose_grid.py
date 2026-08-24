"""
Generate a Unity pose CSV for PoseCsvAutoCapture.cs: scan camera position, keep one photo's orientation.

Default Angle1 = trainingData20260818_2 Photo1 rotation.
User X/Y ranges sit around Photo1's Unity Y/X (not Unity's axis names), so:
  user X [-2.15, -2.33] -> unity_pos_y
  user Y [-0.32, -0.40] -> unity_pos_x
  user Z [13.23, 13.26] -> unity_pos_z
Pass --literal-xy to write user X/Y straight into unity_pos_x/y instead.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POSE_CSV = ROOT / "trainingData20260818_2" / "camera_pose_unity_cam2tag_face_20260818_4.csv"

HEADER = [
    "image_file",
    "tag_id",
    "csv_x_mm",
    "csv_y_mm",
    "csv_z_mm",
    "csv_roll_deg",
    "csv_pitch_deg",
    "csv_yaw_deg",
    "original_x",
    "original_y",
    "original_z",
    "unity_pos_x",
    "unity_pos_y",
    "unity_pos_z",
    "unity_cam2tag_x",
    "unity_cam2tag_y",
    "unity_cam2tag_z",
    "unity_rot_x",
    "unity_rot_y",
    "unity_rot_z",
    "unity_quat_x",
    "unity_quat_y",
    "unity_quat_z",
    "unity_quat_w",
    "look_dot_real",
    "look_dot_unity",
    "rot_mode",
    "grid_ix",
    "grid_iy",
    "grid_iz",
]


def inclusive_range(start: float, end: float, step: float) -> list[float]:
    step = abs(float(step))
    sign = 1.0 if end >= start else -1.0
    n = int(round(abs(end - start) / step)) + 1
    vals = [round(start + sign * i * step, 6) for i in range(n)]
    vals[-1] = round(end, 6)
    return vals


def load_photo_pose(path: Path, photo_id: int) -> dict[str, str]:
    name = f"Photo{photo_id}_tag.jpg"
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["image_file"].strip() == name:
                return row
    raise SystemExit(f"{name} not found in {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--photo", type=int, default=1, help="trainingData20260818_2 Photo id for angles")
    p.add_argument("--out-dir", type=Path, default=ROOT / "recordings" / "Angle1")
    p.add_argument("--x0", type=float, default=-2.15, help="user X start (Unity Y unless --literal-xy)")
    p.add_argument("--x1", type=float, default=-2.33, help="user X end")
    p.add_argument("--y0", type=float, default=-0.32, help="user Y start (Unity X unless --literal-xy)")
    p.add_argument("--y1", type=float, default=-0.40, help="user Y end")
    p.add_argument("--z0", type=float, default=13.23)
    p.add_argument("--z1", type=float, default=13.26)
    p.add_argument("--dx", type=float, default=0.005)
    p.add_argument("--dy", type=float, default=0.005)
    p.add_argument("--dz", type=float, default=0.01)
    p.add_argument(
        "--literal-xy",
        action="store_true",
        help="Write user X->unity_pos_x and user Y->unity_pos_y (not Photo1 axis match)",
    )
    p.add_argument("--pose-csv", type=Path, default=POSE_CSV)
    args = p.parse_args()

    src = load_photo_pose(args.pose_csv, args.photo)
    xs = inclusive_range(args.x0, args.x1, args.dx)
    ys = inclusive_range(args.y0, args.y1, args.dy)
    zs = inclusive_range(args.z0, args.z1, args.dz)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "camera_pose_unity_cam2tag_face.csv"
    index_path = out_dir / "grid_index.csv"

    prefix = out_dir.name  # Angle1
    rows = []
    n = 0
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                n += 1
                if args.literal_xy:
                    px, py = x, y
                else:
                    px, py = y, x  # user Y -> Unity X, user X -> Unity Y
                stem = f"{prefix}_{n:04d}"
                rows.append(
                    {
                        "image_file": f"{stem}.jpg",
                        "tag_id": f"{prefix}_photo{args.photo}_grid",
                        "csv_x_mm": src["csv_x_mm"],
                        "csv_y_mm": src["csv_y_mm"],
                        "csv_z_mm": src["csv_z_mm"],
                        "csv_roll_deg": src["csv_roll_deg"],
                        "csv_pitch_deg": src["csv_pitch_deg"],
                        "csv_yaw_deg": src["csv_yaw_deg"],
                        "original_x": src["original_x"],
                        "original_y": src["original_y"],
                        "original_z": src["original_z"],
                        "unity_pos_x": f"{px:.6f}",
                        "unity_pos_y": f"{py:.6f}",
                        "unity_pos_z": f"{z:.6f}",
                        "unity_cam2tag_x": "0",
                        "unity_cam2tag_y": "0",
                        "unity_cam2tag_z": "0",
                        "unity_rot_x": src["unity_rot_x"],
                        "unity_rot_y": src["unity_rot_y"],
                        "unity_rot_z": src["unity_rot_z"],
                        "unity_quat_x": src["unity_quat_x"],
                        "unity_quat_y": src["unity_quat_y"],
                        "unity_quat_z": src["unity_quat_z"],
                        "unity_quat_w": src["unity_quat_w"],
                        "look_dot_real": src["look_dot_real"],
                        "look_dot_unity": src["look_dot_unity"],
                        "rot_mode": (
                            f"GRID pos only; rot copied from {src['image_file']}; "
                            "edit unity_rot_*/unity_quat_* by hand if needed"
                        ),
                        "grid_ix": str(ix),
                        "grid_iy": str(iy),
                        "grid_iz": str(iz),
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image_file",
                "unity_pos_x",
                "unity_pos_y",
                "unity_pos_z",
                "user_x",
                "user_y",
                "user_z",
                "grid_ix",
                "grid_iy",
                "grid_iz",
            ],
        )
        w.writeheader()
        k = 0
        for iz, z in enumerate(zs):
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    k += 1
                    px, py = (x, y) if args.literal_xy else (y, x)
                    w.writerow(
                        {
                            "image_file": f"{prefix}_{k:04d}.jpg",
                            "unity_pos_x": f"{px:.6f}",
                            "unity_pos_y": f"{py:.6f}",
                            "unity_pos_z": f"{z:.6f}",
                            "user_x": f"{x:.6f}",
                            "user_y": f"{y:.6f}",
                            "user_z": f"{z:.6f}",
                            "grid_ix": ix,
                            "grid_iy": iy,
                            "grid_iz": iz,
                        }
                    )

    print(f"photo{args.photo} rot/quat from {args.pose_csv.name}")
    print(
        f"  unity_rot=({src['unity_rot_x']}, {src['unity_rot_y']}, {src['unity_rot_z']})"
    )
    print(
        f"  unity_quat=({src['unity_quat_x']}, {src['unity_quat_y']}, "
        f"{src['unity_quat_z']}, {src['unity_quat_w']})"
    )
    print(f"user X n={len(xs)}  {xs[0]} .. {xs[-1]} step {args.dx}")
    print(f"user Y n={len(ys)}  {ys[0]} .. {ys[-1]} step {args.dy}")
    print(f"user Z n={len(zs)}  {zs[0]} .. {zs[-1]} step {args.dz}")
    print(f"literal_xy={args.literal_xy}  poses={len(rows)}")
    print(f"wrote {csv_path}")
    print(f"wrote {index_path}")
    print(
        "PoseCsvAutoCapture: set csvRelativePath to this CSV, "
        f"outputFolderName={prefix} (Unity still saves under recording/)"
    )


if __name__ == "__main__":
    main()
