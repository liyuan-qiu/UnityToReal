"""Build aligned Real-RGB / Unity-RGB / Unity-depth training triplets.

Fixed geometry pipeline:
1. Undistort Real NoTag RGB with the calibrated real-camera K and distortion.
2. Rotate Unity RGB and depth by 180 degrees into the Real image orientation.
3. Do not apply any Tag-derived image-space warp (sx/sy/tx/ty/pitch/yaw).

Unity FOV, lens shift, and camera pose are assumed to have been applied during
rendering. The exported depth is preserved as an 8-bit grayscale target; this
script does not claim that the shader output is metric depth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            ids.extend(range(int(first), int(last) + 1))
        else:
            ids.append(int(part))
    return ids


def read_required(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def depth_to_gray(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        return depth
    if depth.shape[2] >= 3:
        bgr = depth[:, :, :3]
        if np.array_equal(bgr[:, :, 0], bgr[:, :, 1]) and np.array_equal(
            bgr[:, :, 1], bgr[:, :, 2]
        ):
            return bgr[:, :, 0]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return depth[:, :, 0]


def make_preview(real: np.ndarray, unity: np.ndarray, depth: np.ndarray) -> np.ndarray:
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    blend = cv2.addWeighted(real, 0.5, unity, 0.5, 0.0)
    return cv2.hconcat([real, unity, blend, depth_color])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Real RGB / Unity RGB / Unity depth triplets without Tag 2D warp."
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=ROOT / "trainingData20260818_2",
    )
    parser.add_argument(
        "--unity-dir",
        type=Path,
        default=ROOT / "trainingData20260818_2/Unity20280818_2_8_notag",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=ROOT / "trainingData20260818_2/capsule_intrinsics.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "trainingData20260818_2/Depth_anything_training",
    )
    parser.add_argument("--ids", default="1-4")
    parser.add_argument("--horizontal-fov", type=float, default=71.25)
    parser.add_argument("--lens-shift", type=float, nargs=2, default=(0.164, -0.081))
    parser.add_argument(
        "--original",
        type=float,
        nargs=3,
        default=(-0.358725, -2.2282, 13.2305),
    )
    parser.add_argument(
        "--no-rotate-unity",
        action="store_true",
        help="Keep Unity RGB/depth orientation unchanged.",
    )
    args = parser.parse_args()

    real_dir = args.real_dir.resolve()
    unity_dir = args.unity_dir.resolve()
    intrinsics_path = args.intrinsics.resolve()
    out_dir = args.out_dir.resolve()
    rotate_unity = not args.no_rotate_unity

    calibration = np.load(intrinsics_path)
    K = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    dist = np.asarray(calibration["dist_coeffs"], dtype=np.float64).reshape(-1)

    output_dirs = {
        "real_rgb_raw": out_dir / "real_rgb_raw",
        "real_rgb": out_dir / "real_rgb",
        "unity_rgb": out_dir / "unity_rgb",
        "unity_depth": out_dir / "unity_depth",
        "preview": out_dir / "preview",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for photo_id in parse_ids(args.ids):
        sample = f"Photo{photo_id}"
        real_source = real_dir / f"{sample}_NoTag.jpg"
        unity_rgb_source = unity_dir / f"{sample}_tag_Unity.jpg"
        unity_depth_source = unity_dir / f"{sample}_tag_Depth.png"

        real = read_required(real_source, cv2.IMREAD_COLOR)
        unity = read_required(unity_rgb_source, cv2.IMREAD_COLOR)
        depth_raw = read_required(unity_depth_source, cv2.IMREAD_UNCHANGED)
        depth = depth_to_gray(depth_raw)

        real_undistorted = cv2.undistort(real, K, dist, None, K)
        if rotate_unity:
            unity = cv2.rotate(unity, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)

        if real_undistorted.shape[:2] != unity.shape[:2] or unity.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"{sample}: shape mismatch: Real={real_undistorted.shape}, "
                f"UnityRGB={unity.shape}, UnityDepth={depth.shape}"
            )
        if depth.dtype != np.uint8:
            raise ValueError(f"{sample}: expected uint8 Unity depth, found {depth.dtype}")

        filename = f"{sample}.png"
        real_raw_output = output_dirs["real_rgb_raw"] / filename
        real_output = output_dirs["real_rgb"] / filename
        unity_output = output_dirs["unity_rgb"] / filename
        depth_output = output_dirs["unity_depth"] / filename
        preview_output = output_dirs["preview"] / filename

        if not cv2.imwrite(str(real_raw_output), real):
            raise OSError(f"Could not write {real_raw_output}")
        if not cv2.imwrite(str(real_output), real_undistorted):
            raise OSError(f"Could not write {real_output}")
        if not cv2.imwrite(str(unity_output), unity):
            raise OSError(f"Could not write {unity_output}")
        if not cv2.imwrite(str(depth_output), depth):
            raise OSError(f"Could not write {depth_output}")
        if not cv2.imwrite(str(preview_output), make_preview(real_undistorted, unity, depth)):
            raise OSError(f"Could not write {preview_output}")

        rows.append(
            {
                "id": photo_id,
                "sample": sample,
                "real_rgb_raw": f"real_rgb_raw/{filename}",
                "real_rgb": f"real_rgb/{filename}",
                "unity_rgb": f"unity_rgb/{filename}",
                "unity_depth": f"unity_depth/{filename}",
                "preview": f"preview/{filename}",
                "width": real_undistorted.shape[1],
                "height": real_undistorted.shape[0],
                "depth_dtype": str(depth.dtype),
                "depth_min_raw": int(depth.min()),
                "depth_max_raw": int(depth.max()),
                "real_source": real_source.relative_to(ROOT).as_posix(),
                "unity_rgb_source": unity_rgb_source.relative_to(ROOT).as_posix(),
                "unity_depth_source": unity_depth_source.relative_to(ROOT).as_posix(),
            }
        )
        print(f"[{sample}] wrote RGB/RGB/depth triplet {real_undistorted.shape[1]}x{real_undistorted.shape[0]}")

    if not rows:
        raise RuntimeError("No samples were generated.")

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "pipeline": [
            "Real RGB: cv2.undistort(source, K_real, dist_real, newCameraMatrix=K_real)",
            "Unity RGB: rotate 180 degrees",
            "Unity depth: rotate 180 degrees and keep grayscale uint8 values",
            "No Tag-derived sx/sy/tx/ty/rotation/pitch/yaw image warp",
        ],
        "sample_count": len(rows),
        "real_intrinsics_source": intrinsics_path.relative_to(ROOT).as_posix(),
        "K_real": K.tolist(),
        "dist_real": dist.tolist(),
        "unity_render_settings": {
            "gate_fit": "Horizontal",
            "horizontal_fov_deg": args.horizontal_fov,
            "lens_shift_xy": list(args.lens_shift),
            "original_xyz": list(args.original),
            "note": "Applied in Unity before rendering; not applied again by this script.",
        },
        "unity_rotated_180": rotate_unity,
        "depth": {
            "storage": "uint8 grayscale PNG copied from Unity shader values",
            "metric": False,
            "warning": "Do not interpret as meters without confirming the Unity depth shader encoding/range.",
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        """# Depth-Anything training pairs

Each sample uses the same basename in three directories:

- `real_rgb/PhotoN.png`: undistorted Real NoTag RGB (training input)
- `real_rgb_raw/PhotoN.png`: original Real NoTag RGB before undistortion
- `unity_rgb/PhotoN.png`: Unity NoTag RGB, rotated 180 degrees (visual check)
- `unity_depth/PhotoN.png`: matching Unity depth, rotated 180 degrees (target)
- `preview/PhotoN.png`: Real | Unity | 50% blend | depth color preview

No Tag-derived 2D warp is applied. Unity FOV, lens shift, and camera pose are
already baked into the Unity render.

The current depth PNG is an 8-bit shader output, not confirmed metric depth.
See `manifest.csv` and `metadata.json`.
""",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
