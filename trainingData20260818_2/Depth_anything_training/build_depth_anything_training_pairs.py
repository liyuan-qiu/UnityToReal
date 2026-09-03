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
import math
from pathlib import Path

import cv2
import numpy as np


# This copy lives inside Depth_anything_training/; dataset root is the parent folder.
PACK_DIR = Path(__file__).resolve().parent
ROOT = PACK_DIR.parent
REPO_ROOT = PACK_DIR.parents[1]


def _relpath(path: Path) -> str:
    path = path.resolve()
    for base in (PACK_DIR, ROOT, REPO_ROOT):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


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


def load_tag_warps(path: Path) -> dict[int, dict[str, float | str]]:
    warps: dict[int, dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            rmse = float(row["id_rmse_px"])
            rmse_py = float(row["id_rmse_pitchyaw_px"])
            if rmse_py <= rmse:
                warps[int(row["id"])] = {
                    "mode": "pitch_yaw_then_sxsy",
                    "pitch": float(row["align_pitch_deg"]),
                    "yaw": float(row["align_yaw_deg"]),
                    "sx": float(row["align_sx_py"]),
                    "sy": float(row["align_sy_py"]),
                    "tx": float(row["align_tx_py"]),
                    "ty": float(row["align_ty_py"]),
                    "tag_rmse_px": rmse_py,
                }
            else:
                warps[int(row["id"])] = {
                    "mode": "sxsy_only",
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sx": float(row["align_sx"]),
                    "sy": float(row["align_sy"]),
                    "tx": float(row["align_tx"]),
                    "ty": float(row["align_ty"]),
                    "tag_rmse_px": rmse,
                }
    return warps


def rotation_homography(K: np.ndarray, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return K @ (Ry @ Rx) @ np.linalg.inv(K)


def apply_tag_warp(
    image: np.ndarray,
    K: np.ndarray,
    warp: dict[str, float | str],
    interpolation: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    H = rotation_homography(K, float(warp["pitch"]), float(warp["yaw"]))
    perspective = cv2.warpPerspective(
        image, H, (width, height), flags=interpolation, borderValue=0
    )
    M = np.array(
        [
            [float(warp["sx"]), 0.0, float(warp["tx"])],
            [0.0, float(warp["sy"]), float(warp["ty"])],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        perspective, M, (width, height), flags=interpolation, borderValue=0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Real RGB / Unity RGB / Unity depth triplets without Tag 2D warp."
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--unity-dir",
        type=Path,
        default=ROOT / "Unity20280818_2_8_notag",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=PACK_DIR / "capsule_intrinsics.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACK_DIR,
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
    parser.add_argument(
        "--warp-csv",
        type=Path,
        default=None,
        help="Optional tagged compare_summary.csv; applies its per-photo warp to Unity RGB/depth.",
    )
    args = parser.parse_args()

    real_dir = args.real_dir.resolve()
    unity_dir = args.unity_dir.resolve()
    intrinsics_path = args.intrinsics.resolve()
    out_dir = args.out_dir.resolve()
    rotate_unity = not args.no_rotate_unity
    warp_csv = args.warp_csv.resolve() if args.warp_csv is not None else None
    tag_warps = load_tag_warps(warp_csv) if warp_csv is not None else {}

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

        warp = tag_warps.get(photo_id)
        if warp_csv is not None and warp is None:
            raise KeyError(f"{sample}: no Tag warp found in {warp_csv}")
        if warp is not None:
            unity = apply_tag_warp(unity, K, warp, cv2.INTER_LINEAR)
            depth = apply_tag_warp(depth, K, warp, cv2.INTER_NEAREST)

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
                "warp_mode": str(warp["mode"]) if warp is not None else "none",
                "warp_pitch_deg": float(warp["pitch"]) if warp is not None else 0.0,
                "warp_yaw_deg": float(warp["yaw"]) if warp is not None else 0.0,
                "warp_sx": float(warp["sx"]) if warp is not None else 1.0,
                "warp_sy": float(warp["sy"]) if warp is not None else 1.0,
                "warp_tx": float(warp["tx"]) if warp is not None else 0.0,
                "warp_ty": float(warp["ty"]) if warp is not None else 0.0,
                "tag_rmse_px": float(warp["tag_rmse_px"]) if warp is not None else "",
                "real_source": _relpath(real_source),
                "unity_rgb_source": _relpath(unity_rgb_source),
                "unity_depth_source": _relpath(unity_depth_source),
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
            (
                "Apply the matching Tag-derived warp to Unity RGB and depth"
                if warp_csv is not None
                else "No Tag-derived sx/sy/tx/ty/rotation/pitch/yaw image warp"
            ),
        ],
        "sample_count": len(rows),
        "real_intrinsics_source": _relpath(intrinsics_path),
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
        "tag_warp": {
            "applied": warp_csv is not None,
            "source": _relpath(warp_csv) if warp_csv is not None else None,
            "order": "pitch/yaw homography, then independent sx/sy and tx/ty",
            "depth_interpolation": "nearest",
        },
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
    warp_description = (
        "A per-photo 2D warp estimated from the matching Tag pair is applied "
        "to both Unity RGB and Unity depth."
        if warp_csv is not None
        else "No Tag-derived 2D warp is applied."
    )
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        (out_dir / "README.md").write_text(
            f"""# Depth-Anything training pairs

Each sample uses the same basename in three directories:

- `real_rgb/PhotoN.png`: undistorted Real NoTag RGB (training input)
- `real_rgb_raw/PhotoN.png`: original Real NoTag RGB before undistortion
- `unity_rgb/PhotoN.png`: Unity NoTag RGB, rotated 180 degrees (visual check)
- `unity_depth/PhotoN.png`: matching Unity depth, rotated 180 degrees (target)
- `preview/PhotoN.png`: Real | Unity | 50% blend | depth color preview

{warp_description} Unity FOV, lens shift, and camera pose are already baked
into the Unity render.

The current depth PNG is an 8-bit shader output, not confirmed metric depth.
See `manifest.csv` and `metadata.json`.
""",
            encoding="utf-8",
        )
    else:
        print(f"Kept existing README: {readme_path}")
    print(f"Wrote {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
