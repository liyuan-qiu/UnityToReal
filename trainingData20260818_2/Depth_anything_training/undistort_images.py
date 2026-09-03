"""Batch-undistort JPEG images using capsule_intrinsics.npz.

Optional post-undistort center scale (FOV / tag-compare residual):
  --scale 1.26          isotropic
  --scale-x / --scale-y anisotropic (e.g. tag mean Unity sx,sy -> use 1/sx, 1/sy on real)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INTRINSICS = Path(__file__).resolve().parent / "capsule_intrinsics.npz"
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Undistort all JPEG images in a directory using a saved capsule calibration."
    )
    parser.add_argument("input_directory", type=Path, help="Directory containing source JPEG images.")
    parser.add_argument("output_directory", type=Path, help="Directory for undistorted JPEG images.")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
        help=f"Calibration .npz file (default: {DEFAULT_INTRINSICS.name}).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process JPEG images in input subdirectories.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Isotropic center scale after undistort (e.g. 1.26). Overridden by --scale-x/y.",
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=None,
        help="Horizontal center scale after undistort (keeps output size).",
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=None,
        help="Vertical center scale after undistort (keeps output size).",
    )
    return parser.parse_args()


def find_jpeg_images(input_directory, recursive):
    paths = input_directory.rglob("*") if recursive else input_directory.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS)


def center_scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale about image center; crop/pad back to original HxW."""
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    ms1, ns1 = ms0 + (xs1 - xs0), ns0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0:ns1, ms0:ms1]
    return canvas


def main():
    args = parse_arguments()
    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    intrinsics_path = args.intrinsics.resolve()

    if not input_directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_directory}")
    if not intrinsics_path.is_file():
        raise SystemExit(f"Intrinsic calibration file does not exist: {intrinsics_path}")
    if input_directory == output_directory:
        raise SystemExit("Output directory must be different from the input directory.")

    try:
        calibration = np.load(intrinsics_path)
        camera_matrix = calibration["camera_matrix"]
        dist_coeffs = calibration["dist_coeffs"]
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load calibration parameters: {error}") from error

    if args.scale_x is not None or args.scale_y is not None:
        sx = float(args.scale_x if args.scale_x is not None else (args.scale or 1.0))
        sy = float(args.scale_y if args.scale_y is not None else (args.scale or 1.0))
    elif args.scale is not None:
        sx = sy = float(args.scale)
    else:
        sx = sy = 1.0

    image_paths = find_jpeg_images(input_directory, args.recursive)
    if not image_paths:
        print(f"No JPEG images found in: {input_directory}")
        return

    print(f"post-undistort scale sx,sy=({sx:.4f},{sy:.4f})")

    processed_count = 0
    skipped_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            skipped_count += 1
            continue

        relative_path = image_path.relative_to(input_directory)
        output_path = output_directory / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        undistorted_image = center_scale(undistorted_image, sx, sy)

        if not cv2.imwrite(str(output_path), undistorted_image):
            print(f"Skipped unwritable output: {output_path}")
            skipped_count += 1
            continue

        processed_count += 1
        print(f"Undistorted: {relative_path}")

    print(f"Completed: {processed_count} image(s) written to {output_directory}")
    if skipped_count:
        print(f"Skipped: {skipped_count} image(s)")


if __name__ == "__main__":
    main()
