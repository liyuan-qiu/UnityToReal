"""Undistort Real images, then center-crop / resize to 1080x720.

Same convention as Depth-Anything training for this pack:
  und = cv2.undistort(img, K, dist, None, K)
  then force output size to 1080x720.

Default intrinsics: capsule_intrinsics.npz next to this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_INTRINSICS = HERE / "capsule_intrinsics.npz"
TARGET_W, TARGET_H = 1080, 720
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Undistort images with capsule_intrinsics.npz, then crop/resize to 1080x720."
    )
    p.add_argument("input", type=Path, help="Input image file or directory.")
    p.add_argument(
        "output",
        type=Path,
        help="Output file (if input is a file) or output directory (if input is a directory).",
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
        help=f"Calibration NPZ (default: {DEFAULT_INTRINSICS.name} in this folder).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, also process images in subdirectories.",
    )
    p.add_argument(
        "--keep-name",
        action="store_true",
        help="Keep original filename; default directory mode writes <stem>_undist_1080x720.png",
    )
    return p.parse_args()


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"Intrinsics not found: {path}")
    data = np.load(path)
    try:
        K = np.asarray(data["camera_matrix"], np.float64)
        dist = np.asarray(data["dist_coeffs"], np.float64).reshape(-1)
    except KeyError as exc:
        raise SystemExit(f"NPZ missing camera_matrix or dist_coeffs: {path}") from exc
    return K, dist


def to_1080x720(image: np.ndarray) -> np.ndarray:
    """Scale to cover 1080x720, then center crop; pad/resize if needed."""
    h, w = image.shape[:2]
    if w == TARGET_W and h == TARGET_H:
        return image

    scale = max(TARGET_W / w, TARGET_H / h)
    if abs(scale - 1.0) > 1e-6:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        h, w = image.shape[:2]

    x0 = max(0, (w - TARGET_W) // 2)
    y0 = max(0, (h - TARGET_H) // 2)
    crop = image[y0 : y0 + TARGET_H, x0 : x0 + TARGET_W]
    if crop.shape[0] != TARGET_H or crop.shape[1] != TARGET_W:
        crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return crop


def undistort_to_target(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    und = cv2.undistort(image, K, dist, None, K)
    return to_1080x720(und)


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Unsupported image type: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input does not exist: {input_path}")
    paths = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def output_path_for(
    src: Path,
    input_root: Path,
    output_arg: Path,
    input_is_file: bool,
    keep_name: bool,
) -> Path:
    if input_is_file:
        out = output_arg
        if out.suffix == "":
            out = out / f"{src.stem}_undist_1080x720.png"
        return out

    relative = src.relative_to(input_root)
    if keep_name:
        name = relative.name
        if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            name = f"{relative.stem}.png"
    else:
        name = f"{relative.stem}_undist_1080x720.png"
    return output_arg / relative.parent / name


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_arg = args.output.resolve()
    K, dist = load_calibration(args.intrinsics.resolve())

    images = collect_images(input_path, args.recursive)
    if not images:
        raise SystemExit(f"No images found under: {input_path}")

    input_is_file = input_path.is_file()
    input_root = input_path.parent if input_is_file else input_path
    ok, skipped = 0, 0

    print(f"intrinsics = {args.intrinsics.resolve()}")
    print(f"target size = {TARGET_W}x{TARGET_H}")
    print(f"images = {len(images)}")

    for src in images:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable: {src}")
            skipped += 1
            continue
        out = undistort_to_target(img, K, dist)
        dest = output_path_for(src, input_root, output_arg, input_is_file, args.keep_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), out):
            print(f"skip write failed: {dest}")
            skipped += 1
            continue
        ok += 1
        print(f"ok {src.name} -> {dest}  ({out.shape[1]}x{out.shape[0]})")

    print(f"done: wrote {ok}, skipped {skipped}")


if __name__ == "__main__":
    main()
