"""
Export aligned training pairs (method 1):

  real  = undistort(samplePhoto2) -> center_scale(tag FOV scale)
  unity = unitySamplePhoto3 RGB + Depth, same XY pixel shift (+3,+1 mm @ depth)
  crop  = common overlap of valid real & unity regions (fixed size across set)

Does NOT rotate 180. Does NOT apply extra NCC warp (deterministic geometry).

Outputs under ../compare_out/depth_train_pairs_<unity-dir>/ :
  {id}_real.png   {id}_unity.png   {id}_depth.png   {id}_mask.png
  preview_{id}.jpg
  manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

K = np.array(
    [
        [762.7627033, 0.0, 661.53817354],
        [0.0, 763.78023472, 360.37587777],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST = np.array(
    [-0.38898088, 0.15099531, -0.00301529, 0.00057045, -0.02746219],
    dtype=np.float64,
)
FX, FY = float(K[0, 0]), float(K[1, 1])

# Defaults from tag FOV residual + XY sweep
DEFAULT_SCALE_X = 1.2935
DEFAULT_SCALE_Y = 1.2264
DEFAULT_DX_MM = 3.0
DEFAULT_DY_MM = 1.0

IDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Baseline often missing on real
IMG_W, IMG_H = 1080, 720


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export real und + Unity RGB/Depth aligned crops for depth training.")
    p.add_argument("--real-dir", type=Path, default=HERE, help="Folder with {id}.jpg real photos")
    p.add_argument("--unity-dir", type=str, default="unitySamplePhoto3", help="Unity capture folder under project root")
    p.add_argument("--pose-csv", type=Path, default=HERE / "camera_pose_unity_real_photos.csv")
    p.add_argument("--out-name", type=str, default="", help="compare_out/<out-name>/ ; default depth_train_pairs_<unity>")
    p.add_argument("--scale-x", type=float, default=DEFAULT_SCALE_X)
    p.add_argument("--scale-y", type=float, default=DEFAULT_SCALE_Y)
    p.add_argument("--dx-mm", type=float, default=DEFAULT_DX_MM, help="Unity cam +X offset (mm) -> image shift")
    p.add_argument("--dy-mm", type=float, default=DEFAULT_DY_MM, help="Unity cam +Y offset (mm) -> image shift")
    p.add_argument("--erode", type=int, default=2, help="Erode valid mask (px) to drop border junk")
    p.add_argument(
        "--crop-mode",
        choices=("fixed_min", "per_image", "resize"),
        default="fixed_min",
        help="fixed_min: one size=min overlap for all; per_image: each bbox; resize: each bbox -> --out-w/--out-h",
    )
    p.add_argument("--out-w", type=int, default=0, help="With crop-mode=resize, output width")
    p.add_argument("--out-h", type=int, default=0, help="With crop-mode=resize, output height")
    p.add_argument("--ids", type=str, default=",".join(IDS), help="Comma-separated ids, e.g. 1,2,3")
    return p.parse_args()


def undistort_bgr(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, DIST, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, K, DIST, None, new_K)


def center_scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0 : ns0 + (ys1 - ys0), ms0 : ms0 + (xs1 - xs0)]
    return canvas


def shift_img(img: np.ndarray, dx: int, dy: int, nearest: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mm_to_px(dX_mm: float, dY_mm: float, depth_m: float) -> tuple[int, int]:
    z = max(depth_m, 1e-3)
    dx = int(round(dX_mm * FX / (z * 1000.0)))
    dy = int(round(-dY_mm * FY / (z * 1000.0)))
    return dx, dy


def load_depths_m(pose_csv: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not pose_csv.is_file():
        return out
    for row in csv.DictReader(pose_csv.open(encoding="utf-8-sig")):
        stem = Path(row.get("image_file", row.get("photo", ""))).stem
        x = float(row["csv_CamX_mm"]) / 1000.0
        y = float(row["csv_CamY_mm"]) / 1000.0
        z = float(row["csv_CamZ_mm"]) / 1000.0
        out[stem] = float(math.sqrt(x * x + y * y + z * z))
    return out


def valid_mask_bgr(img: np.ndarray, thr: int = 8) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    m = (g > thr).astype(np.uint8) * 255
    return m


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def center_crop_to_size(img: np.ndarray, box: tuple[int, int, int, int], tw: int, th: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    xa = max(0, min(img.shape[1] - tw, cx - tw // 2))
    ya = max(0, min(img.shape[0] - th, cy - th // 2))
    return img[ya : ya + th, xa : xa + tw]


def make_preview(real: np.ndarray, unity: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    def rs(im, w=320):
        h = max(1, int(round(im.shape[0] * w / im.shape[1])))
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    if depth.ndim == 2:
        d3 = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    elif depth.shape[2] == 4:
        d3 = depth[:, :, :3]
    else:
        d3 = depth
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tiles = [rs(real), rs(unity), rs(d3), rs(m3)]
    # pad to same h
    hmax = max(t.shape[0] for t in tiles)
    pads = []
    for t in tiles:
        if t.shape[0] < hmax:
            t = cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pads.append(t)
    return np.hstack(pads)


def find_real(real_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = real_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_unity_rgb(unity_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_Unity.jpg", f"{stem}_Unity.png", f"{stem}.jpg"):
        p = unity_dir / name
        if p.exists():
            return p
    return None


def find_unity_depth(unity_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_Depth.png", f"{stem}_depth.png", f"{stem}_Depth.jpg"):
        p = unity_dir / name
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    unity_dir = (ROOT / args.unity_dir).resolve()
    real_dir = args.real_dir.resolve()
    pose_csv = args.pose_csv.resolve()
    out_name = args.out_name.strip() or f"depth_train_pairs_{unity_dir.name}"
    out_dir = ROOT / "compare_out" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    depths = load_depths_m(pose_csv)

    print(f"REAL : {real_dir}")
    print(f"UNITY: {unity_dir}")
    print(f"OUT  : {out_dir}")
    print(
        f"scale=({args.scale_x:.4f},{args.scale_y:.4f})  "
        f"dX,dY=({args.dx_mm:+.1f},{args.dy_mm:+.1f}) mm  "
        f"crop_mode={args.crop_mode}"
    )

    # Pass 1: build full-res aligned images + overlap boxes
    prepared = []
    for stem in ids:
        rp = find_real(real_dir, stem)
        up = find_unity_rgb(unity_dir, stem)
        dp = find_unity_depth(unity_dir, stem)
        if rp is None or up is None or dp is None:
            print(f"skip {stem}: real={rp is not None} rgb={up is not None} depth={dp is not None}")
            continue

        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if raw is None or unity is None or depth is None:
            print(f"fail read {stem}")
            continue

        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        if depth.shape[1] != IMG_W or depth.shape[0] != IMG_H:
            depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        # real: forward undistort + FOV scale
        real = undistort_bgr(raw)
        real = center_scale(real, args.scale_x, args.scale_y)

        # unity RGB+Depth: same XY shift (camera +dX,+dY)
        z_m = depths.get(stem, 0.06)
        dx, dy = mm_to_px(args.dx_mm, args.dy_mm, z_m)
        unity_a = shift_img(unity, dx, dy, nearest=False)
        depth_a = shift_img(depth, dx, dy, nearest=True)

        # valid overlap
        m_real = erode_mask(valid_mask_bgr(real), args.erode)
        m_unity = erode_mask(valid_mask_bgr(unity_a), args.erode)
        # depth valid: not near-black visualization
        if depth_a.ndim == 3:
            m_depth = erode_mask(valid_mask_bgr(depth_a[:, :, :3]), args.erode)
        else:
            m_depth = erode_mask(valid_mask_bgr(depth_a), args.erode)
        mask = cv2.bitwise_and(m_real, m_unity)
        mask = cv2.bitwise_and(mask, m_depth)

        box = bbox_of_mask(mask)
        if box is None:
            print(f"skip {stem}: empty overlap")
            continue

        prepared.append(
            {
                "stem": stem,
                "real": real,
                "unity": unity_a,
                "depth": depth_a,
                "mask": mask,
                "box": box,
                "dx_px": dx,
                "dy_px": dy,
                "depth_m": z_m,
            }
        )
        x0, y0, x1, y1 = box
        print(
            f"[{stem}] Z={z_m*1000:.1f}mm shift=({dx:+d},{dy:+d})px  "
            f"overlap bbox=({x0},{y0})-({x1},{y1}) size={x1-x0}x{y1-y0}"
        )

    if not prepared:
        print("No pairs exported.")
        return

    # Decide crop size
    sizes = [(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) for b in prepared]
    min_w = min(w for w, _ in sizes)
    min_h = min(h for _, h in sizes)
    if args.crop_mode == "resize":
        tw = args.out_w or min_w
        th = args.out_h or min_h
    elif args.crop_mode == "fixed_min":
        tw, th = min_w, min_h
    else:
        tw = th = 0  # per_image

    print(f"crop target: mode={args.crop_mode}  size={tw}x{th}" if tw else f"crop target: per_image")

    rows = []
    for item in prepared:
        stem = item["stem"]
        box = item["box"]
        if args.crop_mode == "per_image":
            x0, y0, x1, y1 = box
            real_c = item["real"][y0:y1, x0:x1]
            unity_c = item["unity"][y0:y1, x0:x1]
            depth_c = item["depth"][y0:y1, x0:x1]
            mask_c = item["mask"][y0:y1, x0:x1]
            cw, ch = x1 - x0, y1 - y0
        elif args.crop_mode == "resize":
            x0, y0, x1, y1 = box
            real_c = cv2.resize(item["real"][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA)
            unity_c = cv2.resize(item["unity"][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA)
            depth_c = cv2.resize(item["depth"][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_NEAREST)
            mask_c = cv2.resize(item["mask"][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_NEAREST)
            cw, ch = tw, th
        else:  # fixed_min center crop inside overlap bbox
            real_c = center_crop_to_size(item["real"], box, tw, th)
            unity_c = center_crop_to_size(item["unity"], box, tw, th)
            depth_c = center_crop_to_size(item["depth"], box, tw, th)
            mask_c = center_crop_to_size(item["mask"], box, tw, th)
            cw, ch = tw, th

        # depth as single-channel gray for training convenience
        if depth_c.ndim == 3:
            depth_gray = depth_c[:, :, 0]
        else:
            depth_gray = depth_c

        r_path = out_dir / f"{stem}_real.png"
        u_path = out_dir / f"{stem}_unity.png"
        d_path = out_dir / f"{stem}_depth.png"
        m_path = out_dir / f"{stem}_mask.png"
        p_path = out_dir / f"preview_{stem}.jpg"

        cv2.imwrite(str(r_path), real_c)
        cv2.imwrite(str(u_path), unity_c)
        cv2.imwrite(str(d_path), depth_gray)
        cv2.imwrite(str(m_path), mask_c)
        cv2.imwrite(str(p_path), make_preview(real_c, unity_c, depth_gray, mask_c), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        rows.append(
            {
                "id": stem,
                "real": r_path.name,
                "unity": u_path.name,
                "depth": d_path.name,
                "mask": m_path.name,
                "preview": p_path.name,
                "width": cw,
                "height": ch,
                "overlap_x0": box[0],
                "overlap_y0": box[1],
                "overlap_x1": box[2],
                "overlap_y1": box[3],
                "shift_dx_px": item["dx_px"],
                "shift_dy_px": item["dy_px"],
                "depth_m": f"{item['depth_m']:.6f}",
                "scale_x": args.scale_x,
                "scale_y": args.scale_y,
                "dx_mm": args.dx_mm,
                "dy_mm": args.dy_mm,
                "crop_mode": args.crop_mode,
            }
        )
        print(f"wrote {stem}: {cw}x{ch}")

    man = out_dir / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # small readme
    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Depth training pairs (method 1: forward undistort real)",
                f"real_dir = {real_dir}",
                f"unity_dir = {unity_dir}",
                f"scale_x,y = {args.scale_x}, {args.scale_y}",
                f"unity cam offset dX,dY mm = {args.dx_mm}, {args.dy_mm}",
                f"crop_mode = {args.crop_mode}",
                "",
                "Files per id:",
                "  {id}_real.png   undistorted+scaled real crop",
                "  {id}_unity.png  Unity RGB after same XY shift + crop",
                "  {id}_depth.png  Unity depth (gray) same warp + crop (NEAREST)",
                "  {id}_mask.png   valid overlap mask in crop",
                "  preview_{id}.jpg  real | unity | depth | mask",
                "",
                "Note: depth PNG from capture is visualization (uint8), not metric meters.",
                "Geometry of RGB and Depth is synchronized.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nDone. {len(rows)} pairs -> {out_dir}")
    print(f"manifest: {man}")


if __name__ == "__main__":
    main()
