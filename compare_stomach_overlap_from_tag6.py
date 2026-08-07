"""
Stomach (NoTag) vs Unity overlap compare, baking TagTraining6 alignment into REAL:

  real  = undistort(K,dist) -> center_scale(1/sx, 1/sy) -> shift(-dx, -dy)
  unity = optional rot180 (tag compare preferred 180), then overlap crop

TagTraining6 defaults (median / mean from compare_summary):
  Unity warp was sx,sy=(0.75,0.80), dx,dy=(+114,+22) on Unity after rot180.
  Inverse on real: scale (1.3333, 1.2500), shift (-114, -22).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
IMG_W, IMG_H = 1080, 720
WORK_W, WORK_H = 540, 360

# TagTraining6 summary
DEFAULT_UNITY_SX = 0.75
DEFAULT_UNITY_SY = 0.80
DEFAULT_DX = 114  # Unity shift that matched und real
DEFAULT_DY = 22


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    k = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    return k, dist


def undistort_bgr(bgr: np.ndarray, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, k, dist, None, new_k)


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


def shift_img(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask(img: np.ndarray, thr: int = 8) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return (g > thr).astype(np.uint8) * 255


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def yellow_cyan(real: np.ndarray, unity: np.ndarray) -> np.ndarray:
    """Real -> yellow channel bias, Unity -> cyan; overlap looks neutral/gray."""
    r = real.astype(np.float32)
    u = unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]  # B from unity (cyan-ish)
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]  # R from real (yellow-ish)
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def resolve_unity_dir(preferred: Path, fallback: Path) -> Path:
    pref = preferred if preferred.is_absolute() else (ROOT / preferred)
    fb = fallback if fallback.is_absolute() else (ROOT / fallback)
    n_pref = len(list(pref.glob("*_Unity.jpg"))) if pref.is_dir() else 0
    if n_pref > 0:
        return pref.resolve()
    if fb.is_dir() and list(fb.glob("*_Unity.jpg")):
        print(f"WARNING: {pref} has no Unity JPGs; using fallback {fb}")
        return fb.resolve()
    raise SystemExit(f"No Unity JPGs in {pref} or {fb}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stomach NoTag und+tag6 warp vs Unity overlap.")
    p.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData/trainingData")
    p.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData/StomachTraining6")
    p.add_argument("--unity-fallback", type=Path, default=ROOT / "trainingData/StomachTraing")
    p.add_argument("--ids", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--real-name", type=str, default="Photo{id}_NoTag.jpg")
    p.add_argument("--unity-name", type=str, default="Photo{id}_tag_Unity.jpg")
    p.add_argument("--unity-sx", type=float, default=DEFAULT_UNITY_SX, help="Tag6 Unity shrink sx")
    p.add_argument("--unity-sy", type=float, default=DEFAULT_UNITY_SY, help="Tag6 Unity shrink sy")
    p.add_argument("--dx", type=int, default=DEFAULT_DX, help="Tag6 Unity dx (px)")
    p.add_argument("--dy", type=int, default=DEFAULT_DY, help="Tag6 Unity dy (px)")
    p.add_argument("--rot180", action="store_true", default=True, help="Rotate Unity 180 (tag default)")
    p.add_argument("--no-rot180", action="store_true", help="Do not rotate Unity")
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--out-name", type=str, default="stomach_NoTag_vs_StomachTraining6_tag6warp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k, dist = load_intrinsics(args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics)
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = resolve_unity_dir(args.unity_dir, args.unity_fallback)
    rot180 = bool(args.rot180) and not args.no_rot180

    # Inverse of Unity warp from TagTraining6
    real_sx = 1.0 / args.unity_sx
    real_sy = 1.0 / args.unity_sy
    real_dx = -int(args.dx)
    real_dy = -int(args.dy)

    out_dir = ROOT / "compare_out" / args.out_name
    und_dir = out_dir / "undistorted_real_warped"
    crop_dir = out_dir / "overlap_crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    und_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    ids = [int(s.strip()) for s in args.ids.split(",") if s.strip()]
    print(f"intrinsics: {args.intrinsics}")
    print(f"  fx,fy=({k[0,0]:.2f},{k[1,1]:.2f}) cx,cy=({k[0,2]:.2f},{k[1,2]:.2f})")
    print(f"  dist={dist.tolist()}")
    print(f"real_dir : {real_dir}")
    print(f"unity_dir: {unity_dir}")
    print(f"Tag6 Unity warp: sx,sy=({args.unity_sx},{args.unity_sy}) dx,dy=({args.dx:+d},{args.dy:+d}) rot180={rot180}")
    print(f"REAL bake:      scale=({real_sx:.4f},{real_sy:.4f}) shift=({real_dx:+d},{real_dy:+d})")
    print(f"OUT: {out_dir}")

    rows = []
    thumbs = []
    boxes = []

    prepared = []
    for i in ids:
        rp = real_dir / args.real_name.format(id=i)
        up = unity_dir / args.unity_name.format(id=i)
        if not rp.is_file() or not up.is_file():
            print(f"skip {i}: missing {rp.name if not rp.is_file() else ''} {up.name if not up.is_file() else ''}")
            continue

        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"fail read {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            print(f"  note: Unity {up.name} is {unity.shape[1]}x{unity.shape[0]} -> resize {IMG_W}x{IMG_H}")
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        if rot180:
            unity = cv2.rotate(unity, cv2.ROTATE_180)

        real = undistort_bgr(raw, k, dist)
        real = center_scale(real, real_sx, real_sy)
        real = shift_img(real, real_dx, real_dy)
        und_path = und_dir / f"Photo{i}_NoTag_undist_tag6warp.jpg"
        cv2.imwrite(str(und_path), real)

        m = erode_mask(cv2.bitwise_and(valid_mask(real), valid_mask(unity)), args.erode)
        box = bbox_of_mask(m)
        if box is None:
            print(f"skip {i}: empty overlap")
            continue
        boxes.append(box)
        prepared.append(
            {
                "id": i,
                "raw": raw,
                "real": real,
                "unity": unity,
                "mask": m,
                "box": box,
                "und_path": und_path.name,
            }
        )
        x0, y0, x1, y1 = box
        print(f"[{i}] overlap bbox=({x0},{y0})-({x1},{y1}) size={x1-x0}x{y1-y0}")

    if not prepared:
        print("No pairs.")
        return

    min_w = min(b[2] - b[0] for b in boxes)
    min_h = min(b[3] - b[1] for b in boxes)
    print(f"fixed overlap crop: {min_w}x{min_h}")

    for item in prepared:
        i = item["id"]
        x0, y0, x1, y1 = item["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        xa = max(0, min(IMG_W - min_w, cx - min_w // 2))
        ya = max(0, min(IMG_H - min_h, cy - min_h // 2))
        real_c = item["real"][ya : ya + min_h, xa : xa + min_w]
        unity_c = item["unity"][ya : ya + min_h, xa : xa + min_w]
        mask_c = item["mask"][ya : ya + min_h, xa : xa + min_w]
        yc = yellow_cyan(real_c, unity_c)
        diff = np.clip(np.abs(real_c.astype(np.float32) - unity_c.astype(np.float32)), 0, 255).astype(np.uint8)
        mean_abs = float(np.mean(np.abs(real_c.astype(np.float32) - unity_c.astype(np.float32))))

        cv2.imwrite(str(crop_dir / f"Photo{i}_real_crop.jpg"), real_c)
        cv2.imwrite(str(crop_dir / f"Photo{i}_unity_crop.jpg"), unity_c)
        cv2.imwrite(str(crop_dir / f"Photo{i}_mask_crop.png"), mask_c)
        cv2.imwrite(str(crop_dir / f"Photo{i}_yellow_cyan.jpg"), yc)
        cv2.imwrite(str(crop_dir / f"Photo{i}_absdiff.jpg"), diff)

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        tiles = [
            (rs(item["raw"]), "1 real raw NoTag"),
            (rs(item["real"]), "2 real und+scale+shift"),
            (rs(item["unity"]), f"3 unity{' rot180' if rot180 else ''}"),
            (rs(real_c), "4 real overlap crop"),
            (rs(unity_c), "5 unity overlap crop"),
            (rs(yc), f"6 yellow/cyan (mean|d|={mean_abs:.1f})"),
        ]
        gap, header, footer = 6, 44, 70
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        f, fs = font(15), font(12)
        draw.text(
            (gap, 10),
            f"Photo{i}_NoTag: und(K,dist)+scale({real_sx:.3f},{real_sy:.3f})+shift({real_dx:+d},{real_dy:+d}) vs Unity",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"Tag6 Unity was sx,sy=({args.unity_sx},{args.unity_sy}) dx,dy=({args.dx:+d},{args.dy:+d}); "
            f"overlap crop {min_w}x{min_h} @({xa},{ya})",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"mean|real-unity|={mean_abs:.2f}  unity_src={unity_dir.name}",
            fill=(255, 220, 120),
            font=fs,
        )
        panel_path = out_dir / f"compare_{i}.png"
        canvas.save(panel_path)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))

        rows.append(
            {
                "id": i,
                "real_scale_x": f"{real_sx:.4f}",
                "real_scale_y": f"{real_sy:.4f}",
                "real_dx": real_dx,
                "real_dy": real_dy,
                "unity_rot180": int(rot180),
                "overlap_w": min_w,
                "overlap_h": min_h,
                "crop_x": xa,
                "crop_y": ya,
                "mean_absdiff": f"{mean_abs:.2f}",
                "undist": item["und_path"],
                "panel": panel_path.name,
            }
        )
        print(f"[{i}] crop@({xa},{ya}) mean|diff|={mean_abs:.1f} -> {panel_path.name}")

    csv_path = out_dir / "compare_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if thumbs:
        tw, th = thumbs[0].size
        triage = Image.new("RGB", (tw, th * len(thumbs) + 8 * (len(thumbs) + 1)), (18, 18, 18))
        y = 8
        for t in thumbs:
            triage.paste(t, (0, y))
            y += th + 8
        triage.save(out_dir / "triage_all.png")

    diffs = [float(r["mean_absdiff"]) for r in rows]
    print("\n=== SUMMARY ===")
    print(f"pairs={len(rows)}  overlap_crop={min_w}x{min_h}")
    print(f"mean|absdiff| mean={np.mean(diffs):.1f} median={np.median(diffs):.1f}")
    print(f"summary: {csv_path}")
    print(f"warped reals: {und_dir}")
    print(f"crops: {crop_dir}")


if __name__ == "__main__":
    main()
