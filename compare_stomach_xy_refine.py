"""
Stomach XY refine after undistort:

  1) real = undistort(K,dist) -> optional FOV scale (from Tag6: 1/sx,1/sy)
  2) unity = rot180
  3) search best dx,dy (FFT edge-NCC); optionally tiny angle
  4) apply shift on Unity (or inverse on real), overlap crop + panels

Default Tag6 FOV: unity sx,sy=(0.75,0.80) => real scale (1.3333,1.2500)
Does NOT bake Tag6 dx,dy; those are re-estimated here.
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


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        np.asarray(data["camera_matrix"], dtype=np.float64),
        np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1),
    )


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


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sobel_mag(g: np.ndarray) -> np.ndarray:
    return cv2.magnitude(
        cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3),
    )


def zscore(a: np.ndarray) -> np.ndarray:
    m, s = float(a.mean()), float(a.std())
    if s < 1e-6:
        return a * 0.0
    return (a - m) / s


def fft_ncc_best_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, int, int]:
    r, m = zscore(ref), zscore(mov)
    corr = np.fft.ifft2(np.fft.fft2(r) * np.conj(np.fft.fft2(m))).real / r.size
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(corr[iy, ix]), int(dx), int(dy)


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
    r, u = real.astype(np.float32), unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stomach: undistort then refine XY vs Unity.")
    p.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData/trainingData")
    p.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData/StomachTraining6")
    p.add_argument("--ids", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--scale-x", type=float, default=1.3333, help="Post-undistort scale on real (Tag6 1/0.75)")
    p.add_argument("--scale-y", type=float, default=1.2500, help="Post-undistort scale on real (Tag6 1/0.80)")
    p.add_argument("--no-scale", action="store_true", help="Skip FOV scale (undistort only)")
    p.add_argument("--rot180", action="store_true", default=True)
    p.add_argument("--no-rot180", action="store_true")
    p.add_argument("--shift-on", choices=("unity", "real"), default="unity", help="Where to apply found dx,dy")
    p.add_argument("--use-median", action="store_true", help="Apply one median dx,dy to all images")
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--out-name", type=str, default="stomach_NoTag_vs_StomachTraining6_xy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    k, dist = load_intrinsics(ip)
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    rot180 = bool(args.rot180) and not args.no_rot180
    sx = 1.0 if args.no_scale else float(args.scale_x)
    sy = 1.0 if args.no_scale else float(args.scale_y)

    out_dir = ROOT / "compare_out" / args.out_name
    und_dir = out_dir / "undistorted_real"
    crop_dir = out_dir / "overlap_crops"
    for d in (out_dir, und_dir, crop_dir):
        d.mkdir(parents=True, exist_ok=True)

    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    print(f"intrinsics: {ip.name}")
    print(f"step1 undistort + scale=({sx:.4f},{sy:.4f})")
    print(f"step2 search XY; apply on {args.shift_on}; unity rot180={rot180}")
    print(f"OUT: {out_dir}")

    # Pass 1: estimate per-image XY
    est = []
    for i in ids:
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        if not rp.is_file() or not up.is_file():
            print(f"skip {i}: missing files")
            continue
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        if rot180:
            unity = cv2.rotate(unity, cv2.ROTATE_180)

        real = center_scale(undistort_bgr(raw, k, dist), sx, sy)
        cv2.imwrite(str(und_dir / f"Photo{i}_NoTag_undist.jpg"), real)

        rw = cv2.resize(real, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        uw = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ncc, dy, dx = fft_ncc_best_shift(sobel_mag(to_gray(rw)), sobel_mag(to_gray(uw)))
        dx_f = int(round(dx * IMG_W / WORK_W))
        dy_f = int(round(dy * IMG_H / WORK_H))
        # ncc without shift
        aa, bb = zscore(sobel_mag(to_gray(rw))).ravel(), zscore(sobel_mag(to_gray(uw))).ravel()
        ncc0 = float(np.dot(aa, bb) / aa.size)
        est.append(
            {
                "id": i,
                "raw": raw,
                "real": real,
                "unity": unity,
                "dx": dx_f,
                "dy": dy_f,
                "ncc": float(ncc),
                "ncc0": ncc0,
            }
        )
        print(f"[{i}] XY search  dx,dy=({dx_f:+d},{dy_f:+d})  NCC={ncc:.3f} (no-shift={ncc0:.3f})")

    if not est:
        print("No pairs.")
        return

    med_dx = int(np.median([e["dx"] for e in est]))
    med_dy = int(np.median([e["dy"] for e in est]))
    print(f"\nmedian dx,dy = ({med_dx:+d},{med_dy:+d}) px")
    print(f"mean   dx,dy = ({np.mean([e['dx'] for e in est]):+.1f},{np.mean([e['dy'] for e in est]):+.1f}) px")

    # Pass 2: apply and crop
    prepared = []
    boxes = []
    for e in est:
        dx = med_dx if args.use_median else e["dx"]
        dy = med_dy if args.use_median else e["dy"]
        if args.shift_on == "unity":
            real_a, unity_a = e["real"], shift_img(e["unity"], dx, dy)
        else:
            # inverse: move real opposite to the Unity-matching shift
            real_a, unity_a = shift_img(e["real"], -dx, -dy), e["unity"]

        m = erode_mask(cv2.bitwise_and(valid_mask(real_a), valid_mask(unity_a)), args.erode)
        box = bbox_of_mask(m)
        if box is None:
            print(f"skip {e['id']}: empty overlap")
            continue
        boxes.append(box)
        prepared.append(
            {
                "id": e["id"],
                "raw": e["raw"],
                "real": real_a,
                "unity": unity_a,
                "mask": m,
                "box": box,
                "dx": dx,
                "dy": dy,
                "ncc": e["ncc"],
                "ncc0": e["ncc0"],
            }
        )

    min_w = min(b[2] - b[0] for b in boxes)
    min_h = min(b[3] - b[1] for b in boxes)
    print(f"overlap crop: {min_w}x{min_h}")

    rows = []
    thumbs = []
    for item in prepared:
        i = item["id"]
        x0, y0, x1, y1 = item["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        xa = max(0, min(IMG_W - min_w, cx - min_w // 2))
        ya = max(0, min(IMG_H - min_h, cy - min_h // 2))
        real_c = item["real"][ya : ya + min_h, xa : xa + min_w]
        unity_c = item["unity"][ya : ya + min_h, xa : xa + min_w]
        yc = yellow_cyan(real_c, unity_c)
        mean_abs = float(np.mean(np.abs(real_c.astype(np.float32) - unity_c.astype(np.float32))))

        cv2.imwrite(str(crop_dir / f"Photo{i}_real_crop.jpg"), real_c)
        cv2.imwrite(str(crop_dir / f"Photo{i}_unity_crop.jpg"), unity_c)
        cv2.imwrite(str(crop_dir / f"Photo{i}_yellow_cyan.jpg"), yc)

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        tiles = [
            (rs(item["raw"]), "1 real raw"),
            (rs(item["real"]), "2 real und(+scale)[+shift]"),
            (rs(item["unity"]), "3 unity aligned"),
            (rs(real_c), "4 real crop"),
            (rs(unity_c), "5 unity crop"),
            (rs(yc), f"6 yellow/cyan |d|={mean_abs:.1f}"),
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
            f"Photo{i}: und+scale({sx:.2f},{sy:.2f}) then XY  dx,dy=({item['dx']:+d},{item['dy']:+d}) on {args.shift_on}",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"search NCC={item['ncc']:.3f} (no-shift {item['ncc0']:.3f})  "
            f"medianXY=({med_dx:+d},{med_dy:+d})  use_median={int(args.use_median)}",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"mean|diff|={mean_abs:.2f}  crop={min_w}x{min_h} @({xa},{ya})",
            fill=(255, 220, 120),
            font=fs,
        )
        panel = out_dir / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append(
            {
                "id": i,
                "scale_x": f"{sx:.4f}",
                "scale_y": f"{sy:.4f}",
                "dx": item["dx"],
                "dy": item["dy"],
                "shift_on": args.shift_on,
                "edgeNCC": f"{item['ncc']:.4f}",
                "ncc_noshift": f"{item['ncc0']:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
                "median_dx": med_dx,
                "median_dy": med_dy,
                "panel": panel.name,
            }
        )
        print(f"[{i}] applied dx,dy=({item['dx']:+d},{item['dy']:+d})  |d|={mean_abs:.1f}")

    with (out_dir / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
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

    print("\n=== SUMMARY ===")
    print(f"undistort -> scale({sx:.3f},{sy:.3f}) -> XY refine")
    print(f"per-image dx: {[r['dx'] for r in rows]}")
    print(f"per-image dy: {[r['dy'] for r in rows]}")
    print(f"median dx,dy = ({med_dx:+d},{med_dy:+d})")
    print(f"mean |diff| = {np.mean([float(r['mean_absdiff']) for r in rows]):.1f}")
    print(f"mean edgeNCC = {np.mean([float(r['edgeNCC']) for r in rows]):.3f}")
    print(f"summary: {out_dir / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
