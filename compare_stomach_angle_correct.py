"""
Stomach angle correction on top of TagTraining6 warp:

  real  = undistort(K,dist) -> scale(1/sx,1/sy) -> shift(-dx,-dy)
  unity = rot180 (tag default) -> search small residual rotation + shift
  then overlap crop + yellow/cyan panels

Also reports a single recommended global residual angle (median over ids).
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

DEFAULT_UNITY_SX, DEFAULT_UNITY_SY = 0.75, 0.80
DEFAULT_DX, DEFAULT_DY = 114, 22


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


def rotate_keep(img: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sobel_mag(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


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
    r = real.astype(np.float32)
    u = unity.astype(np.float32)
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


def search_angle(
    ref_e: np.ndarray,
    unity_w: np.ndarray,
    angles: np.ndarray,
) -> dict:
    best = {"ncc": -1e9, "angle": 0.0, "dx": 0, "dy": 0}
    for ang in angles:
        mov = rotate_keep(unity_w, float(ang))
        ncc, dy, dx = fft_ncc_best_shift(ref_e, sobel_mag(to_gray(mov)))
        if ncc > best["ncc"]:
            best = {"ncc": float(ncc), "angle": float(ang), "dx": int(dx), "dy": int(dy)}
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stomach residual angle correction vs Unity.")
    p.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData/trainingData")
    p.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData/StomachTraining6")
    p.add_argument("--ids", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--unity-sx", type=float, default=DEFAULT_UNITY_SX)
    p.add_argument("--unity-sy", type=float, default=DEFAULT_UNITY_SY)
    p.add_argument("--dx", type=int, default=DEFAULT_DX)
    p.add_argument("--dy", type=int, default=DEFAULT_DY)
    p.add_argument("--base-rot180", action="store_true", default=True)
    p.add_argument("--no-base-rot180", action="store_true")
    p.add_argument("--angle-min", type=float, default=-20.0)
    p.add_argument("--angle-max", type=float, default=20.0)
    p.add_argument("--angle-step", type=float, default=1.0)
    p.add_argument("--refine-step", type=float, default=0.25)
    p.add_argument("--global-angle", type=float, default=None, help="Force one angle for all (deg on Unity)")
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--out-name", type=str, default="stomach_NoTag_vs_StomachTraining6_angle")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k, dist = load_intrinsics(args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics)
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    base180 = bool(args.base_rot180) and not args.no_base_rot180

    real_sx, real_sy = 1.0 / args.unity_sx, 1.0 / args.unity_sy
    real_dx, real_dy = -int(args.dx), -int(args.dy)

    out_dir = ROOT / "compare_out" / args.out_name
    crop_dir = out_dir / "overlap_crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    coarse = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step)

    print(f"real bake: scale=({real_sx:.4f},{real_sy:.4f}) shift=({real_dx:+d},{real_dy:+d})")
    print(f"unity base rot180={base180}; residual angle search [{args.angle_min},{args.angle_max}] step={args.angle_step}")
    print(f"OUT: {out_dir}")

    # Pass 1: per-image best angle
    per = []
    for i in ids:
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        if not rp.is_file() or not up.is_file():
            print(f"skip {i}")
            continue
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        if base180:
            unity = cv2.rotate(unity, cv2.ROTATE_180)

        real = shift_img(center_scale(undistort_bgr(raw, k, dist), real_sx, real_sy), real_dx, real_dy)
        real_w = cv2.resize(real, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(real_w))

        if args.global_angle is not None:
            best = search_angle(ref_e, unity_w, np.array([args.global_angle], dtype=float))
        else:
            best = search_angle(ref_e, unity_w, coarse)
            # refine
            refine = np.arange(best["angle"] - 2.0, best["angle"] + 2.0 + 0.5 * args.refine_step, args.refine_step)
            best = search_angle(ref_e, unity_w, refine)

        # scale shift to full res
        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        per.append(
            {
                "id": i,
                "raw": raw,
                "real": real,
                "unity0": unity,
                "angle": best["angle"],
                "dx": dx_f,
                "dy": dy_f,
                "ncc": best["ncc"],
            }
        )
        print(
            f"[{i}] residual_angle={best['angle']:+.2f} deg  "
            f"extra_shift=({dx_f:+d},{dy_f:+d})  edgeNCC={best['ncc']:.3f}"
        )

    if not per:
        print("No pairs.")
        return

    angles = [p["angle"] for p in per]
    med_ang = float(np.median(angles))
    print(f"\nper-image angles: {[round(a, 2) for a in angles]}")
    print(f"median residual angle = {med_ang:+.2f} deg  (suggest Unity extra rotate by this)")

    # Pass 2: apply per-image best (or global) and write panels
    use_global = args.global_angle is not None
    apply_ang = args.global_angle if use_global else None
    rows = []
    thumbs = []
    boxes = []
    prepared = []

    for p in per:
        ang = float(apply_ang) if use_global else float(p["angle"])
        # If using median global for a cleaner second run, user can pass --global-angle
        unity = rotate_keep(p["unity0"], ang)
        # also apply the shift found at that angle (work->full already stored for per-image angle;
        # recompute shift at chosen angle for consistency)
        real_w = cv2.resize(p["real"], (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ncc, dy, dx = fft_ncc_best_shift(sobel_mag(to_gray(real_w)), sobel_mag(to_gray(unity_w)))
        dx_f = int(round(dx * IMG_W / WORK_W))
        dy_f = int(round(dy * IMG_H / WORK_H))
        unity = shift_img(unity, dx_f, dy_f)

        m = erode_mask(cv2.bitwise_and(valid_mask(p["real"]), valid_mask(unity)), args.erode)
        box = bbox_of_mask(m)
        if box is None:
            print(f"skip {p['id']}: empty overlap")
            continue
        boxes.append(box)
        prepared.append(
            {
                "id": p["id"],
                "raw": p["raw"],
                "real": p["real"],
                "unity": unity,
                "mask": m,
                "box": box,
                "angle": ang,
                "dx": dx_f,
                "dy": dy_f,
                "ncc": float(ncc),
            }
        )

    min_w = min(b[2] - b[0] for b in boxes)
    min_h = min(b[3] - b[1] for b in boxes)
    print(f"overlap crop: {min_w}x{min_h}")

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
            (rs(item["real"]), "2 real und+tag6warp"),
            (rs(item["unity"]), f"3 unity rot180+{item['angle']:+.1f}+shift"),
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
            f"Photo{i}: angle corr {item['angle']:+.2f}deg  shift=({item['dx']:+d},{item['dy']:+d})  NCC={item['ncc']:.3f}",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"base: tag6 scale/shift on real; Unity rot180 + residual angle; "
            f"median_angle={med_ang:+.2f}deg",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"mean|diff|={mean_abs:.2f}  crop={min_w}x{min_h}",
            fill=(255, 220, 120),
            font=fs,
        )
        panel = out_dir / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append(
            {
                "id": i,
                "residual_angle_deg": f"{item['angle']:.3f}",
                "extra_dx": item["dx"],
                "extra_dy": item["dy"],
                "edgeNCC": f"{item['ncc']:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
                "median_angle_all": f"{med_ang:.3f}",
                "panel": panel.name,
            }
        )
        print(f"[{i}] applied angle={item['angle']:+.2f} shift=({item['dx']:+d},{item['dy']:+d}) |d|={mean_abs:.1f}")

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
    print(f"median residual angle = {med_ang:+.2f} deg")
    print(f"mean |diff| = {np.mean([float(r['mean_absdiff']) for r in rows]):.1f}")
    print(f"mean edgeNCC = {np.mean([float(r['edgeNCC']) for r in rows]):.3f}")
    print(f"If you want one Unity extra rotation for all: {med_ang:+.2f} deg (after rot180)")
    print(f"summary: {out_dir / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
