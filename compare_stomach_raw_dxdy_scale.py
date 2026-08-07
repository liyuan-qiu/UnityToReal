"""
Stomach RAW compare (NO undistort):

  1) optional Unity rot180
  2) search best dx,dy (edge FFT-NCC) at scale=1
  3) search best sx,sy (+ refine dx,dy) about center
  4) overlap panels

Usage:
  python compare_stomach_raw_dxdy_scale.py
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
SX_MIN, SX_MAX, SX_STEPS = 0.70, 1.40, 15
SY_MIN, SY_MAX, SY_STEPS = 0.70, 1.40, 15


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


def warp_gray(g: np.ndarray, sx: float, sy: float, dx: int = 0, dy: int = 0) -> np.ndarray:
    h, w = g.shape
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w), dtype=np.float32)
    x0 = (w - nw) // 2 + dx
    y0 = (h - nh) // 2 + dy
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0 : ns0 + (ys1 - ys0), ms0 : ms0 + (xs1 - xs0)]
    return canvas


def warp_bgr(img: np.ndarray, sx: float, sy: float, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    x0 = (w - nw) // 2 + dx
    y0 = (h - nh) // 2 + dy
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0 : ns0 + (ys1 - ys0), ms0 : ms0 + (xs1 - xs0)]
    return canvas


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
    p = argparse.ArgumentParser(description="RAW stomach: dx,dy then scale (no undistort).")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData/trainingData")
    p.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData/StomachTraining6")
    p.add_argument("--ids", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--rot180", action="store_true", default=True)
    p.add_argument("--no-rot180", action="store_true")
    p.add_argument("--also-try-0", action="store_true", help="Also try rot=0 and pick better")
    p.add_argument("--out-name", type=str, default="stomach_raw_dxdy_scale")
    return p.parse_args()


def search_one(ref_e: np.ndarray, unity_w: np.ndarray) -> dict:
    # 1) dx,dy at scale 1
    ncc1, dy1, dx1 = fft_ncc_best_shift(ref_e, sobel_mag(to_gray(unity_w)))
    # 2) scale search; for each scale re-solve shift
    best = {"ncc": ncc1, "sx": 1.0, "sy": 1.0, "dx": dx1, "dy": dy1}
    mov_e = sobel_mag(to_gray(unity_w))
    for sx in np.linspace(SX_MIN, SX_MAX, SX_STEPS):
        for sy in np.linspace(SY_MIN, SY_MAX, SY_STEPS):
            scaled = warp_gray(mov_e, float(sx), float(sy), 0, 0)
            ncc, dy, dx = fft_ncc_best_shift(ref_e, scaled)
            if ncc > best["ncc"]:
                best = {"ncc": float(ncc), "sx": float(sx), "sy": float(sy), "dx": int(dx), "dy": int(dy)}
    best["ncc_shift_only"] = float(ncc1)
    best["dx_shift_only"] = int(dx1)
    best["dy_shift_only"] = int(dy1)
    return best


def main() -> None:
    args = parse_args()
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    out_dir = ROOT / "compare_out" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    try_rots = (0, 180) if args.also_try_0 else ((0,) if args.no_rot180 else (180,))

    print(f"NO undistort. real={real_dir}")
    print(f"unity={unity_dir}  try_rots={try_rots}")
    print(f"step1: dx,dy @ scale=1; step2: sx,sy grid + dx,dy")
    print(f"OUT={out_dir}")

    rows = []
    thumbs = []
    for i in ids:
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        if not rp.is_file() or not up.is_file():
            print(f"skip {i}")
            continue
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity0 = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity0 is None:
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity0.shape[1] != IMG_W or unity0.shape[0] != IMG_H:
            unity0 = cv2.resize(unity0, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        real_w = cv2.resize(raw, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(real_w))

        best = None
        for rot in try_rots:
            u = unity0 if rot == 0 else cv2.rotate(unity0, cv2.ROTATE_180)
            uw = cv2.resize(u, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
            cand = search_one(ref_e, uw)
            cand["rot"] = rot
            if best is None or cand["ncc"] > best["ncc"]:
                best = cand
                best_unity = u
        assert best is not None

        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        dx1_f = int(round(best["dx_shift_only"] * IMG_W / WORK_W))
        dy1_f = int(round(best["dy_shift_only"] * IMG_H / WORK_H))
        unity_aligned = warp_bgr(best_unity, best["sx"], best["sy"], dx_f, dy_f)

        o = cv2.resize(raw, (WORK_W, WORK_H)).astype(np.float32)
        u = cv2.resize(unity_aligned, (WORK_W, WORK_H)).astype(np.float32)
        mean_abs = float(np.mean(np.abs(o - u)))
        yc = yellow_cyan(
            cv2.resize(raw, (WORK_W, WORK_H)),
            cv2.resize(unity_aligned, (WORK_W, WORK_H)),
        )
        diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        tiles = [
            (rs(raw), "1 real RAW (no und)"),
            (rs(best_unity), f"2 unity rot={best['rot']}"),
            (rs(warp_bgr(best_unity, 1.0, 1.0, dx1_f, dy1_f)), f"3 unity +dxdy only ({dx1_f:+d},{dy1_f:+d})"),
            (rs(unity_aligned), f"4 +scale({best['sx']:.2f},{best['sy']:.2f})+xy"),
            (yc, f"5 yellow/cyan |d|={mean_abs:.1f}"),
            (diff, "6 absdiff"),
        ]
        gap, header, footer = 6, 44, 78
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        f, fs = font(15), font(12)
        draw.text(
            (gap, 10),
            f"Photo{i}_NoTag RAW vs Unity: first dxdy then scale (NO undistort)",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"step1 dx,dy=({dx1_f:+d},{dy1_f:+d}) NCC={best['ncc_shift_only']:.3f}   "
            f"step2 sx,sy=({best['sx']:.3f},{best['sy']:.3f}) dx,dy=({dx_f:+d},{dy_f:+d}) NCC={best['ncc']:.3f}",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"rot={best['rot']}  mean|raw-unity_aligned|={mean_abs:.2f}",
            fill=(255, 220, 120),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 64),
            "Warp applied on Unity to match real RAW.",
            fill=(180, 180, 180),
            font=fs,
        )
        panel = out_dir / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))

        rows.append(
            {
                "id": i,
                "rot": best["rot"],
                "dx_shift_only": dx1_f,
                "dy_shift_only": dy1_f,
                "ncc_shift_only": f"{best['ncc_shift_only']:.4f}",
                "sx": f"{best['sx']:.4f}",
                "sy": f"{best['sy']:.4f}",
                "dx": dx_f,
                "dy": dy_f,
                "edgeNCC": f"{best['ncc']:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
                "panel": panel.name,
            }
        )
        print(
            f"[{i}] rot={best['rot']:3d}  "
            f"step1 dx,dy=({dx1_f:+d},{dy1_f:+d}) NCC={best['ncc_shift_only']:.3f}  "
            f"step2 sx,sy=({best['sx']:.2f},{best['sy']:.2f}) dx,dy=({dx_f:+d},{dy_f:+d}) "
            f"NCC={best['ncc']:.3f}  |d|={mean_abs:.1f}"
        )

    if not rows:
        print("No pairs.")
        return

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

    print("\n=== SUMMARY (no undistort) ===")
    print(f"median step1 dx,dy = ({np.median([r['dx_shift_only'] for r in rows]):+.0f}, "
          f"{np.median([r['dy_shift_only'] for r in rows]):+.0f})")
    print(f"median step2 sx,sy = ({np.median([float(r['sx']) for r in rows]):.3f}, "
          f"{np.median([float(r['sy']) for r in rows]):.3f})")
    print(f"median step2 dx,dy = ({np.median([r['dx'] for r in rows]):+.0f}, "
          f"{np.median([r['dy'] for r in rows]):+.0f})")
    print(f"mean edgeNCC = {np.mean([float(r['edgeNCC']) for r in rows]):.3f}")
    print(f"mean |diff| = {np.mean([float(r['mean_absdiff']) for r in rows]):.1f}")
    print(f"summary: {out_dir / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
