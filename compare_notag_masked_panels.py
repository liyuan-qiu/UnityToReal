"""
Side-by-side compare of masked NoTag real vs Unity (intersection mask).
Reads outputs from export_notag_color_masks.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WORK_W, WORK_H = 540, 360


def font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def yellow_cyan(real, unity):
    r, u = real.astype(np.float32), unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0, 255).astype(np.uint8)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sobel_mag(g):
    return cv2.magnitude(
        cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3),
    )


def zscore(a):
    m, s = float(a.mean()), float(a.std())
    return a * 0.0 if s < 1e-6 else (a - m) / s


def edge_ncc_masked(real, unity, mask):
    sel = mask > 0
    if int(sel.sum()) < 64:
        return float("nan")
    a = zscore(sobel_mag(to_gray(real))[sel])
    b = zscore(sobel_mag(to_gray(unity))[sel])
    return float(np.dot(a, b) / a.size)


def bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mask-dir",
        type=Path,
        default=ROOT / "compare_out/training20260813_NoTag_vs_NoTagUnity10_masks",
    )
    ap.add_argument("--ids", type=str, default="1,2,3")
    ap.add_argument(
        "--out-name",
        type=str,
        default="training20260813_NoTag_vs_NoTagUnity10_masked_compare",
    )
    args = ap.parse_args()

    mask_dir = args.mask_dir if args.mask_dir.is_absolute() else ROOT / args.mask_dir
    out = ROOT / "compare_out" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]

    rows, thumbs = [], []
    for i in ids:
        real = cv2.imread(str(mask_dir / f"Photo{i}_both_masked_real.png"), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(mask_dir / f"Photo{i}_both_masked_unity.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"Photo{i}_both_mask.png"), cv2.IMREAD_GRAYSCALE)
        real_full = cv2.imread(str(mask_dir / f"Photo{i}_real_masked.png"), cv2.IMREAD_COLOR)
        unity_full = cv2.imread(str(mask_dir / f"Photo{i}_unity_masked.png"), cv2.IMREAD_COLOR)
        if any(x is None for x in (real, unity, mask, real_full, unity_full)):
            print(f"skip {i}: missing files in {mask_dir}")
            continue

        sel = mask > 0
        mean_d = float(np.mean(np.abs(real.astype(np.float32) - unity.astype(np.float32))[sel]))
        ncc = edge_ncc_masked(real, unity, mask)
        box = bbox(mask)
        if box is None:
            print(f"skip {i}: empty mask")
            continue
        x0, y0, x1, y1 = box
        # pad a bit
        pad = 8
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)
        real_c = real[y0:y1, x0:x1]
        unity_c = unity[y0:y1, x0:x1]
        yc = yellow_cyan(real_c, unity_c)
        mask_c = cv2.cvtColor(mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        tiles = [
            (rs(real_full), "1 real masked"),
            (rs(unity_full), "2 unity masked"),
            (rs(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), "3 both mask"),
            (rs(real_c), "4 real crop@both"),
            (rs(unity_c), "5 unity crop@both"),
            (rs(yc), f"6 yellow/cyan |d|={mean_d:.1f}"),
        ]
        gap, header, footer = 6, 44, 72
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        f, fs = font(15), font(12)
        draw.text(
            (gap, 10),
            f"Photo{i}: NoTag masked compare (intersection of color masks)",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"mean|d|@both={mean_d:.2f}   edgeNCC@both={ncc:.3f}   mask_cover={100.0*sel.mean():.1f}%",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 48),
            "Similarity: lower |d| and higher edgeNCC => better geometric match (lighting still affects |d|)",
            fill=(255, 220, 120),
            font=fs,
        )
        panel = out / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append(
            {
                "id": i,
                "mean_absdiff_both": f"{mean_d:.2f}",
                "edgeNCC_both": f"{ncc:.4f}",
                "mask_cover_pct": f"{100.0*sel.mean():.2f}",
                "panel": panel.name,
            }
        )
        print(f"[{i}] |d|={mean_d:.1f}  edgeNCC={ncc:.3f}  -> {panel}")

    with (out / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
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
        triage.save(out / "triage_all.png")

    print(
        f"mean |d|={np.mean([float(r['mean_absdiff_both']) for r in rows]):.1f}  "
        f"mean edgeNCC={np.mean([float(r['edgeNCC_both']) for r in rows]):.3f}"
    )
    print(f"OUT={out}")


if __name__ == "__main__":
    main()
