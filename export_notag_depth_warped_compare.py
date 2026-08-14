"""
Warp NoTag Unity Depth with the same transform as Unity RGB, then
add depth panels into the masked NoTag compare folder.

  depth = rot180 -> center_scale(sx,sy) -> shift(dx,dy)  [NEAREST]
  also masked with Photo{i}_both_mask.png
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
WORK_W, WORK_H = 400, 267


def font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def warp_nearest(img, sx, sy, dx, dy):
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        canvas = np.zeros((h, w), dtype=img.dtype)
    else:
        canvas = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    x0, y0 = (w - nw) // 2 + dx, (h - nh) // 2 + dy
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0 : ns0 + (ys1 - ys0), ms0 : ms0 + (xs1 - xs0)]
    return canvas


def depth_to_bgr(depth):
    if depth.ndim == 2:
        return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    if depth.shape[2] == 4:
        return depth[:, :, :3].copy()
    return depth[:, :, :3].copy()


def colorize_depth_vis(depth_bgr, mask=None):
    """Turn (possibly gray) depth viz into a jet-like preview for panels."""
    g = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        valid = (mask > 0) & (g > 1)
    else:
        valid = g > 1
    out = np.zeros((*g.shape, 3), np.uint8)
    if not np.any(valid):
        return out
    vals = g[valid].astype(np.float32)
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    if hi <= lo + 1e-3:
        lo, hi = float(vals.min()), float(vals.max() + 1)
    norm = np.zeros_like(g, np.float32)
    norm[valid] = np.clip((g[valid].astype(np.float32) - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    out[valid] = colored[valid]
    return out


def yellow_cyan(real, unity):
    r, u = real.astype(np.float32), unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0, 255).astype(np.uint8)


def bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData20260813/NoTagUnity10")
    ap.add_argument(
        "--mask-dir",
        type=Path,
        default=ROOT / "compare_out/training20260813_NoTag_vs_NoTagUnity10_masks",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "compare_out/training20260813_NoTag_vs_NoTagUnity10_masked_compare",
    )
    ap.add_argument("--ids", type=str, default="1,2,3")
    ap.add_argument("--sx", type=float, default=0.75)
    ap.add_argument("--sy", type=float, default=0.85)
    ap.add_argument("--dx", type=int, default=32)
    ap.add_argument("--dy", type=int, default=78)
    ap.add_argument("--unity-depth-name", type=str, default="Photo{id}_tag_Depth.png")
    args = ap.parse_args()

    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    mask_dir = args.mask_dir if args.mask_dir.is_absolute() else ROOT / args.mask_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]

    print(f"Depth warp: rot180 -> s({args.sx},{args.sy}) -> ({args.dx:+d},{args.dy:+d})")
    print(f"OUT={out_dir}")

    rows, thumbs = [], []
    for i in ids:
        dp = unity_dir / args.unity_depth_name.format(id=i)
        depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        real = cv2.imread(str(mask_dir / f"Photo{i}_both_masked_real.png"), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(mask_dir / f"Photo{i}_both_masked_unity.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"Photo{i}_both_mask.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None or real is None or unity is None or mask is None:
            print(f"skip {i}: missing depth/mask/rgb")
            continue

        if depth.shape[1] != IMG_W or depth.shape[0] != IMG_H:
            depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        depth = cv2.rotate(depth, cv2.ROTATE_180)
        depth = warp_nearest(depth, args.sx, args.sy, args.dx, args.dy)
        depth_bgr = depth_to_bgr(depth)
        depth_masked = depth_bgr.copy()
        depth_masked[mask == 0] = 0
        depth_color = colorize_depth_vis(depth_bgr, mask)
        depth_color_full = colorize_depth_vis(depth_bgr, None)

        # overlay depth edges on real for visual check
        g = cv2.cvtColor(depth_masked, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 40, 120)
        overlay = real.copy()
        overlay[edges > 0] = (0, 255, 255)

        cv2.imwrite(str(out_dir / f"Photo{i}_depth_warp.png"), depth_bgr)
        cv2.imwrite(str(out_dir / f"Photo{i}_depth_warp_masked.png"), depth_masked)
        cv2.imwrite(str(out_dir / f"Photo{i}_depth_warp_color.png"), depth_color)
        cv2.imwrite(str(out_dir / f"Photo{i}_depth_on_real.png"), overlay)

        box = bbox(mask)
        if box is None:
            print(f"skip {i}: empty mask")
            continue
        x0, y0, x1, y1 = box
        pad = 8
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(IMG_W, x1 + pad), min(IMG_H, y1 + pad)
        real_c = real[y0:y1, x0:x1]
        unity_c = unity[y0:y1, x0:x1]
        depth_c = depth_color[y0:y1, x0:x1]
        yc = yellow_cyan(real_c, unity_c)
        overlay_c = overlay[y0:y1, x0:x1]

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        tiles = [
            (rs(real_c), "1 real@mask"),
            (rs(unity_c), "2 unity RGB@mask"),
            (rs(depth_c), "3 depth warped@mask"),
            (rs(yc), "4 yellow/cyan RGB"),
            (rs(overlay_c), "5 depth edges on real"),
            (rs(depth_color_full[y0:y1, x0:x1]), "6 depth color (full warp)"),
        ]
        gap, header, footer = 6, 44, 70
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        f, fs = font(14), font(11)
        draw.text(
            (gap, 10),
            f"Photo{i}: NoTag masked RGB + Depth (Unity warp same as RGB)",
            fill=(240, 240, 240),
            font=f,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"Depth: rot180 -> sx,sy=({args.sx},{args.sy}) dx,dy=({args.dx:+d},{args.dy:+d}) NEAREST",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"source={dp.name}  saved depth_warp / depth_warp_masked / depth_on_real",
            fill=(255, 220, 120),
            font=fs,
        )
        panel = out_dir / f"compare_depth_{i}.png"
        canvas.save(panel)
        # also refresh main compare_{i} to include depth as last extra? keep separate compare_depth_*
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append({"id": i, "depth_src": dp.name, "panel": panel.name})
        print(f"[{i}] wrote {panel.name} + depth pngs")

    with (out_dir / "depth_compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
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
        triage.save(out_dir / "triage_depth_all.png")

    print("done.")


if __name__ == "__main__":
    main()
