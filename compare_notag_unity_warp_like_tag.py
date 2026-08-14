"""
NoTag vs Unity, same side as tagged compare:

  real  = undistort(K, dist) only
  unity = rot180 -> center_scale(sx,sy) -> shift(dx,dy)

Defaults from training20260811 tag vs TagUnity median/mean:
  sx,sy=(0.75, 0.85), dx,dy=(+4, +26)
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


def load_intrinsics(path: Path):
    d = np.load(path)
    return np.asarray(d["camera_matrix"], np.float64), np.asarray(d["dist_coeffs"], np.float64).reshape(-1)


def undistort_bgr(bgr, k, dist):
    h, w = bgr.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, k, dist, None, new_k)


def warp_bgr(img, sx, sy, dx, dy):
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    x0, y0 = (w - nw) // 2 + dx, (h - nh) // 2 + dy
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0 : ns0 + (ys1 - ys0), ms0 : ms0 + (xs1 - xs0)]
    return canvas


def valid_mask(img, thr=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return (g > thr).astype(np.uint8) * 255


def erode_mask(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def bbox_of_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def yellow_cyan(real, unity):
    r, u = real.astype(np.float32), unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "trainingData20260811")
    ap.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData20260811/RealUnity")
    ap.add_argument("--ids", type=str, default="1,2,3,4")
    ap.add_argument("--real-name", type=str, default="Photo{id}_NoTag.jpg")
    ap.add_argument("--unity-name", type=str, default="Photo{id}_tag_Unity.jpg")
    ap.add_argument("--sx", type=float, default=0.75)
    ap.add_argument("--sy", type=float, default=0.85)
    ap.add_argument("--dx", type=int, default=4)
    ap.add_argument("--dy", type=int, default=26)
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--out-name", type=str, default="training20260811_NoTag_vs_RealUnity_unityWarp")
    args = ap.parse_args()

    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    k, dist = load_intrinsics(ip)

    out = ROOT / "compare_out" / args.out_name
    und_dir = out / "undistorted_real"
    crop_dir = out / "overlap_crops"
    for d in (out, und_dir, crop_dir):
        d.mkdir(parents=True, exist_ok=True)

    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    print("TAG-like pipeline: real=undistort only; Unity=rot180 -> scale -> shift")
    print(f"Unity warp: sx,sy=({args.sx},{args.sy}) dx,dy=({args.dx:+d},{args.dy:+d})")
    print(f"OUT={out}")

    prepared, boxes = [], []
    for i in ids:
        rp = real_dir / args.real_name.format(id=i)
        up = unity_dir / args.unity_name.format(id=i)
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"skip {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        real = undistort_bgr(raw, k, dist)
        cv2.imwrite(str(und_dir / f"Photo{i}_NoTag_undist.jpg"), real)

        unity = cv2.rotate(unity, cv2.ROTATE_180)
        unity = warp_bgr(unity, args.sx, args.sy, args.dx, args.dy)

        m = erode_mask(cv2.bitwise_and(valid_mask(real), valid_mask(unity)), args.erode)
        box = bbox_of_mask(m)
        if box is None:
            print(f"skip {i}: empty overlap")
            continue
        boxes.append(box)
        prepared.append({"id": i, "raw": raw, "real": real, "unity": unity, "mask": m, "box": box})
        x0, y0, x1, y1 = box
        print(f"[{i}] overlap {x1-x0}x{y1-y0}")

    min_w = min(b[2] - b[0] for b in boxes)
    min_h = min(b[3] - b[1] for b in boxes)
    rows, thumbs = [], []

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
            (rs(item["raw"]), "1 real raw NoTag"),
            (rs(item["real"]), "2 real undistort ONLY"),
            (rs(item["unity"]), f"3 unity rot180+s({args.sx:.2f},{args.sy:.2f})+({args.dx:+d},{args.dy:+d})"),
            (rs(real_c), "4 real crop"),
            (rs(unity_c), "5 unity crop"),
            (rs(yc), f"6 yellow/cyan |d|={mean_abs:.1f}"),
        ]
        gap, header, footer = 6, 44, 70
        canvas = Image.new("RGB", (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer), (22, 22, 22))
        draw = ImageDraw.Draw(canvas)
        f, fs = font(15), font(12)
        draw.text((gap, 10), f"Photo{i}: TAG-like — und on real; scale+shift on Unity", fill=(240, 240, 240), font=f)
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"from tag compare: Unity sx,sy=({args.sx},{args.sy}) dx,dy=({args.dx:+d},{args.dy:+d}) after rot180",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text((gap, header + WORK_H + 46), f"mean|diff|={mean_abs:.2f} crop={min_w}x{min_h}", fill=(255, 220, 120), font=fs)
        panel = out / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append({"id": i, "sx": args.sx, "sy": args.sy, "dx": args.dx, "dy": args.dy, "mean_absdiff": f"{mean_abs:.2f}", "panel": panel.name})
        print(f"[{i}] |d|={mean_abs:.1f} -> {panel.name}")

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
    print(f"mean|d|={np.mean([float(r['mean_absdiff']) for r in rows]):.1f}  summary={out/'compare_summary.csv'}")


if __name__ == "__main__":
    main()
