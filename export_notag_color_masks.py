"""
Export color masks for NoTag real + Unity (after tag-like warp).

  real  = undistort(K, dist)
  unity = rot180 -> scale(sx,sy) -> shift(dx,dy)
  mask  = pixels that are not near-black (keep colored / content)

Writes per id:
  Photo{i}_real_undist.jpg / _real_mask.png / _real_masked.png
  Photo{i}_unity_warp.jpg  / _unity_mask.png / _unity_masked.png
  Photo{i}_both_mask.png   (intersection)
  Photo{i}_both_masked_real.png / _both_masked_unity.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_W, IMG_H = 1080, 720


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


def color_mask(bgr, thr=8, sat_thr=40, erode=1, close=3):
    """Keep non-black AND sufficiently saturated (colored) pixels."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    m = ((g > thr) & (s >= sat_thr)).astype(np.uint8) * 255
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # keep largest connected component (main organ / content)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        m = np.where(labels == keep, 255, 0).astype(np.uint8)
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        m = cv2.erode(m, k)
    return m


def apply_mask(bgr, mask):
    out = bgr.copy()
    out[mask == 0] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description="Export NoTag color masks for real and Unity.")
    ap.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData20260813/capsule_intrinsics.npz")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "trainingData20260813")
    ap.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData20260813/NoTagUnity10")
    ap.add_argument("--ids", type=str, default="1,2,3")
    ap.add_argument("--real-name", type=str, default="Photo{id}_NoTag.jpg")
    ap.add_argument("--unity-name", type=str, default="Photo{id}_tag_Unity.jpg")
    ap.add_argument("--sx", type=float, default=0.75)
    ap.add_argument("--sy", type=float, default=0.85)
    ap.add_argument("--dx", type=int, default=32)
    ap.add_argument("--dy", type=int, default=78)
    ap.add_argument("--thr", type=int, default=8, help="Gray threshold: keep pixels > thr")
    ap.add_argument("--sat-thr", type=int, default=40, help="HSV S threshold: keep saturated (colored) pixels")
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--out-name", type=str, default="training20260813_NoTag_vs_NoTagUnity10_masks")
    args = ap.parse_args()

    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    k, dist = load_intrinsics(ip)

    out = ROOT / "compare_out" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]

    print(f"OUT={out}")
    print(
        f"Unity warp sx,sy=({args.sx},{args.sy}) dx,dy=({args.dx:+d},{args.dy:+d}) "
        f"thr={args.thr} sat_thr={args.sat_thr}"
    )

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
        unity = warp_bgr(cv2.rotate(unity, cv2.ROTATE_180), args.sx, args.sy, args.dx, args.dy)

        m_real = color_mask(real, thr=args.thr, sat_thr=args.sat_thr, erode=args.erode)
        m_unity = color_mask(unity, thr=args.thr, sat_thr=args.sat_thr, erode=args.erode)
        m_both = cv2.bitwise_and(m_real, m_unity)

        real_m = apply_mask(real, m_real)
        unity_m = apply_mask(unity, m_unity)
        real_b = apply_mask(real, m_both)
        unity_b = apply_mask(unity, m_both)

        stem = f"Photo{i}"
        cv2.imwrite(str(out / f"{stem}_real_undist.jpg"), real)
        cv2.imwrite(str(out / f"{stem}_unity_warp.jpg"), unity)
        cv2.imwrite(str(out / f"{stem}_real_mask.png"), m_real)
        cv2.imwrite(str(out / f"{stem}_unity_mask.png"), m_unity)
        cv2.imwrite(str(out / f"{stem}_both_mask.png"), m_both)
        cv2.imwrite(str(out / f"{stem}_real_masked.png"), real_m)
        cv2.imwrite(str(out / f"{stem}_unity_masked.png"), unity_m)
        cv2.imwrite(str(out / f"{stem}_both_masked_real.png"), real_b)
        cv2.imwrite(str(out / f"{stem}_both_masked_unity.png"), unity_b)

        pr = float(np.mean(m_real > 0) * 100)
        pu = float(np.mean(m_unity > 0) * 100)
        pb = float(np.mean(m_both > 0) * 100)
        if np.any(m_both):
            d = float(
                np.mean(
                    np.abs(real.astype(np.float32) - unity.astype(np.float32))[m_both > 0]
                )
            )
        else:
            d = float("nan")
        print(f"[{i}] mask% real={pr:.1f} unity={pu:.1f} both={pb:.1f}  mean|d|@both={d:.1f}")

    print("done.")


if __name__ == "__main__":
    main()
