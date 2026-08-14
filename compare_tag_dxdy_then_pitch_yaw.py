"""
Tag only: first dx,dy (after rot180 + FOV scale), then add pitch/yaw.

  real  = undistort
  unity = rot180 -> scale(sx,sy) -> shift(dx,dy) -> H(pitch,yaw)

Compare NCC vs dxdy-only (zero angle).
"""
from __future__ import annotations

import argparse
import csv
import math
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
    return cv2.undistort(bgr, k, dist, None, new_k), new_k


def center_scale_shift(img, sx, sy, dx=0, dy=0):
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


def Rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def Ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def Rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def warp_rot(img, k, pitch, yaw, roll=0.0):
    if abs(pitch) < 1e-9 and abs(yaw) < 1e-9 and abs(roll) < 1e-9:
        return img
    r = Rz(roll) @ Ry(yaw) @ Rx(pitch)
    H = k @ r @ np.linalg.inv(k)
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sobel_mag(g):
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def zscore(a):
    m, s = float(a.mean()), float(a.std())
    return a * 0.0 if s < 1e-6 else (a - m) / s


def fft_ncc_best_shift(ref, mov):
    r, m = zscore(ref), zscore(mov)
    corr = np.fft.ifft2(np.fft.fft2(r) * np.conj(np.fft.fft2(m))).real / r.size
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(corr[iy, ix]), int(dy), int(dx)  # ncc, dy, dx


def edge_ncc(ref_bgr, mov_bgr):
    a = zscore(sobel_mag(to_gray(ref_bgr))).ravel()
    b = zscore(sobel_mag(to_gray(mov_bgr))).ravel()
    return float(np.dot(a, b) / a.size)


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


def apply_unity(unity, k, sx, sy, dx, dy, pitch, yaw):
    u = cv2.rotate(unity, cv2.ROTATE_180)
    u = center_scale_shift(u, sx, sy, dx, dy)
    u = warp_rot(u, k, pitch, yaw)
    return u


def make_panel(raw, real, u_dxdy, u_full, title, line2, out_path):
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    tiles = [
        (rs(raw), "1 real raw"),
        (rs(real), "2 real und"),
        (rs(u_dxdy), "3 unity FOV+dxdy"),
        (rs(u_full), "4 +pitch/yaw"),
        (yellow_cyan(rs(real), rs(u_dxdy)), "5 yc dxdy-only"),
        (yellow_cyan(rs(real), rs(u_full)), "6 yc dxdy+ang"),
    ]
    gap, header, footer = 6, 44, 70
    canvas = Image.new("RGB", (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)
    for i, (bgr, label) in enumerate(tiles):
        x = gap + i * (WORK_W + gap)
        canvas.paste(bgr_to_pil(bgr), (x, header))
        draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
    draw.text((gap, header + WORK_H + 28), line2, fill=(160, 200, 255), font=fs)
    d0 = float(np.mean(np.abs(rs(real).astype(np.float32) - rs(u_dxdy).astype(np.float32))))
    d1 = float(np.mean(np.abs(rs(real).astype(np.float32) - rs(u_full).astype(np.float32))))
    draw.text((gap, header + WORK_H + 46), f"|d| dxdy={d0:.1f}  dxdy+ang={d1:.1f}", fill=(255, 220, 120), font=fs)
    canvas.save(out_path)
    return d0, d1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "trainingData20260811")
    ap.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData20260811/TagUnity")
    ap.add_argument("--ids", type=str, default="1,2,3,4")
    ap.add_argument("--sx", type=float, default=0.75)
    ap.add_argument("--sy", type=float, default=0.85)
    ap.add_argument("--pitch-min", type=float, default=-6.0)
    ap.add_argument("--pitch-max", type=float, default=6.0)
    ap.add_argument("--yaw-min", type=float, default=-6.0)
    ap.add_argument("--yaw-max", type=float, default=6.0)
    ap.add_argument("--ang-step", type=float, default=0.5)
    ap.add_argument("--refine-step", type=float, default=0.25)
    ap.add_argument("--out-name", type=str, default="training20260811_tag_dxdy_then_pitch_yaw")
    args = ap.parse_args()

    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    k0, dist = load_intrinsics(ip)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    out = ROOT / "compare_out" / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    print("TAG: step1 FOV+dxdy; step2 add pitch/yaw (no extra translation)")
    print(f"FOV=({args.sx},{args.sy})  ang pitch[{args.pitch_min},{args.pitch_max}] yaw[{args.yaw_min},{args.yaw_max}]")
    print(f"OUT={out}")

    sx_w, sy_w = WORK_W / IMG_W, WORK_H / IMG_H
    pitches = np.arange(args.pitch_min, args.pitch_max + 0.5 * args.ang_step, args.ang_step)
    yaws = np.arange(args.yaw_min, args.yaw_max + 0.5 * args.ang_step, args.ang_step)
    rows = []

    for i in ids:
        rp = real_dir / f"Photo{i}_tag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"skip {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        real, new_k = undistort_bgr(raw, k0, dist)
        real_w = cv2.resize(real, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        k_w = new_k.copy()
        k_w[0, 0] *= sx_w
        k_w[1, 1] *= sy_w
        k_w[0, 2] *= sx_w
        k_w[1, 2] *= sy_w

        # step1: dx,dy after FOV scale (rot180 inside apply)
        u_scaled = center_scale_shift(cv2.rotate(unity_w, cv2.ROTATE_180), args.sx, args.sy, 0, 0)
        ncc1, dy1, dx1 = fft_ncc_best_shift(sobel_mag(to_gray(real_w)), sobel_mag(to_gray(u_scaled)))
        dx_f = int(round(dx1 * IMG_W / WORK_W))
        dy_f = int(round(dy1 * IMG_H / WORK_H))
        # work-res shift for search
        dx_w, dy_w = int(dx1), int(dy1)

        u_dxdy_w = apply_unity(unity_w, k_w, args.sx, args.sy, dx_w, dy_w, 0.0, 0.0)
        ncc_dxdy = edge_ncc(real_w, u_dxdy_w)

        # step2: pitch/yaw on top of FOV+dxdy
        best = {"ncc": ncc_dxdy, "pitch": 0.0, "yaw": 0.0}
        for pitch in pitches:
            for yaw in yaws:
                mov = apply_unity(unity_w, k_w, args.sx, args.sy, dx_w, dy_w, float(pitch), float(yaw))
                ncc = edge_ncc(real_w, mov)
                if ncc > best["ncc"]:
                    best = {"ncc": ncc, "pitch": float(pitch), "yaw": float(yaw)}

        for pitch in np.arange(best["pitch"] - 1.0, best["pitch"] + 1.0 + 0.5 * args.refine_step, args.refine_step):
            for yaw in np.arange(best["yaw"] - 1.0, best["yaw"] + 1.0 + 0.5 * args.refine_step, args.refine_step):
                mov = apply_unity(unity_w, k_w, args.sx, args.sy, dx_w, dy_w, float(pitch), float(yaw))
                ncc = edge_ncc(real_w, mov)
                if ncc > best["ncc"]:
                    best = {"ncc": ncc, "pitch": float(pitch), "yaw": float(yaw)}

        u_dxdy = apply_unity(unity, new_k, args.sx, args.sy, dx_f, dy_f, 0.0, 0.0)
        u_full = apply_unity(unity, new_k, args.sx, args.sy, dx_f, dy_f, best["pitch"], best["yaw"])
        d0, d1 = make_panel(
            raw,
            real,
            u_dxdy,
            u_full,
            f"TAG Photo{i}: FOV+dxdy then pitch/yaw",
            f"dx,dy=({dx_f:+d},{dy_f:+d})  pitch={best['pitch']:+.2f} yaw={best['yaw']:+.2f}  "
            f"NCC dxdy={ncc_dxdy:.3f} +ang={best['ncc']:.3f}",
            out / f"compare_{i}.png",
        )
        rows.append(
            {
                "id": i,
                "sx": args.sx,
                "sy": args.sy,
                "dx": dx_f,
                "dy": dy_f,
                "pitch_deg": f"{best['pitch']:.3f}",
                "yaw_deg": f"{best['yaw']:.3f}",
                "ncc_dxdy_only": f"{ncc_dxdy:.4f}",
                "ncc_dxdy_pitch_yaw": f"{best['ncc']:.4f}",
                "ncc_gain": f"{best['ncc'] - ncc_dxdy:.4f}",
                "mean_absdiff_dxdy": f"{d0:.2f}",
                "mean_absdiff_full": f"{d1:.2f}",
            }
        )
        print(
            f"[{i}] dx,dy=({dx_f:+d},{dy_f:+d})  pitch={best['pitch']:+.2f} yaw={best['yaw']:+.2f}  "
            f"NCC {ncc_dxdy:.3f} -> {best['ncc']:.3f} (gain {best['ncc']-ncc_dxdy:+.3f})  "
            f"|d| {d0:.1f} -> {d1:.1f}"
        )

    with (out / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    n0 = np.mean([float(r["ncc_dxdy_only"]) for r in rows])
    n1 = np.mean([float(r["ncc_dxdy_pitch_yaw"]) for r in rows])
    print(f"mean NCC dxdy-only     = {n0:.3f}")
    print(f"mean NCC dxdy+pitch/yaw= {n1:.3f}  gain={n1-n0:+.3f}")
    print(f"median pitch,yaw = ({np.median([float(r['pitch_deg']) for r in rows]):+.2f}, "
          f"{np.median([float(r['yaw_deg']) for r in rows]):+.2f})")
    print(f"median dx,dy = ({np.median([r['dx'] for r in rows]):+.0f}, {np.median([r['dy'] for r in rows]):+.0f})")
    print(f"summary: {out / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
