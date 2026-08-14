"""
Tag roll calibration (in-plane rotation about image center).

  real  = undistort(K, dist)
  unity = rot180 -> FOV scale(sx,sy) -> optional dx,dy -> rotate(roll)

Searches roll that maximizes edge-NCC. Optionally refine dx,dy after roll.
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


def rotate_img(img, deg):
    if abs(deg) < 1e-9:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


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


def fft_ncc_best_shift(ref, mov):
    r, m = zscore(ref), zscore(mov)
    corr = np.fft.ifft2(np.fft.fft2(r) * np.conj(np.fft.fft2(m))).real / r.size
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(corr[iy, ix]), int(dy), int(dx)


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


def apply_unity(unity, sx, sy, dx, dy, roll):
    """rot180 -> FOV scale -> roll (about center) -> dx,dy shift."""
    u = cv2.rotate(unity, cv2.ROTATE_180)
    u = center_scale_shift(u, sx, sy, 0, 0)
    u = rotate_img(u, roll)
    if dx or dy:
        u = center_scale_shift(u, 1.0, 1.0, dx, dy)
    return u


def parse_args():
    p = argparse.ArgumentParser(description="Tag roll calibration.")
    p.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData20260811")
    p.add_argument("--unity-dir", type=Path, default=ROOT / "trainingData20260811/tagUnity1")
    p.add_argument("--ids", type=str, default="1,2,3,4")
    p.add_argument("--sx", type=float, default=0.75)
    p.add_argument("--sy", type=float, default=0.825)
    p.add_argument("--roll-min", type=float, default=-30.0)
    p.add_argument("--roll-max", type=float, default=30.0)
    p.add_argument("--roll-step", type=float, default=1.0)
    p.add_argument("--refine-step", type=float, default=0.25)
    p.add_argument("--search-dxdy", action="store_true", default=True, help="FFT dxdy before roll (default on)")
    p.add_argument("--no-search-dxdy", action="store_true")
    p.add_argument("--refine-dxdy-after-roll", action="store_true", default=True)
    p.add_argument("--out-name", type=str, default="training20260811_tagUnity1_roll")
    return p.parse_args()


def main():
    args = parse_args()
    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    unity_dir = args.unity_dir if args.unity_dir.is_absolute() else ROOT / args.unity_dir
    k, dist = load_intrinsics(ip)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    do_dxdy = bool(args.search_dxdy) and not args.no_search_dxdy

    out = ROOT / "compare_out" / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"TAG roll calib | FOV=({args.sx},{args.sy}) search_dxdy={do_dxdy}")
    print(f"roll [{args.roll_min},{args.roll_max}] step={args.roll_step}")
    print(f"OUT={out}")

    rolls = np.arange(args.roll_min, args.roll_max + 0.5 * args.roll_step, args.roll_step)
    rows = []
    thumbs = []

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

        real = undistort_bgr(raw, k, dist)
        real_w = cv2.resize(real, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(real_w))

        # baseline: FOV (+ optional dxdy), roll=0
        dx_w = dy_w = 0
        if do_dxdy:
            base = center_scale_shift(cv2.rotate(unity_w, cv2.ROTATE_180), args.sx, args.sy, 0, 0)
            _, dy_w, dx_w = fft_ncc_best_shift(ref_e, sobel_mag(to_gray(base)))

        u0 = apply_unity(unity_w, args.sx, args.sy, dx_w, dy_w, 0.0)
        ncc0 = edge_ncc(real_w, u0)

        best = {"ncc": ncc0, "roll": 0.0}
        for roll in rolls:
            mov = apply_unity(unity_w, args.sx, args.sy, dx_w, dy_w, float(roll))
            ncc = edge_ncc(real_w, mov)
            if ncc > best["ncc"]:
                best = {"ncc": ncc, "roll": float(roll)}

        # refine roll
        for roll in np.arange(best["roll"] - 2.0, best["roll"] + 2.0 + 0.5 * args.refine_step, args.refine_step):
            mov = apply_unity(unity_w, args.sx, args.sy, dx_w, dy_w, float(roll))
            ncc = edge_ncc(real_w, mov)
            if ncc > best["ncc"]:
                best = {"ncc": ncc, "roll": float(roll)}

        # optional: re-solve dxdy after roll
        dx2, dy2 = dx_w, dy_w
        if args.refine_dxdy_after_roll and do_dxdy:
            u_rs = rotate_img(
                center_scale_shift(cv2.rotate(unity_w, cv2.ROTATE_180), args.sx, args.sy, 0, 0),
                best["roll"],
            )
            ncc_s, dy2, dx2 = fft_ncc_best_shift(ref_e, sobel_mag(to_gray(u_rs)))
            mov = center_scale_shift(u_rs, 1.0, 1.0, dx2, dy2)
            ncc_final = edge_ncc(real_w, mov)
            if ncc_final >= best["ncc"]:
                best["ncc"] = ncc_final
            else:
                dx2, dy2 = dx_w, dy_w

        dx_f = int(round(dx2 * IMG_W / WORK_W))
        dy_f = int(round(dy2 * IMG_H / WORK_H))

        u_base = center_scale_shift(cv2.rotate(unity, cv2.ROTATE_180), args.sx, args.sy, 0, 0)
        u_noroll = center_scale_shift(u_base, 1.0, 1.0, dx_f, dy_f)
        u_full = center_scale_shift(rotate_img(u_base, best["roll"]), 1.0, 1.0, dx_f, dy_f)

        def rs(im):
            return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        ncc_nr = edge_ncc(rs(real), rs(u_noroll))
        ncc_full = edge_ncc(rs(real), rs(u_full))
        # if roll does not help final metric, keep roll=0 for report/panel
        if ncc_full < ncc_nr - 1e-6:
            best["roll"] = 0.0
            u_full = u_noroll
            ncc_full = ncc_nr
        d0 = float(np.mean(np.abs(rs(real).astype(np.float32) - rs(u_noroll).astype(np.float32))))
        d1 = float(np.mean(np.abs(rs(real).astype(np.float32) - rs(u_full).astype(np.float32))))

        tiles = [
            (rs(raw), "1 real raw"),
            (rs(real), "2 real und"),
            (rs(u_noroll), "3 unity FOV+dxdy roll=0"),
            (rs(u_full), f"4 +roll={best['roll']:+.2f}"),
            (yellow_cyan(rs(real), rs(u_noroll)), f"5 yc no-roll |d|={d0:.1f}"),
            (yellow_cyan(rs(real), rs(u_full)), f"6 yc +roll |d|={d1:.1f}"),
        ]
        gap, header, footer = 6, 44, 70
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        f, fs = font(15), font(12)
        draw.text((gap, 10), f"TAG Photo{i}: roll calibration (in-plane)", fill=(240, 240, 240), font=f)
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"FOV=({args.sx},{args.sy}) dx,dy=({dx_f:+d},{dy_f:+d})  roll={best['roll']:+.2f}deg  "
            f"NCC {ncc_nr:.3f} -> {ncc_full:.3f} (gain {ncc_full-ncc_nr:+.3f})",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 46),
            f"|d| {d0:.1f} -> {d1:.1f}   search_ncc0={ncc0:.3f} best_search={best['ncc']:.3f}",
            fill=(255, 220, 120),
            font=fs,
        )
        panel = out / f"compare_{i}.png"
        canvas.save(panel)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))

        rows.append(
            {
                "id": i,
                "sx": args.sx,
                "sy": args.sy,
                "dx": dx_f,
                "dy": dy_f,
                "roll_deg": f"{best['roll']:.3f}",
                "ncc_roll0": f"{ncc_nr:.4f}",
                "ncc_with_roll": f"{ncc_full:.4f}",
                "ncc_gain": f"{ncc_full - ncc_nr:.4f}",
                "mean_absdiff_roll0": f"{d0:.2f}",
                "mean_absdiff_with_roll": f"{d1:.2f}",
            }
        )
        print(
            f"[{i}] roll={best['roll']:+.2f}deg  dx,dy=({dx_f:+d},{dy_f:+d})  "
            f"NCC {ncc_nr:.3f} -> {ncc_full:.3f} (gain {ncc_full-ncc_nr:+.3f})  "
            f"|d| {d0:.1f} -> {d1:.1f}"
        )

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

    rolls = [float(r["roll_deg"]) for r in rows]
    print("\n=== SUMMARY ===")
    print(f"rolls = {[round(x, 2) for x in rolls]}")
    print(f"median roll = {np.median(rolls):+.2f} deg")
    print(f"mean NCC roll0 = {np.mean([float(r['ncc_roll0']) for r in rows]):.3f}")
    print(f"mean NCC +roll = {np.mean([float(r['ncc_with_roll']) for r in rows]):.3f}")
    print(f"mean gain     = {np.mean([float(r['ncc_gain']) for r in rows]):+.3f}")
    print(f"summary: {out / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
