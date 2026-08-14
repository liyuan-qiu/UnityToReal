"""
Tag: NO dx/dy. Calibrate pitch (俯仰) + yaw (左右摆) via camera-rotation homography.
Then apply the same Unity-side warp to NoTag.

Pipeline (same side as tagged compare):
  real  = undistort(K, dist)
  unity = rot180
        -> optional FOV center_scale(sx, sy) with dx=dy=0
        -> warpPerspective(H), H = K @ R(yaw,pitch) @ K^{-1}

Usage:
  python compare_tag_pitch_yaw_then_notag.py
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
    k = np.asarray(d["camera_matrix"], np.float64)
    dist = np.asarray(d["dist_coeffs"], np.float64).reshape(-1)
    return k, dist


def undistort_bgr(bgr, k, dist):
    h, w = bgr.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, k, dist, None, new_k), new_k


def center_scale(img, sx, sy):
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


def Rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def Ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rot_homography(k: np.ndarray, pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """H = K R K^{-1} for pure camera rotation. R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    r = Rz(roll_deg) @ Ry(yaw_deg) @ Rx(pitch_deg)
    kinv = np.linalg.inv(k)
    return k @ r @ kinv


def warp_rot(img, k, pitch, yaw, roll=0.0):
    if abs(pitch) < 1e-9 and abs(yaw) < 1e-9 and abs(roll) < 1e-9:
        return img
    h, w = img.shape[:2]
    H = rot_homography(k, pitch, yaw, roll)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


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


def edge_ncc(ref_bgr, mov_bgr) -> float:
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


def apply_unity(unity, k_warp, sx, sy, pitch, yaw, roll=0.0):
    u = cv2.rotate(unity, cv2.ROTATE_180)
    u = center_scale(u, sx, sy)
    u = warp_rot(u, k_warp, pitch, yaw, roll)
    return u


def make_panel(raw, real, unity, title, line2, line3, out_path):
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    o = rs(real).astype(np.float32)
    u = rs(unity).astype(np.float32)
    mean_abs = float(np.mean(np.abs(o - u)))
    yc = yellow_cyan(rs(real), rs(unity))
    diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)
    tiles = [
        (rs(raw), "1 real raw"),
        (rs(real), "2 real undistort"),
        (rs(unity), "3 unity warped"),
        (yc, f"4 yellow/cyan |d|={mean_abs:.1f}"),
        (diff, "5 absdiff"),
        (rs(cv2.addWeighted(rs(real), 0.5, rs(unity), 0.5, 0)), "6 blend"),
    ]
    gap, header, footer = 6, 44, 70
    canvas = Image.new(
        "RGB",
        (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
        (22, 22, 22),
    )
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)
    for i, (bgr, label) in enumerate(tiles):
        x = gap + i * (WORK_W + gap)
        canvas.paste(bgr_to_pil(bgr), (x, header))
        draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
    draw.text((gap, header + WORK_H + 28), line2, fill=(160, 200, 255), font=fs)
    draw.text((gap, header + WORK_H + 46), line3 + f"  mean|d|={mean_abs:.2f}", fill=(255, 220, 120), font=fs)
    canvas.save(out_path)
    return mean_abs, canvas


def parse_args():
    p = argparse.ArgumentParser(description="Tag pitch/yaw calib (no dxdy), then apply to NoTag.")
    p.add_argument("--intrinsics", type=Path, default=ROOT / "trainingData/trainingData/capsule_intrinsics.npz")
    p.add_argument("--real-dir", type=Path, default=ROOT / "trainingData20260811")
    p.add_argument("--tag-unity-dir", type=Path, default=ROOT / "trainingData20260811/TagUnity")
    p.add_argument("--notag-unity-dir", type=Path, default=ROOT / "trainingData20260811/RealUnity")
    p.add_argument("--ids", type=str, default="1,2,3,4")
    p.add_argument("--sx", type=float, default=0.75, help="FOV scale on Unity (from tag); dx=dy=0")
    p.add_argument("--sy", type=float, default=0.85)
    p.add_argument("--pitch-min", type=float, default=-8.0)
    p.add_argument("--pitch-max", type=float, default=8.0)
    p.add_argument("--yaw-min", type=float, default=-8.0)
    p.add_argument("--yaw-max", type=float, default=8.0)
    p.add_argument("--ang-step", type=float, default=0.5)
    p.add_argument("--refine-step", type=float, default=0.25)
    p.add_argument("--out-name", type=str, default="training20260811_pitch_yaw")
    return p.parse_args()


def main():
    args = parse_args()
    ip = args.intrinsics if args.intrinsics.is_absolute() else ROOT / args.intrinsics
    real_dir = args.real_dir if args.real_dir.is_absolute() else ROOT / args.real_dir
    tag_u = args.tag_unity_dir if args.tag_unity_dir.is_absolute() else ROOT / args.tag_unity_dir
    notag_u = args.notag_unity_dir if args.notag_unity_dir.is_absolute() else ROOT / args.notag_unity_dir
    k, dist = load_intrinsics(ip)
    ids = [int(s) for s in args.ids.split(",") if s.strip()]

    out = ROOT / "compare_out" / args.out_name
    tag_out = out / "tag"
    notag_out = out / "notag"
    for d in (out, tag_out, notag_out):
        d.mkdir(parents=True, exist_ok=True)

    print("NO dx/dy. Search pitch+yaw on Unity after rot180+FOV scale.")
    print(f"FOV sx,sy=({args.sx},{args.sy})  pitch[{args.pitch_min},{args.pitch_max}] yaw[{args.yaw_min},{args.yaw_max}] step={args.ang_step}")
    print(f"OUT={out}")

    pitches = np.arange(args.pitch_min, args.pitch_max + 0.5 * args.ang_step, args.ang_step)
    yaws = np.arange(args.yaw_min, args.yaw_max + 0.5 * args.ang_step, args.ang_step)

    tag_rows = []
    # Use work-size K scaled for speed in search
    scale_x, scale_y = WORK_W / IMG_W, WORK_H / IMG_H

    for i in ids:
        rp = real_dir / f"Photo{i}_tag.jpg"
        up = tag_u / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"skip tag {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        real, new_k = undistort_bgr(raw, k, dist)
        # search at work res
        real_w = cv2.resize(real, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w0 = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        k_w = new_k.copy()
        k_w[0, 0] *= scale_x
        k_w[1, 1] *= scale_y
        k_w[0, 2] *= scale_x
        k_w[1, 2] *= scale_y

        best = {"ncc": -1e9, "pitch": 0.0, "yaw": 0.0}
        base = apply_unity(unity_w0, k_w, args.sx, args.sy, 0.0, 0.0)
        ncc0 = edge_ncc(real_w, base)

        for pitch in pitches:
            for yaw in yaws:
                mov = apply_unity(unity_w0, k_w, args.sx, args.sy, float(pitch), float(yaw))
                ncc = edge_ncc(real_w, mov)
                if ncc > best["ncc"]:
                    best = {"ncc": ncc, "pitch": float(pitch), "yaw": float(yaw)}

        # refine
        rp0 = np.arange(best["pitch"] - 1.0, best["pitch"] + 1.0 + 0.5 * args.refine_step, args.refine_step)
        ry0 = np.arange(best["yaw"] - 1.0, best["yaw"] + 1.0 + 0.5 * args.refine_step, args.refine_step)
        for pitch in rp0:
            for yaw in ry0:
                mov = apply_unity(unity_w0, k_w, args.sx, args.sy, float(pitch), float(yaw))
                ncc = edge_ncc(real_w, mov)
                if ncc > best["ncc"]:
                    best = {"ncc": ncc, "pitch": float(pitch), "yaw": float(yaw)}

        unity_full = apply_unity(unity, new_k, args.sx, args.sy, best["pitch"], best["yaw"])
        mean_abs, panel = make_panel(
            raw,
            real,
            unity_full,
            f"TAG Photo{i}: und | Unity rot180+FOV+pitch/yaw (NO dxdy)",
            f"pitch={best['pitch']:+.2f}deg  yaw={best['yaw']:+.2f}deg  FOV=({args.sx},{args.sy})  NCC={best['ncc']:.3f} (0ang={ncc0:.3f})",
            "H=K R(yaw,pitch) K^-1 on Unity",
            tag_out / f"compare_{i}.png",
        )
        tag_rows.append(
            {
                "id": i,
                "pitch_deg": f"{best['pitch']:.3f}",
                "yaw_deg": f"{best['yaw']:.3f}",
                "sx": args.sx,
                "sy": args.sy,
                "dx": 0,
                "dy": 0,
                "edgeNCC": f"{best['ncc']:.4f}",
                "ncc_zero_angle": f"{ncc0:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
            }
        )
        print(
            f"[tag {i}] pitch={best['pitch']:+.2f} yaw={best['yaw']:+.2f}  "
            f"NCC={best['ncc']:.3f} (zeroAng={ncc0:.3f})  |d|={mean_abs:.1f}"
        )

    if not tag_rows:
        raise SystemExit("No tag pairs")

    med_pitch = float(np.median([float(r["pitch_deg"]) for r in tag_rows]))
    med_yaw = float(np.median([float(r["yaw_deg"]) for r in tag_rows]))
    print(f"\nTAG median pitch,yaw = ({med_pitch:+.2f}, {med_yaw:+.2f}) deg")

    with (tag_out / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tag_rows[0].keys()))
        w.writeheader()
        w.writerows(tag_rows)

    # ---- NoTag with median pitch/yaw on Unity ----
    print("\nApply median pitch/yaw to NoTag (Unity side)...")
    notag_rows = []
    thumbs = []
    for i in ids:
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = notag_u / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"skip notag {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        real, new_k = undistort_bgr(raw, k, dist)
        unity_w = apply_unity(unity, new_k, args.sx, args.sy, med_pitch, med_yaw)

        # also per-image tag angles for reference panel variant? use median for transfer test
        mean_abs, panel = make_panel(
            raw,
            real,
            unity_w,
            f"NoTag Photo{i}: und | Unity rot180+FOV+median pitch/yaw",
            f"from TAG median pitch={med_pitch:+.2f} yaw={med_yaw:+.2f}  FOV=({args.sx},{args.sy})  NO dxdy",
            "same warp side as tag compare",
            notag_out / f"compare_{i}.png",
        )
        ncc = edge_ncc(
            cv2.resize(real, (WORK_W, WORK_H)),
            cv2.resize(unity_w, (WORK_W, WORK_H)),
        )
        # baseline: FOV only, zero angle
        unity0 = apply_unity(unity, new_k, args.sx, args.sy, 0.0, 0.0)
        mean0 = float(
            np.mean(
                np.abs(
                    cv2.resize(real, (WORK_W, WORK_H)).astype(np.float32)
                    - cv2.resize(unity0, (WORK_W, WORK_H)).astype(np.float32)
                )
            )
        )
        ncc0 = edge_ncc(cv2.resize(real, (WORK_W, WORK_H)), cv2.resize(unity0, (WORK_W, WORK_H)))

        notag_rows.append(
            {
                "id": i,
                "pitch_deg": f"{med_pitch:.3f}",
                "yaw_deg": f"{med_yaw:.3f}",
                "sx": args.sx,
                "sy": args.sy,
                "edgeNCC": f"{ncc:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
                "ncc_zero_angle": f"{ncc0:.4f}",
                "mean_absdiff_zero_angle": f"{mean0:.2f}",
            }
        )
        thumbs.append(panel.resize((panel.width // 3, panel.height // 3), Image.Resampling.BILINEAR))
        print(
            f"[notag {i}] pitch/yaw=({med_pitch:+.2f},{med_yaw:+.2f})  "
            f"NCC={ncc:.3f} (0ang={ncc0:.3f})  |d|={mean_abs:.1f} (0ang={mean0:.1f})"
        )

    with (notag_out / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(notag_rows[0].keys()))
        w.writeheader()
        w.writerows(notag_rows)

    if thumbs:
        tw, th = thumbs[0].size
        triage = Image.new("RGB", (tw, th * len(thumbs) + 8 * (len(thumbs) + 1)), (18, 18, 18))
        y = 8
        for t in thumbs:
            triage.paste(t, (0, y))
            y += th + 8
        triage.save(notag_out / "triage_all.png")

    # also per-image pitch/yaw transfer (each tag angle -> same id notag)
    print("\nPer-image pitch/yaw transfer (tag i -> notag i)...")
    per_rows = []
    per_out = out / "notag_per_image_angle"
    per_out.mkdir(exist_ok=True)
    ang_map = {int(r["id"]): (float(r["pitch_deg"]), float(r["yaw_deg"])) for r in tag_rows}
    for i in ids:
        if i not in ang_map:
            continue
        pitch, yaw = ang_map[i]
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = notag_u / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        real, new_k = undistort_bgr(raw, k, dist)
        unity_w = apply_unity(unity, new_k, args.sx, args.sy, pitch, yaw)
        mean_abs, _ = make_panel(
            raw,
            real,
            unity_w,
            f"NoTag Photo{i}: per-image tag pitch/yaw",
            f"pitch={pitch:+.2f} yaw={yaw:+.2f} FOV=({args.sx},{args.sy}) NO dxdy",
            "from same-id tag calib",
            per_out / f"compare_{i}.png",
        )
        ncc = edge_ncc(cv2.resize(real, (WORK_W, WORK_H)), cv2.resize(unity_w, (WORK_W, WORK_H)))
        per_rows.append(
            {
                "id": i,
                "pitch_deg": f"{pitch:.3f}",
                "yaw_deg": f"{yaw:.3f}",
                "edgeNCC": f"{ncc:.4f}",
                "mean_absdiff": f"{mean_abs:.2f}",
            }
        )
        print(f"[notag-per {i}] pitch/yaw=({pitch:+.2f},{yaw:+.2f}) NCC={ncc:.3f} |d|={mean_abs:.1f}")

    with (per_out / "compare_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    print("\n=== SUMMARY ===")
    print(f"TAG median pitch,yaw = ({med_pitch:+.2f}, {med_yaw:+.2f}) deg  (dx=dy=0)")
    print(f"TAG mean NCC = {np.mean([float(r['edgeNCC']) for r in tag_rows]):.3f}")
    print(f"NoTag median-angle mean NCC = {np.mean([float(r['edgeNCC']) for r in notag_rows]):.3f}  "
          f"|d|={np.mean([float(r['mean_absdiff']) for r in notag_rows]):.1f}")
    print(f"NoTag zero-angle   mean NCC = {np.mean([float(r['ncc_zero_angle']) for r in notag_rows]):.3f}  "
          f"|d|={np.mean([float(r['mean_absdiff_zero_angle']) for r in notag_rows]):.1f}")
    print(f"NoTag per-image    mean NCC = {np.mean([float(r['edgeNCC']) for r in per_rows]):.3f}  "
          f"|d|={np.mean([float(r['mean_absdiff']) for r in per_rows]):.1f}")
    print(f"Prev dxdy unityWarp NoTag mean |d| was ~44.3")
    print(f"Outputs under {out}")


if __name__ == "__main__":
    main()
