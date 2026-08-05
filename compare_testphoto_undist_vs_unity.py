"""
1) Undistort TestPhoto with calibrated K + Brown-Conrady dist
2) Optional post-undistort center scale (FOV residual from tag compare)
3) Compare vs Unity captures in camera_pose_unity_facing_xyz/
4) Write panels under compare_out/<out-name>/

Tag FOV residual (Unity shrink sx,sy≈0.77,0.82 to match und) =>
  enlarge und by (1/sx, 1/sy) ≈ (1.30, 1.23):

  python compare_testphoto_undist_vs_unity.py --scale-x 1.300 --scale-y 1.227 \\
      --out-name testphoto_undist_scaled_vs_unity
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
REAL_DIR = ROOT / "TestPhoto"
UNITY_DIR = ROOT / "camera_pose_unity_facing_xyz"
OUT_DIR = ROOT / "compare_out" / "testphoto_undist_vs_unity"
UNDIST_DIR = OUT_DIR / "undistorted_testphoto"
UNDIST_SX, UNDIST_SY = 1.0, 1.0

# Calibrated capsule intrinsics (user-provided)
K = np.array(
    [
        [762.7627033, 0.0, 661.53817354],
        [0.0, 763.78023472, 360.37587777],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# OpenCV order: k1, k2, p1, p2, k3
DIST = np.array(
    [-0.38898088, 0.15099531, -0.00301529, 0.00057045, -0.02746219],
    dtype=np.float64,
)

IMG_W, IMG_H = 1080, 720
WORK_W, WORK_H = 540, 360
IDS = list(range(1, 14))
TRY_ROTS = (0, 180)
SX_MIN, SX_MAX, SX_STEPS = 0.75, 1.40, 14
SY_MIN, SY_MAX, SY_STEPS = 0.75, 1.40, 14


def undistort_bgr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, DIST, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, K, DIST, None, new_K), new_K


def center_scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale about center; keep original HxW (crop/pad)."""
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + nw), min(h, y0 + nh)
    ms0, ns0 = xs0 - x0, ys0 - y0
    ms1, ns1 = ms0 + (xs1 - xs0), ns0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0:ns1, ms0:ms1]
    return canvas


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
    return float(corr[iy, ix]), int(dy), int(dx)


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
    ms1, ns1 = ms0 + (xs1 - xs0), ns0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0:ns1, ms0:ms1]
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
    ms1, ns1 = ms0 + (xs1 - xs0), ns0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = scaled[ns0:ns1, ms0:ms1]
    return canvas


def search_best(ref_e: np.ndarray, mov_e: np.ndarray) -> dict:
    best = {"ncc": -1e9, "sx": 1.0, "sy": 1.0, "dx": 0, "dy": 0}
    for sx in np.linspace(SX_MIN, SX_MAX, SX_STEPS):
        for sy in np.linspace(SY_MIN, SY_MAX, SY_STEPS):
            scaled = warp_gray(mov_e, float(sx), float(sy), 0, 0)
            ncc, dy, dx = fft_ncc_best_shift(ref_e, scaled)
            if ncc > best["ncc"]:
                best = {"ncc": ncc, "sx": float(sx), "sy": float(sy), "dx": int(dx), "dy": int(dy)}
    aa, bb = zscore(ref_e).ravel(), zscore(mov_e).ravel()
    best["ncc_nowarp"] = float(np.dot(aa, bb) / aa.size)
    return best


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_panel(
    raw: np.ndarray,
    und: np.ndarray,
    unity: np.ndarray,
    unity180: np.ndarray,
    unity_aligned: np.ndarray,
    params: dict,
    title: str,
) -> Image.Image:
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    o = rs(und).astype(np.float32)
    u = rs(unity_aligned).astype(np.float32)
    blend = np.clip(0.5 * o + 0.5 * u, 0, 255).astype(np.uint8)
    diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)
    mean_abs = float(np.mean(np.abs(o - u)))
    params["mean_absdiff"] = mean_abs

    tiles = [
        (rs(raw), "1 real raw"),
        (rs(und), "2 real undistorted"),
        (rs(unity), "3 unity raw"),
        (rs(unity180), "4 unity rot180"),
        (blend, f"5 blend und+unity@{params['rot']}"),
        (diff, f"6 absdiff (mean={mean_abs:.1f})"),
    ]
    gap, header, footer = 6, 44, 78
    cols = len(tiles)
    canvas = Image.new(
        "RGB",
        (cols * WORK_W + (cols + 1) * gap, header + WORK_H + footer),
        (22, 22, 22),
    )
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)
    for i, (bgr, label) in enumerate(tiles):
        x = gap + i * (WORK_W + gap)
        canvas.paste(bgr_to_pil(bgr), (x, header))
        draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)

    line1 = (
        f"calib K fx,fy=({K[0,0]:.1f},{K[1,1]:.1f}) cx,cy=({K[0,2]:.1f},{K[1,2]:.1f})  "
        f"post-undist scale sx,sy=({UNDIST_SX:.3f},{UNDIST_SY:.3f})"
    )
    line2 = (
        f"best rot={params['rot']}  sx,sy=({params['sx']:.3f},{params['sy']:.3f})  "
        f"dx,dy_full=({params['dx_full']:+d},{params['dy_full']:+d})px  "
        f"edgeNCC={params['ncc']:.4f}  nowarp={params['ncc_nowarp']:.4f}  "
        f"mean|und-unity|={mean_abs:.2f}"
    )
    line3 = "Order: undistort TestPhoto first, then compare to Unity. absdiff = |undistorted_real - aligned_unity|"
    draw.text((gap, header + WORK_H + 28), line1, fill=(160, 200, 255), font=fs)
    draw.text((gap, header + WORK_H + 46), line2, fill=(255, 220, 120), font=fs)
    draw.text((gap, header + WORK_H + 64), line3, fill=(180, 180, 180), font=fs)
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag: undistort TestPhoto (+optional scale) vs Unity.")
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Isotropic post-undistort scale on real (e.g. 1.26 = 1/0.79 from tag).",
    )
    p.add_argument("--scale-x", type=float, default=None, help="Post-undistort X scale on real.")
    p.add_argument("--scale-y", type=float, default=None, help="Post-undistort Y scale on real.")
    p.add_argument(
        "--from-tag-summary",
        type=Path,
        default=None,
        help="Read mean Unity sx,sy from a previous compare_summary.csv and use 1/sx,1/sy on und.",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Folder under compare_out/ (default: testphoto_undist_vs_unity or *_scaled_*).",
    )
    p.add_argument(
        "--fix-search-scale",
        action="store_true",
        help="After baking scale into und, only search shift (sx=sy=1) to measure residual.",
    )
    return p.parse_args()


def scales_from_args(args: argparse.Namespace) -> tuple[float, float]:
    if args.from_tag_summary is not None:
        path = args.from_tag_summary
        if not path.is_file():
            path = ROOT / path
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        mx = float(np.mean([float(r["sx"]) for r in rows]))
        my = float(np.mean([float(r["sy"]) for r in rows]))
        sx, sy = 1.0 / mx, 1.0 / my
        print(f"from summary {path.name}: Unity mean sx,sy=({mx:.4f},{my:.4f}) -> und scale ({sx:.4f},{sy:.4f})")
        return sx, sy
    if args.scale_x is not None or args.scale_y is not None:
        sx = float(args.scale_x if args.scale_x is not None else (args.scale or 1.0))
        sy = float(args.scale_y if args.scale_y is not None else (args.scale or 1.0))
        return sx, sy
    if args.scale is not None:
        return float(args.scale), float(args.scale)
    return 1.0, 1.0


def main() -> None:
    global OUT_DIR, UNDIST_DIR, UNDIST_SX, UNDIST_SY, SX_MIN, SX_MAX, SY_MIN, SY_MAX, SX_STEPS, SY_STEPS
    args = parse_args()
    UNDIST_SX, UNDIST_SY = scales_from_args(args)
    scaled = abs(UNDIST_SX - 1.0) > 1e-6 or abs(UNDIST_SY - 1.0) > 1e-6
    out_name = args.out_name.strip() or (
        "testphoto_undist_scaled_vs_unity" if scaled else "testphoto_undist_vs_unity"
    )
    OUT_DIR = ROOT / "compare_out" / out_name
    UNDIST_DIR = OUT_DIR / "undistorted_testphoto"

    if args.fix_search_scale:
        SX_MIN = SX_MAX = SY_MIN = SY_MAX = 1.0
        SX_STEPS = SY_STEPS = 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    UNDIST_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = ROOT / "capsule_intrinsics.npz"
    np.savez(npz_path, camera_matrix=K, dist_coeffs=DIST)
    print(f"Wrote {npz_path}")
    print(f"OUT = {OUT_DIR}")
    print(f"post-undistort scale sx,sy=({UNDIST_SX:.4f},{UNDIST_SY:.4f})")
    print(f"Unity scale search: sx[{SX_MIN},{SX_MAX}] sy[{SY_MIN},{SY_MAX}] steps={SX_STEPS}")

    rows = []
    thumbs = []

    for i in IDS:
        rp = REAL_DIR / f"CamCoordTest_{i}.jpg"
        up = UNITY_DIR / f"CamCoordTest_{i}_Unity.jpg"
        if not rp.exists() or not up.exists():
            print(f"skip {i}")
            continue

        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"fail read {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        # 1) undistort + optional FOV scale
        und, new_K = undistort_bgr(raw)
        und = center_scale(und, UNDIST_SX, UNDIST_SY)
        und_path = UNDIST_DIR / f"CamCoordTest_{i}_undist.jpg"
        cv2.imwrite(str(und_path), und)

        # 2) compare undistorted(+scaled) real vs Unity
        und_w = cv2.resize(und, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(und_w))

        best = None
        scores = {}
        for rot in TRY_ROTS:
            mov = unity_w if rot == 0 else cv2.rotate(unity_w, cv2.ROTATE_180)
            cand = search_best(ref_e, sobel_mag(to_gray(mov)))
            cand["rot"] = rot
            scores[rot] = cand
            if best is None or cand["ncc"] > best["ncc"]:
                best = cand
        assert best is not None

        unity180 = cv2.rotate(unity, cv2.ROTATE_180)
        unity_best = unity if best["rot"] == 0 else unity180
        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        best["dx_full"] = dx_f
        best["dy_full"] = dy_f
        unity_aligned = warp_bgr(unity_best, best["sx"], best["sy"], dx_f, dy_f)

        o_full = und.astype(np.float32)
        u_full = unity_aligned.astype(np.float32)
        mean_abs_full = float(np.mean(np.abs(o_full - u_full)))

        panel = make_panel(
            raw,
            und,
            unity,
            unity180,
            unity_aligned,
            best,
            f"CamCoordTest_{i}: undistort+scale({UNDIST_SX:.2f},{UNDIST_SY:.2f}) vs Unity",
        )
        panel_path = OUT_DIR / f"compare_{i}.png"
        panel.save(panel_path)
        thumbs.append(panel.resize((panel.width // 3, panel.height // 3), Image.Resampling.BILINEAR))

        o = cv2.resize(und, (WORK_W, WORK_H)).astype(np.float32)
        u = cv2.resize(unity_aligned, (WORK_W, WORK_H)).astype(np.float32)
        diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / f"{i}_absdiff.png"), diff)

        row = {
            "id": i,
            "undist_scale_x": f"{UNDIST_SX:.4f}",
            "undist_scale_y": f"{UNDIST_SY:.4f}",
            "rot": best["rot"],
            "sx": f"{best['sx']:.4f}",
            "sy": f"{best['sy']:.4f}",
            "dx_full": dx_f,
            "dy_full": dy_f,
            "edgeNCC": f"{best['ncc']:.4f}",
            "ncc_nowarp": f"{best['ncc_nowarp']:.4f}",
            "edgeNCC_rot0": f"{scores[0]['ncc']:.4f}",
            "edgeNCC_rot180": f"{scores[180]['ncc']:.4f}",
            "mean_absdiff_full": f"{mean_abs_full:.2f}",
            "mean_absdiff_work": f"{best.get('mean_absdiff', float(np.mean(np.abs(o - u)))):.2f}",
            "undist": und_path.name,
            "panel": panel_path.name,
            "new_fx": f"{new_K[0,0]:.4f}",
            "new_fy": f"{new_K[1,1]:.4f}",
            "new_cx": f"{new_K[0,2]:.4f}",
            "new_cy": f"{new_K[1,2]:.4f}",
        }
        rows.append(row)
        print(
            f"[{i:2d}] und->Unity  best_rot={best['rot']:3d}  "
            f"NCC={best['ncc']:.3f} (r0={scores[0]['ncc']:.3f}, r180={scores[180]['ncc']:.3f})  "
            f"sx,sy=({best['sx']:.2f},{best['sy']:.2f})  "
            f"dx,dy=({dx_f:+d},{dy_f:+d})  mean|diff|={mean_abs_full:.1f}"
        )

    if not rows:
        print("No pairs.")
        return

    csv_path = OUT_DIR / "compare_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
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
        triage.save(OUT_DIR / "triage_all.png")

    nccs = [float(r["edgeNCC"]) for r in rows]
    diffs = [float(r["mean_absdiff_full"]) for r in rows]
    sxs = [float(r["sx"]) for r in rows]
    sys_ = [float(r["sy"]) for r in rows]
    winners = [int(r["rot"]) for r in rows]
    print("\n=== SUMMARY ===")
    print(f"output: {OUT_DIR}")
    print(f"undistorted reals: {UNDIST_DIR}")
    print(f"post-undist scale sx,sy=({UNDIST_SX:.4f},{UNDIST_SY:.4f})")
    print(f"best_rot counts: 0={winners.count(0)}  180={winners.count(180)}")
    print(f"edgeNCC mean={np.mean(nccs):.3f} median={np.median(nccs):.3f}")
    print(f"residual Unity sx,sy mean=({np.mean(sxs):.3f},{np.mean(sys_):.3f})")
    print(f"mean|absdiff| mean={np.mean(diffs):.1f} median={np.median(diffs):.1f}")
    print(f"summary csv: {csv_path}")


if __name__ == "__main__":
    main()

