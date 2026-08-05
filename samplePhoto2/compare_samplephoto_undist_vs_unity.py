"""
REAL photos (NOT tag):
  1) Undistort samplePhoto2/{id}.jpg with capsule K + Brown-Conrady
  2) Optional post-undistort scale (same FOV residual as tag)
  3) Compare vs <unity_dir>/{id}_Unity.jpg  (no rot180)
  4) Write panels under ../compare_out/<out_name>/

Examples:
  python compare_samplephoto_undist_vs_unity.py --unity-dir unitySamplePhoto3
  python compare_samplephoto_undist_vs_unity.py --unity-dir unitySamplePhoto3 \\
      --from-tag-summary ../compare_out/testphoto_undist_vs_unity/compare_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REAL_DIR = HERE
UNITY_DIR = ROOT / "unitysamplephoto2"
OUT_DIR = ROOT / "compare_out" / "samplephoto2_undist_vs_unity"
UNDIST_DIR = OUT_DIR / "undistorted_real"
UNDIST_SX, UNDIST_SY = 1.0, 1.0

K = np.array(
    [
        [762.7627033, 0.0, 661.53817354],
        [0.0, 763.78023472, 360.37587777],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST = np.array(
    [-0.38898088, 0.15099531, -0.00301529, 0.00057045, -0.02746219],
    dtype=np.float64,
)

IMG_W, IMG_H = 1080, 720
WORK_W, WORK_H = 540, 360
# photo names matching coordinate2 / Unity export (Baseline may only exist on Unity side)
IDS = ["Baseline", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Real photos: do NOT rotate Unity 180 (unlike tag triage which often preferred rot180).
TRY_ROTS = (0,)
SX_MIN, SX_MAX, SX_STEPS = 0.75, 1.40, 14
SY_MIN, SY_MAX, SY_STEPS = 0.75, 1.40, 14


def undistort_bgr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, DIST, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, K, DIST, None, new_K), new_K


def center_scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
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


def yellow_cyan_overlay(und_bgr: np.ndarray, unity_bgr: np.ndarray) -> np.ndarray:
    """
    False-color registration view (OpenCV BGR):
      samplePhoto2 / und -> pure YELLOW  BGR=(0, g, g)
      Unity               -> pure CYAN    BGR=(g, g, 0)
      overlay = clip(yellow + cyan) = BGR=(g_u, g_und+g_u, g_und)
    Aligned ~ pale/white; mismatch ~ yellow or cyan fringe (not tissue-red).
    """
    g_und = cv2.cvtColor(und_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_u = cv2.cvtColor(unity_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def stretch(g):
        lo, hi = np.percentile(g, (2, 98))
        if hi <= lo + 1e-3:
            return g
        return np.clip((g - lo) * (255.0 / (hi - lo)), 0, 255)

    g_und = stretch(g_und)
    g_u = stretch(g_u)
    out = np.zeros((*g_und.shape, 3), dtype=np.float32)
    out[..., 0] = g_u  # B: Unity only
    out[..., 1] = np.clip(g_und + g_u, 0, 255)  # G: both (yellow + cyan)
    out[..., 2] = g_und  # R: Real only -> Real reads as yellow
    return out.astype(np.uint8)


def absdiff_heat(und_bgr: np.ndarray, unity_bgr: np.ndarray) -> np.ndarray:
    """|gray und - gray unity| as JET heatmap (easier than raw RGB absdiff)."""
    g0 = cv2.cvtColor(und_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(unity_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.abs(g0 - g1)
    d = np.clip(d * (255.0 / max(float(d.max()), 1.0)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


def make_panel(raw, und, unity, unity_aligned, params, title) -> Image.Image:
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    o = rs(und)
    u = rs(unity_aligned)
    o_f = o.astype(np.float32)
    u_f = u.astype(np.float32)
    blend = np.clip(0.5 * o_f + 0.5 * u_f, 0, 255).astype(np.uint8)
    yc = yellow_cyan_overlay(o, u)
    heat = absdiff_heat(o, u)
    mean_abs = float(np.mean(np.abs(o_f - u_f)))
    params["mean_absdiff"] = mean_abs

    tiles = [
        (rs(raw), "1 real raw"),
        (rs(und), "2 real undist+scale"),
        (rs(unity), "3 unity (no rot)"),
        (blend, "4 blend gray"),
        (yc, "5 YELLOW=real / CYAN=unity"),
        (heat, f"6 |diff| heat (meanRGB={mean_abs:.1f})"),
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
    line3 = (
        f"Diff colors: samplePhoto2=YELLOW, {UNITY_DIR.name}=CYAN; "
        "aligned~pale, mismatch~yellow/cyan fringe. Panel6=JET |gray diff|."
    )
    draw.text((gap, header + WORK_H + 28), line1, fill=(160, 200, 255), font=fs)
    draw.text((gap, header + WORK_H + 46), line2, fill=(255, 220, 120), font=fs)
    draw.text((gap, header + WORK_H + 64), line3, fill=(180, 180, 180), font=fs)
    return canvas


def find_real(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = REAL_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_unity(stem: str) -> Path | None:
    for name in (f"{stem}_Unity.jpg", f"{stem}_Unity.png", f"{stem}.jpg", f"{stem}.png"):
        p = UNITY_DIR / name
        if p.exists():
            return p
    return None


def scales_from_args(args: argparse.Namespace) -> tuple[float, float]:
    if args.from_tag_summary is not None:
        path = Path(args.from_tag_summary)
        if not path.is_file():
            path = ROOT / path
        if not path.is_file():
            path = HERE / path
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        mx = float(np.mean([float(r["sx"]) for r in rows]))
        my = float(np.mean([float(r["sy"]) for r in rows]))
        sx, sy = 1.0 / mx, 1.0 / my
        print(
            f"from tag summary {path.name}: Unity mean sx,sy=({mx:.4f},{my:.4f}) "
            f"-> und scale ({sx:.4f},{sy:.4f})"
        )
        return sx, sy
    if args.scale_x is not None or args.scale_y is not None:
        sx = float(args.scale_x if args.scale_x is not None else (args.scale or 1.0))
        sy = float(args.scale_y if args.scale_y is not None else (args.scale or 1.0))
        return sx, sy
    if args.scale is not None:
        return float(args.scale), float(args.scale)
    return 1.0, 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Undistort samplePhoto2 and compare to Unity captures.")
    p.add_argument(
        "--unity-dir",
        type=str,
        default="unitysamplephoto2",
        help="Unity capture folder under project root (default: unitysamplephoto2)",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Output folder name under compare_out/",
    )
    p.add_argument("--scale", type=float, default=None, help="Isotropic post-undistort scale on real.")
    p.add_argument("--scale-x", type=float, default=None, help="Post-undistort X scale on real.")
    p.add_argument("--scale-y", type=float, default=None, help="Post-undistort Y scale on real.")
    p.add_argument(
        "--from-tag-summary",
        type=Path,
        default=None,
        help="Tag compare_summary.csv; use und scale = 1/mean(Unity sx), 1/mean(sy).",
    )
    return p.parse_args()


def main() -> None:
    global UNITY_DIR, OUT_DIR, UNDIST_DIR, UNDIST_SX, UNDIST_SY
    args = parse_args()
    UNDIST_SX, UNDIST_SY = scales_from_args(args)
    scaled = abs(UNDIST_SX - 1.0) > 1e-6 or abs(UNDIST_SY - 1.0) > 1e-6

    UNITY_DIR = (ROOT / args.unity_dir).resolve()
    out_name = args.out_name.strip() or (
        f"samplephoto2_undist_scaled_vs_{UNITY_DIR.name}"
        if scaled
        else f"samplephoto2_undist_vs_{UNITY_DIR.name}"
    )
    OUT_DIR = ROOT / "compare_out" / out_name
    UNDIST_DIR = OUT_DIR / "undistorted_real"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    UNDIST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"REAL : {REAL_DIR}")
    print(f"UNITY: {UNITY_DIR}")
    print(f"OUT  : {OUT_DIR}")
    print(f"post-undistort scale sx,sy=({UNDIST_SX:.4f},{UNDIST_SY:.4f})")

    rows = []
    thumbs = []

    for stem in IDS:
        rp = find_real(stem)
        up = find_unity(stem)
        if rp is None or up is None:
            print(f"skip {stem}: real={'ok' if rp else 'MISSING'} unity={'ok' if up else 'MISSING'}")
            continue

        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"fail read {stem}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        und, new_K = undistort_bgr(raw)
        und = center_scale(und, UNDIST_SX, UNDIST_SY)
        und_path = UNDIST_DIR / f"{stem}_undist.jpg"
        cv2.imwrite(str(und_path), und)

        und_w = cv2.resize(und, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(und_w))

        best = search_best(ref_e, sobel_mag(to_gray(unity_w)))
        best["rot"] = 0

        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        best["dx_full"] = dx_f
        best["dy_full"] = dy_f
        unity_aligned = warp_bgr(unity, best["sx"], best["sy"], dx_f, dy_f)

        mean_abs_full = float(
            np.mean(np.abs(und.astype(np.float32) - unity_aligned.astype(np.float32)))
        )

        panel = make_panel(
            raw, und, unity, unity_aligned, best,
            f"{stem}: samplePhoto2 undistort+scale -> {UNITY_DIR.name} (no rot180)",
        )
        panel_path = OUT_DIR / f"compare_{stem}.png"
        panel.save(panel_path)
        thumbs.append(panel.resize((panel.width // 3, panel.height // 3), Image.Resampling.BILINEAR))

        o_w = cv2.resize(und, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        u_w = cv2.resize(unity_aligned, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT_DIR / f"{stem}_yellow_cyan.png"), yellow_cyan_overlay(o_w, u_w))
        cv2.imwrite(str(OUT_DIR / f"{stem}_absdiff_heat.png"), absdiff_heat(o_w, u_w))

        row = {
            "id": stem,
            "undist_scale_x": f"{UNDIST_SX:.4f}",
            "undist_scale_y": f"{UNDIST_SY:.4f}",
            "rot": best["rot"],
            "sx": f"{best['sx']:.4f}",
            "sy": f"{best['sy']:.4f}",
            "dx_full": dx_f,
            "dy_full": dy_f,
            "edgeNCC": f"{best['ncc']:.4f}",
            "ncc_nowarp": f"{best['ncc_nowarp']:.4f}",
            "mean_absdiff_full": f"{mean_abs_full:.2f}",
            "undist": und_path.name,
            "panel": panel_path.name,
            "new_fx": f"{new_K[0,0]:.4f}",
            "new_fy": f"{new_K[1,1]:.4f}",
        }
        rows.append(row)
        print(
            f"[{stem:>8}] rot=0  "
            f"NCC={best['ncc']:.3f} nowarp={best['ncc_nowarp']:.3f}  "
            f"sx,sy=({best['sx']:.2f},{best['sy']:.2f})  "
            f"dx,dy=({dx_f:+d},{dy_f:+d})  mean|diff|={mean_abs_full:.1f}"
        )

    if not rows:
        print("No pairs compared.")
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
    print("\n=== SUMMARY ===")
    print(f"output: {OUT_DIR}")
    print("rotation: fixed rot=0 (no Unity 180)")
    print(f"post-undist scale sx,sy=({UNDIST_SX:.4f},{UNDIST_SY:.4f})")
    print(f"edgeNCC mean={np.mean(nccs):.3f} median={np.median(nccs):.3f}")
    print(f"residual Unity sx,sy mean=({np.mean(sxs):.3f},{np.mean(sys_):.3f})")
    print(f"mean|absdiff| mean={np.mean(diffs):.1f} median={np.median(diffs):.1f}")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()

