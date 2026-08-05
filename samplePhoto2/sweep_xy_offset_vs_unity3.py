"""
Sweep camera-frame XY offsets (mm) when comparing
  samplePhoto2 undistort(+tag scale) vs unitySamplePhoto3.

Offsets tested (dX, dY) mm:
  (-3,-1), (+3,+1), (-3,+1), (+3,-1), and (0,0) baseline.

Convention (same as tag compare notes):
  Applying image shift (dx,dy) to Unity to match real means
  Unity camera should move (+dX, +dY) in cam/Unity axes.
  Pixel <-> mm at depth Z:
    dx_px =  dX_mm * fx / (Z_m * 1000)
    dy_px = -dY_mm * fy / (Z_m * 1000)   # image +y down, cam +Y up

Outputs under compare_out/samplephoto2_xy_offset_sweep_unitySamplePhoto3/
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import compare_samplephoto_undist_vs_unity as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_ROOT = ROOT / "compare_out" / "samplephoto2_xy_offset_sweep_unitySamplePhoto3"
POSE_CSV = HERE / "camera_pose_unity_real_photos.csv"
TAG_SUMMARY = ROOT / "compare_out" / "testphoto_undist_vs_unity" / "compare_summary.csv"

# tag-derived post-undistort scale
UNDIST_SX, UNDIST_SY = 1.2935, 1.2264

OFFSETS_MM = [
    (0.0, 0.0),
    (-3.0, -1.0),
    (3.0, 1.0),
    (-3.0, 1.0),
    (3.0, -1.0),
]

# calibrated fx,fy (same as undistort K)
FX, FY = float(C.K[0, 0]), float(C.K[1, 1])


def load_depths_m() -> dict[str, float]:
    out: dict[str, float] = {}
    if not POSE_CSV.exists():
        return out
    for row in csv.DictReader(POSE_CSV.open(encoding="utf-8-sig")):
        stem = Path(row.get("image_file", row.get("photo", ""))).stem
        x = float(row["csv_CamX_mm"]) / 1000.0
        y = float(row["csv_CamY_mm"]) / 1000.0
        z = float(row["csv_CamZ_mm"]) / 1000.0
        out[stem] = float(math.sqrt(x * x + y * y + z * z))
    return out


def mm_to_px(dX_mm: float, dY_mm: float, depth_m: float) -> tuple[int, int]:
    z = max(depth_m, 1e-3)
    dx = int(round(dX_mm * FX / (z * 1000.0)))
    dy = int(round(-dY_mm * FY / (z * 1000.0)))
    return dx, dy


def shift_bgr(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    C.UNITY_DIR = (ROOT / "unitySamplePhoto3").resolve()
    C.UNDIST_SX, C.UNDIST_SY = UNDIST_SX, UNDIST_SY
    depths = load_depths_m()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Unity: {C.UNITY_DIR}")
    print(f"undist scale=({UNDIST_SX},{UNDIST_SY})")
    print(f"fx,fy=({FX:.2f},{FY:.2f})")
    print(f"OUT: {OUT_ROOT}")

    # accumulate per-offset metrics
    offset_rows = []
    best_by_stem: dict[str, dict] = {}

    for dX, dY in OFFSETS_MM:
        tag = f"dX{dX:+.0f}_dY{dY:+.0f}".replace("+", "p").replace("-", "m")
        # clearer folder names
        tag = f"dx{dX:+.0f}mm_dy{dY:+.0f}mm".replace("+", "p").replace("-", "m")
        sub = OUT_ROOT / tag
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n=== offset dX,dY=({dX:+.1f},{dY:+.1f}) mm  -> {sub.name} ===")

        nccs, diffs, nowarps = [], [], []
        for stem in C.IDS:
            rp = C.find_real(stem)
            up = C.find_unity(stem)
            if rp is None or up is None:
                continue

            raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
            unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
            if raw is None or unity is None:
                continue
            if raw.shape[1] != C.IMG_W or raw.shape[0] != C.IMG_H:
                raw = cv2.resize(raw, (C.IMG_W, C.IMG_H), interpolation=cv2.INTER_AREA)
            if unity.shape[1] != C.IMG_W or unity.shape[0] != C.IMG_H:
                unity = cv2.resize(unity, (C.IMG_W, C.IMG_H), interpolation=cv2.INTER_CUBIC)

            und, _ = C.undistort_bgr(raw)
            und = C.center_scale(und, UNDIST_SX, UNDIST_SY)

            depth = depths.get(stem, 0.06)
            dx0, dy0 = mm_to_px(dX, dY, depth)
            unity_shift = shift_bgr(unity, dx0, dy0)

            und_w = cv2.resize(und, (C.WORK_W, C.WORK_H), interpolation=cv2.INTER_AREA)
            unity_w = cv2.resize(unity_shift, (C.WORK_W, C.WORK_H), interpolation=cv2.INTER_AREA)
            ref_e = C.sobel_mag(C.to_gray(und_w))
            best = C.search_best(ref_e, C.sobel_mag(C.to_gray(unity_w)))
            best["rot"] = 0

            dx_f = int(round(best["dx"] * C.IMG_W / C.WORK_W))
            dy_f = int(round(best["dy"] * C.IMG_H / C.WORK_H))
            best["dx_full"] = dx_f
            best["dy_full"] = dy_f
            unity_aligned = C.warp_bgr(unity_shift, best["sx"], best["sy"], dx_f, dy_f)

            mean_abs = float(
                np.mean(np.abs(und.astype(np.float32) - unity_aligned.astype(np.float32)))
            )
            nccs.append(best["ncc"])
            nowarps.append(best["ncc_nowarp"])
            diffs.append(mean_abs)

            # panel
            params = dict(best)
            panel = C.make_panel(
                raw,
                und,
                unity_shift,
                unity_aligned,
                params,
                f"{stem}: dX,dY=({dX:+.0f},{dY:+.0f})mm -> px({dx0:+d},{dy0:+d}) Z={depth*1000:.0f}mm",
            )
            panel.save(sub / f"compare_{stem}.png")
            o_w = cv2.resize(und, (C.WORK_W, C.WORK_H), interpolation=cv2.INTER_AREA)
            u_w = cv2.resize(unity_aligned, (C.WORK_W, C.WORK_H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(sub / f"{stem}_yellow_cyan.png"), C.yellow_cyan_overlay(o_w, u_w))

            prev = best_by_stem.get(stem)
            if prev is None or mean_abs < prev["mean_abs"]:
                best_by_stem[stem] = {
                    "dX": dX,
                    "dY": dY,
                    "mean_abs": mean_abs,
                    "ncc": best["ncc"],
                    "dx0": dx0,
                    "dy0": dy0,
                }

            print(
                f"  [{stem:>8}] Z={depth*1000:5.1f}mm px=({dx0:+4d},{dy0:+4d})  "
                f"NCC={best['ncc']:.3f} nowarp={best['ncc_nowarp']:.3f}  "
                f"extra_shift=({dx_f:+d},{dy_f:+d})  mean|diff|={mean_abs:.1f}"
            )

        if not diffs:
            continue
        offset_rows.append(
            {
                "dX_mm": dX,
                "dY_mm": dY,
                "folder": tag,
                "edgeNCC_mean": f"{float(np.mean(nccs)):.4f}",
                "edgeNCC_median": f"{float(np.median(nccs)):.4f}",
                "ncc_nowarp_mean": f"{float(np.mean(nowarps)):.4f}",
                "mean_absdiff_mean": f"{float(np.mean(diffs)):.2f}",
                "mean_absdiff_median": f"{float(np.median(diffs)):.2f}",
                "n_pairs": len(diffs),
            }
        )
        print(
            f"  >> mean NCC={np.mean(nccs):.3f}  mean|diff|={np.mean(diffs):.1f}  "
            f"nowarp={np.mean(nowarps):.3f}"
        )

    # rank offsets
    offset_rows.sort(key=lambda r: float(r["mean_absdiff_mean"]))
    csv_path = OUT_ROOT / "offset_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(offset_rows[0].keys()))
        w.writeheader()
        w.writerows(offset_rows)

    print("\n======== RANK by mean|absdiff| (lower better) ========")
    for i, r in enumerate(offset_rows, 1):
        print(
            f"{i}. dX,dY=({float(r['dX_mm']):+.1f},{float(r['dY_mm']):+.1f}) mm  "
            f"mean|diff|={r['mean_absdiff_mean']}  NCC={r['edgeNCC_mean']}  "
            f"nowarp={r['ncc_nowarp_mean']}"
        )

    print("\nPer-image best offset (by mean|absdiff|):")
    for stem, b in best_by_stem.items():
        print(
            f"  {stem}: dX,dY=({b['dX']:+.0f},{b['dY']:+.0f}) mm  "
            f"|diff|={b['mean_abs']:.1f}  NCC={b['ncc']:.3f}"
        )

    # overview text board
    lines = ["XY offset sweep: samplePhoto2 und+scale vs unitySamplePhoto3", ""]
    for r in offset_rows:
        lines.append(
            f"dX,dY=({float(r['dX_mm']):+.1f},{float(r['dY_mm']):+.1f}) mm  "
            f"mean|diff|={r['mean_absdiff_mean']}  NCC={r['edgeNCC_mean']}"
        )
    board = Image.new("RGB", (900, 40 + 28 * len(lines)), (20, 20, 20))
    draw = ImageDraw.Draw(board)
    try:
        fnt = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        fnt = ImageFont.load_default()
    y = 12
    for i, line in enumerate(lines):
        color = (255, 220, 120) if i == 2 else (220, 220, 220)  # highlight best after header
        if i >= 2 and offset_rows and line.startswith(
            f"dX,dY=({float(offset_rows[0]['dX_mm']):+.1f}"
        ):
            color = (120, 255, 160)
        draw.text((16, y), line, fill=color, font=fnt)
        y += 28
    board.save(OUT_ROOT / "rank_overview.png")
    print(f"\nWrote {csv_path}")
    print(f"Panels under {OUT_ROOT}/dx*mm_dy*mm/")


if __name__ == "__main__":
    main()
