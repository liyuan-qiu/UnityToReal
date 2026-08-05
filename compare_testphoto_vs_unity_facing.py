"""
Compare TestPhoto real JPGs vs Unity captures in camera_pose_unity_facing_xyz/.

Pipeline (same idea as samplePhoto2/undistort_real_and_compare.py):
  1) undistort real with Brown-Conrady
  2) resize both to work size
  3) edge-NCC search: rot {0,180} x anisotropic scale x FFT shift
  4) report pixel shift + approximate XYZ offset at tag depth

Outputs:
  compare_out/facing_xyz/{id}_side_by_side.png
  compare_out/facing_xyz/compare_summary.csv
  compare_out/facing_xyz/triage_all.png
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REAL_DIR = ROOT / "TestPhoto"
UNITY_DIR = ROOT / "camera_pose_unity_facing_xyz"
POSE_CSV = ROOT / "camera_pose_unity_cam2tag_face.csv"
OUT_DIR = ROOT / "compare_out" / "facing_xyz"

IMG_W, IMG_H = 1080, 720  # real / calibration size
HFOV_DEG = 71.6
VFOV_DEG = 51.2
DIST = np.array(
    [-0.38898088, 0.15099531, -0.00301529, 0.00057045, -0.02746219],
    dtype=np.float64,
)

WORK_W, WORK_H = 540, 360
IDS = list(range(1, 14))
TRY_ROTS = (0, 180)
SX_MIN, SX_MAX, SX_STEPS = 0.75, 1.40, 14
SY_MIN, SY_MAX, SY_STEPS = 0.75, 1.40, 14


def build_K(w=IMG_W, h=IMG_H, hfov=HFOV_DEG, vfov=VFOV_DEG) -> np.ndarray:
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    fy = (h / 2.0) / math.tan(math.radians(vfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)


def load_depths_m() -> dict[int, float]:
    """|t_cam| from pose CSV (tag depth along cam2tag), meters."""
    out: dict[int, float] = {}
    if not POSE_CSV.exists():
        return out
    for i, row in enumerate(csv.DictReader(POSE_CSV.open(encoding="utf-8-sig")), 1):
        x = float(row["csv_x_mm"]) / 1000.0
        y = float(row["csv_y_mm"]) / 1000.0
        z = float(row["csv_z_mm"]) / 1000.0
        out[i] = float(math.sqrt(x * x + y * y + z * z))
    return out


def undistort_bgr(bgr: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0, newImgSize=(w, h))
    return cv2.undistort(bgr, K, dist, None, new_K), new_K


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


def make_panel(raw, und, unity, unity_warped, params, title) -> Image.Image:
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    tiles = [
        (rs(raw), "real raw"),
        (rs(und), "real undistorted"),
        (rs(unity), f"unity rot={params['rot']}"),
        (rs(unity_warped), "unity warped→und"),
    ]
    o = rs(und).astype(np.float32)
    u = rs(unity_warped).astype(np.float32)
    tiles += [
        (np.clip(0.5 * o + 0.5 * u, 0, 255).astype(np.uint8), "blend"),
        (np.clip(np.abs(o - u), 0, 255).astype(np.uint8), "abs diff"),
    ]
    gap, header, footer = 6, 44, 72
    cols = len(tiles)
    canvas = Image.new("RGB", (cols * WORK_W + (cols + 1) * gap, header + WORK_H + footer), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)
    for i, (bgr, label) in enumerate(tiles):
        x = gap + i * (WORK_W + gap)
        canvas.paste(bgr_to_pil(bgr), (x, header))
        draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
    info = (
        f"rot={params['rot']}  sx={params['sx']:.3f} sy={params['sy']:.3f}  "
        f"dx,dy_full=({params['dx_full']},{params['dy_full']})px  "
        f"edgeNCC={params['ncc']:.4f}  nowarp={params['ncc_nowarp']:.4f}  "
        f"approx dX,dY,dZ=({params['dX_mm']:+.1f},{params['dY_mm']:+.1f},{params['dZ_mm']:+.1f}) mm"
    )
    draw.text((gap, header + WORK_H + 28), info, fill=(255, 220, 120), font=fs)
    note = "shift: +dx = Unity content right of real → Unity cam needs +X (right); scale>1 → Unity FOV narrower / closer"
    draw.text((gap, header + WORK_H + 48), note, fill=(160, 160, 160), font=fs)
    return canvas


def approx_xyz_mm(dx_full: int, dy_full: int, sx: float, sy: float, depth_m: float, K: np.ndarray) -> tuple[float, float, float]:
    """
    Approximate camera-frame offset that would explain image misalignment.
    Convention: shift applied to Unity to match real.
      +dx_full means Unity image moved right → Unity saw scene too far left → Unity cam should move +X.
    Scale s>1 (Unity enlarged to match) → Unity was too small → Unity farther / narrower FOV → +Z approx.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    z = max(depth_m, 1e-3)
    dX = dx_full * z / fx * 1000.0
    dY = -dy_full * z / fy * 1000.0  # image +y down, cam +Y up
    s = 0.5 * (sx + sy)
    # rough: scale ≈ Z_unity / Z_real for similar FOV; dZ ≈ (1/s - 1) * Z
    dZ = (1.0 / s - 1.0) * z * 1000.0
    return dX, dY, dZ


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    K = build_K()
    depths = load_depths_m()
    print("K =\n", K)
    print(f"Unity dir: {UNITY_DIR}")
    print(f"Real dir : {REAL_DIR}")

    rows = []
    thumbs = []
    for i in IDS:
        rp = REAL_DIR / f"CamCoordTest_{i}.jpg"
        up = UNITY_DIR / f"CamCoordTest_{i}_Unity.jpg"
        if not rp.exists() or not up.exists():
            print(f"skip {i}: missing {rp.name if not rp.exists() else up.name}")
            continue

        raw = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"fail read {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        # Unity may be 640x480 — upsample to real size for fair warp/panel
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)

        und, new_K = undistort_bgr(raw, K, DIST)
        und_w = cv2.resize(und, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(und_w))

        best = None
        for rot in TRY_ROTS:
            mov = unity_w if rot == 0 else cv2.rotate(unity_w, cv2.ROTATE_180)
            cand = search_best(ref_e, sobel_mag(to_gray(mov)))
            cand["rot"] = rot
            if best is None or cand["ncc"] > best["ncc"]:
                best = cand
        assert best is not None

        unity_rot = unity if best["rot"] == 0 else cv2.rotate(unity, cv2.ROTATE_180)
        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        depth = depths.get(i, 0.06)
        dX, dY, dZ = approx_xyz_mm(dx_f, dy_f, best["sx"], best["sy"], depth, new_K)
        best["dx_full"] = dx_f
        best["dy_full"] = dy_f
        best["dX_mm"] = dX
        best["dY_mm"] = dY
        best["dZ_mm"] = dZ

        unity_warped = warp_bgr(unity_rot, best["sx"], best["sy"], dx_f, dy_f)
        panel = make_panel(
            raw, und, unity_rot, unity_warped, best,
            f"CamCoordTest_{i}: TestPhoto vs Unity facing_xyz",
        )
        panel_path = OUT_DIR / f"{i}_side_by_side.png"
        panel.save(panel_path)
        thumbs.append(panel.resize((panel.width // 3, panel.height // 3), Image.Resampling.BILINEAR))

        row = {
            "id": i,
            "rot": best["rot"],
            "sx": f"{best['sx']:.4f}",
            "sy": f"{best['sy']:.4f}",
            "dx_work": best["dx"],
            "dy_work": best["dy"],
            "dx_full": dx_f,
            "dy_full": dy_f,
            "depth_m": f"{depth:.4f}",
            "dX_mm": f"{dX:.2f}",
            "dY_mm": f"{dY:.2f}",
            "dZ_mm": f"{dZ:.2f}",
            "edgeNCC": f"{best['ncc']:.4f}",
            "ncc_nowarp": f"{best['ncc_nowarp']:.4f}",
            "panel": panel_path.name,
        }
        rows.append(row)
        print(
            f"[{i:2d}] rot={best['rot']:3d}  sx,sy=({best['sx']:.3f},{best['sy']:.3f})  "
            f"dx,dy=({dx_f:+4d},{dy_f:+4d})px  "
            f"dXYZ=({dX:+6.1f},{dY:+6.1f},{dZ:+6.1f})mm  "
            f"NCC={best['ncc']:.3f} (nowarp={best['ncc_nowarp']:.3f})"
        )

    if not rows:
        print("No pairs compared.")
        return

    csv_path = OUT_DIR / "compare_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary stats
    dXs = [float(r["dX_mm"]) for r in rows]
    dYs = [float(r["dY_mm"]) for r in rows]
    dZs = [float(r["dZ_mm"]) for r in rows]
    nccs = [float(r["edgeNCC"]) for r in rows]
    sxs = [float(r["sx"]) for r in rows]
    sys_ = [float(r["sy"]) for r in rows]
    print("\n=== mean / median over all pairs ===")
    print(f"edgeNCC     mean={np.mean(nccs):.3f}  median={np.median(nccs):.3f}")
    print(f"scale sx,sy mean=({np.mean(sxs):.3f},{np.mean(sys_):.3f})")
    print(f"dX mm       mean={np.mean(dXs):+.2f}  median={np.median(dXs):+.2f}  std={np.std(dXs):.2f}")
    print(f"dY mm       mean={np.mean(dYs):+.2f}  median={np.median(dYs):+.2f}  std={np.std(dYs):.2f}")
    print(f"dZ mm       mean={np.mean(dZs):+.2f}  median={np.median(dZs):+.2f}  std={np.std(dZs):.2f}")
    print(f"(approx: Unity camera should move by mean dXYZ relative to current pose)")

    if thumbs:
        tw, th = thumbs[0].size
        cols = 1
        triage = Image.new("RGB", (tw, th * len(thumbs) + 8 * (len(thumbs) + 1)), (18, 18, 18))
        y = 8
        for t in thumbs:
            triage.paste(t, (0, y))
            y += th + 8
        triage_path = OUT_DIR / "triage_all.png"
        triage.save(triage_path)
        print(f"\nWrote {csv_path}")
        print(f"Wrote {triage_path}")
        print(f"Panels in {OUT_DIR}")


if __name__ == "__main__":
    main()
