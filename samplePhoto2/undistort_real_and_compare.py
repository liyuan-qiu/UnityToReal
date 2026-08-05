"""
Approach A: undistort real JPGs with OpenCV Brown-Conrady coeffs,
then compare against Unity PNGs (pinhole, no distortion).

Camera matrix is derived from the documented pinhole FOV:
  HFOV=71.6 deg, VFOV=51.2 deg @ 1080x720
  (override FX/FY/CX/CY below if you have exact calibration).

Outputs:
  compare_out/undistorted_real/{id}_undist.png
  compare_out/undistorted_real/{id}_side_by_side.png
  compare_out/undistorted_real/compare_summary.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ORIG_DIR = ROOT / "samplePhoto2"
UNITY_DIR = ROOT / "Unity based on read transform"
OUT_DIR = ROOT / "compare_out" / "undistorted_real"

# Image / intrinsics (from your calibration note: pinhole FOV)
IMG_W, IMG_H = 1080, 720
HFOV_DEG = 71.6
VFOV_DEG = 51.2

# OpenCV distortion: (k1, k2, p1, p2, k3)
DIST = np.array(
    [-0.38898088, 0.15099531, -0.00301529, 0.00057045, -0.02746219],
    dtype=np.float64,
)

# Working size for comparison panels
WORK_W, WORK_H = 540, 360
IDS = list(range(1, 10))
TRY_ROTS = (0, 180)
SX_MIN, SX_MAX, SX_STEPS = 0.75, 1.40, 14
SY_MIN, SY_MAX, SY_STEPS = 0.75, 1.40, 14


def build_camera_matrix(
    w: int = IMG_W,
    h: int = IMG_H,
    hfov_deg: float = HFOV_DEG,
    vfov_deg: float = VFOV_DEG,
) -> np.ndarray:
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def undistort_bgr(bgr: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0, newImgSize=(w, h))
    und = cv2.undistort(bgr, K, dist, None, new_K)
    return und, new_K


def to_gray(img_bgr_or_rgb: np.ndarray) -> np.ndarray:
    if img_bgr_or_rgb.ndim == 2:
        return img_bgr_or_rgb.astype(np.float32)
    # assume BGR from cv2
    g = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


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
    # paste with clipping
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
    best["ncc_nowarp"] = float(
        np.mean(zscore(ref_e) * zscore(mov_e))
    )  # approximate; better use full ncc
    # proper zero-shift ncc:
    aa, bb = zscore(ref_e).ravel(), zscore(mov_e).ravel()
    best["ncc_nowarp"] = float(np.dot(aa, bb) / aa.size)
    return best


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_panel(
    raw_bgr: np.ndarray,
    und_bgr: np.ndarray,
    unity_bgr: np.ndarray,
    unity_warped: np.ndarray,
    params: dict,
    title: str,
) -> Image.Image:
    # resize all to work size for panel
    def rs(im):
        return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    tiles_bgr = [
        (rs(raw_bgr), "real raw (distorted)"),
        (rs(und_bgr), "real undistorted"),
        (rs(unity_bgr), f"unity (rot={params['rot']})"),
        (rs(unity_warped), "unity warped to und"),
    ]
    o = rs(und_bgr).astype(np.float32)
    u = rs(unity_warped).astype(np.float32)
    blend = np.clip(0.5 * o + 0.5 * u, 0, 255).astype(np.uint8)
    diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)
    tiles_bgr += [(blend, "blend und+unity"), (diff, "abs diff")]

    gap, header, footer = 6, 44, 56
    cols = len(tiles_bgr)
    canvas = Image.new(
        "RGB",
        (cols * WORK_W + (cols + 1) * gap, header + WORK_H + footer),
        (22, 22, 22),
    )
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)
    for i, (bgr, label) in enumerate(tiles_bgr):
        x = gap + i * (WORK_W + gap)
        canvas.paste(bgr_to_pil(bgr), (x, header))
        draw.text((x, header + WORK_H + 4), label, fill=(200, 200, 200), font=fs)
    info = (
        f"rot={params['rot']}  sx={params['sx']:.3f} sy={params['sy']:.3f}  "
        f"dx,dy=({params['dx']},{params['dy']})  edgeNCC={params['ncc']:.4f}  "
        f"ncc_nowarp={params['ncc_nowarp']:.4f}"
    )
    draw.text((gap, header + WORK_H + 28), info, fill=(255, 220, 120), font=fs)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    K = build_camera_matrix()
    print("Camera matrix K (from pinhole FOV):")
    print(K)
    print("dist (k1,k2,p1,p2,k3) =", DIST.tolist())

    rows = []
    for i in IDS:
        op = ORIG_DIR / f"{i}.jpg"
        up = UNITY_DIR / f"{i}.png"
        if not op.exists() or not up.exists():
            print(f"skip {i}")
            continue

        raw = cv2.imread(str(op), cv2.IMREAD_COLOR)
        unity = cv2.imread(str(up), cv2.IMREAD_COLOR)
        if raw is None or unity is None:
            print(f"fail read {i}")
            continue
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        und, new_K = undistort_bgr(raw, K, DIST)
        und_path = OUT_DIR / f"{i}_undist.png"
        cv2.imwrite(str(und_path), und)

        # compare at work res using undistorted real as reference
        und_w = cv2.resize(und, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        unity_w = cv2.resize(unity, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        ref_e = sobel_mag(to_gray(und_w))

        best = None
        for rot in TRY_ROTS:
            mov = unity_w if rot == 0 else cv2.rotate(unity_w, cv2.ROTATE_180)
            mov_e = sobel_mag(to_gray(mov))
            cand = search_best(ref_e, mov_e)
            cand["rot"] = rot
            if best is None or cand["ncc"] > best["ncc"]:
                best = cand
        assert best is not None

        unity_rot = unity if best["rot"] == 0 else cv2.rotate(unity, cv2.ROTATE_180)
        # map shift work -> full
        dx_f = int(round(best["dx"] * IMG_W / WORK_W))
        dy_f = int(round(best["dy"] * IMG_H / WORK_H))
        unity_warped = warp_bgr(unity_rot, best["sx"], best["sy"], dx_f, dy_f)

        panel = make_panel(raw, und, unity_rot, unity_warped, best, f"pair {i}: undistorted real vs Unity")
        panel_path = OUT_DIR / f"{i}_side_by_side.png"
        panel.save(panel_path)

        row = {
            "id": i,
            "rot": best["rot"],
            "sx": f"{best['sx']:.4f}",
            "sy": f"{best['sy']:.4f}",
            "dx_work": best["dx"],
            "dy_work": best["dy"],
            "dx_full": dx_f,
            "dy_full": dy_f,
            "edgeNCC": f"{best['ncc']:.4f}",
            "ncc_nowarp": f"{best['ncc_nowarp']:.4f}",
            "undist": und_path.name,
            "panel": panel_path.name,
            "fx": f"{K[0,0]:.4f}",
            "fy": f"{K[1,1]:.4f}",
            "new_fx": f"{new_K[0,0]:.4f}",
            "new_fy": f"{new_K[1,1]:.4f}",
        }
        rows.append(row)
        print(
            f"[{i}] saved {und_path.name}  "
            f"rot={best['rot']} sx={best['sx']:.3f} sy={best['sy']:.3f}  "
            f"NCC={best['ncc']:.4f} (nowarp={best['ncc_nowarp']:.4f})"
        )

    csv_path = OUT_DIR / "compare_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. Undistorted reals + comparisons in:\n  {OUT_DIR}")
    print("Unity should use pinhole FOV H=71.6 / V=51.2 (or fx,fy from K), NOT the distorted optical FOV.")


if __name__ == "__main__":
    main()
