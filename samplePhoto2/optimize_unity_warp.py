"""
Search best image-space alignment of Unity PNGs to original JPGs:
  - translation (dx, dy)  ~ XYZ / aim residual in image plane
  - anisotropic scale (sx, sy)  ~ independent horizontal / vertical FOV
  - also tries rot0 and rot180

Writes:
  compare_out/warped/{id}_unity_warped.png   best-warped Unity
  compare_out/warped/{id}_side_by_side.png   orig | warped | blend | diff
  compare_out/warped/best_params.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ORIG_DIR = ROOT / "samplePhoto2"
UNITY_DIR = ROOT / "Unity based on read transform"
OUT_DIR = ROOT / "compare_out" / "warped"

# Keep aspect; work at half-res for speed, then apply params at full-res.
WORK_W, WORK_H = 540, 360
SX_MIN, SX_MAX, SX_STEPS = 0.70, 1.50, 17
SY_MIN, SY_MAX, SY_STEPS = 0.70, 1.50, 17
IDS = list(range(1, 10))
TRY_ROTS = (0, 180)


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return img


def to_gray(img: Image.Image) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return 0.2989 * a[..., 0] + 0.5870 * a[..., 1] + 0.1140 * a[..., 2]


def sobel_mag(g: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    mag = np.sqrt(gx * gx + gy * gy)
    # 3x3 separable blur
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    tmp = np.pad(mag, ((0, 0), (1, 1)), mode="edge")
    tmp = tmp[:, :-2] * k[0] + tmp[:, 1:-1] * k[1] + tmp[:, 2:] * k[2]
    tmp = np.pad(tmp, ((1, 1), (0, 0)), mode="edge")
    return tmp[:-2, :] * k[0] + tmp[1:-1, :] * k[1] + tmp[2:, :] * k[2]


def zscore(a: np.ndarray) -> np.ndarray:
    m = float(a.mean())
    s = float(a.std())
    if s < 1e-6:
        return a * 0.0
    return (a - m) / s


def fft_ncc_best_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, int, int]:
    r = zscore(ref)
    m = zscore(mov)
    corr = np.fft.ifft2(np.fft.fft2(r) * np.conj(np.fft.fft2(m))).real / r.size
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(corr[iy, ix]), int(dy), int(dx)


def warp_gray(g: np.ndarray, sx: float, sy: float, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Anisotropic scale about center, then translate. Output same HxW."""
    h, w = g.shape
    nw = max(1, int(round(w * sx)))
    nh = max(1, int(round(h * sy)))
    # normalize display range for PIL
    gmax = float(g.max()) if float(g.max()) > 1e-6 else 1.0
    pil = Image.fromarray(np.clip(g / gmax * 255.0, 0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (w, h), 0)
    canvas.paste(pil, ((w - nw) // 2 + dx, (h - nh) // 2 + dy))
    return np.asarray(canvas, dtype=np.float32)


def warp_rgb(img: Image.Image, sx: float, sy: float, dx: int, dy: int) -> Image.Image:
    w, h = img.size
    nw = max(1, int(round(w * sx)))
    nh = max(1, int(round(h * sy)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(resized, ((w - nw) // 2 + dx, (h - nh) // 2 + dy))
    return canvas


def rotate_img(img: Image.Image, rot: int) -> Image.Image:
    return img.rotate(rot) if rot else img.copy()


def search_best(ref_e: np.ndarray, mov_e: np.ndarray) -> dict:
    best = {"ncc": -1e9, "sx": 1.0, "sy": 1.0, "dx": 0, "dy": 0}
    sxs = np.linspace(SX_MIN, SX_MAX, SX_STEPS)
    sys = np.linspace(SY_MIN, SY_MAX, SY_STEPS)
    for sx in sxs:
        for sy in sys:
            # scale first; shift solved by FFT
            scaled = warp_gray(mov_e, float(sx), float(sy), 0, 0)
            ncc, dy, dx = fft_ncc_best_shift(ref_e, scaled)
            if ncc > best["ncc"]:
                best = {
                    "ncc": ncc,
                    "sx": float(sx),
                    "sy": float(sy),
                    "dx": int(dx),
                    "dy": int(dy),
                }
    return best


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_side_by_side(orig: Image.Image, warped: Image.Image, title: str, params: dict) -> Image.Image:
    w, h = orig.size
    gap = 8
    header, footer = 40, 50
    canvas = Image.new("RGB", (4 * w + 5 * gap, header + h + footer), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    f, fs = font(16), font(13)
    draw.text((gap, 10), title, fill=(240, 240, 240), font=f)

    o = np.asarray(orig, dtype=np.float32)
    u = np.asarray(warped, dtype=np.float32)
    blend = np.clip(0.5 * o + 0.5 * u, 0, 255).astype(np.uint8)
    diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)

    tiles = [
        (orig, "original"),
        (warped, "unity warped"),
        (Image.fromarray(blend), "blend"),
        (Image.fromarray(diff), "abs diff"),
    ]
    for i, (im, label) in enumerate(tiles):
        x = gap + i * (w + gap)
        canvas.paste(im, (x, header))
        draw.text((x, header + h + 4), label, fill=(200, 200, 200), font=fs)

    info = (
        f"rot={params['rot']}  sx={params['sx']:.3f}  sy={params['sy']:.3f}  "
        f"dx={params['dx']}  dy={params['dy']}  edgeNCC={params['ncc']:.4f}"
    )
    draw.text((gap, header + h + 28), info, fill=(255, 220, 120), font=fs)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for i in IDS:
        op = ORIG_DIR / f"{i}.jpg"
        up = UNITY_DIR / f"{i}.png"
        if not op.exists() or not up.exists():
            print(f"skip {i}")
            continue

        # Full-res originals for final warp output
        orig_full = load_rgb(op)
        unity_full = load_rgb(up)
        fw, fh = orig_full.size

        # Work-res for search
        orig_w = orig_full.resize((WORK_W, WORK_H), Image.Resampling.BILINEAR)
        unity_w = unity_full.resize((WORK_W, WORK_H), Image.Resampling.BILINEAR)
        ref_e = sobel_mag(to_gray(orig_w))

        best_all = None
        for rot in TRY_ROTS:
            mov = rotate_img(unity_w, rot)
            mov_e = sobel_mag(to_gray(mov))
            cand = search_best(ref_e, mov_e)
            cand["rot"] = rot
            if best_all is None or cand["ncc"] > best_all["ncc"]:
                best_all = cand

        assert best_all is not None

        # Map work-res shift to full-res
        scale_x = fw / WORK_W
        scale_y = fh / WORK_H
        dx_full = int(round(best_all["dx"] * scale_x))
        dy_full = int(round(best_all["dy"] * scale_y))

        unity_rot = rotate_img(unity_full, best_all["rot"])
        warped_full = warp_rgb(
            unity_rot, best_all["sx"], best_all["sy"], dx_full, dy_full
        )

        # Also warp work-res for side-by-side panel at work size
        warped_work = warp_rgb(
            rotate_img(unity_w, best_all["rot"]),
            best_all["sx"],
            best_all["sy"],
            best_all["dx"],
            best_all["dy"],
        )
        panel = make_side_by_side(
            orig_w,
            warped_work,
            f"pair {i}: best sx/sy + shift (+rot search)",
            best_all,
        )

        out_img = OUT_DIR / f"{i}_unity_warped.png"
        out_panel = OUT_DIR / f"{i}_side_by_side.png"
        warped_full.save(out_img)
        panel.save(out_panel)

        row = {
            "id": i,
            "rot": best_all["rot"],
            "sx": f"{best_all['sx']:.4f}",
            "sy": f"{best_all['sy']:.4f}",
            "dx_work": best_all["dx"],
            "dy_work": best_all["dy"],
            "dx_full": dx_full,
            "dy_full": dy_full,
            "edgeNCC": f"{best_all['ncc']:.4f}",
            "fov_hint_x": f"scale_x={best_all['sx']:.3f} (>1 Unity too narrow / zoomed-in on X)",
            "fov_hint_y": f"scale_y={best_all['sy']:.3f} (>1 Unity too narrow / zoomed-in on Y)",
            "warped": out_img.name,
            "panel": out_panel.name,
        }
        rows.append(row)
        print(
            f"[{i}] rot={best_all['rot']:3d}  "
            f"sx={best_all['sx']:.3f} sy={best_all['sy']:.3f}  "
            f"dx,dy(full)=({dx_full},{dy_full})  "
            f"NCC={best_all['ncc']:.4f}"
        )

    csv_path = OUT_DIR / "best_params.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate suggestion for Unity
    sxs = [float(r["sx"]) for r in rows]
    sys_ = [float(r["sy"]) for r in rows]
    rots = [int(r["rot"]) for r in rows]
    print("\n======== DONE ========")
    print(f"Output: {OUT_DIR}")
    print(f"rot counts: {{{', '.join(f'{k}:{rots.count(k)}' for k in sorted(set(rots)))}}}")
    print(f"sx mean={np.mean(sxs):.3f} std={np.std(sxs):.3f}")
    print(f"sy mean={np.mean(sys_):.3f} std={np.std(sys_):.3f}")
    print(
        "Unity FOV rough map (perspective):\n"
        "  if sx>1, image was stretched wider to match => Unity horizontal FOV too small "
        f"(try FOV_x ~ current * sx, mean sx={np.mean(sxs):.3f})\n"
        "  same for sy / vertical FOV "
        f"(mean sy={np.mean(sys_):.3f})\n"
        "  dx,dy are residual image shifts after that FOV fix (pose/aim)."
    )


if __name__ == "__main__":
    main()
