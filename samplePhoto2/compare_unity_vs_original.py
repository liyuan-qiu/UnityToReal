"""
Pairwise compare original JPGs vs Unity-captured PNGs (aspect-preserving).

Outputs under ../compare_out/:
  - compare_{id}.png       : orig | unity | unity_rot180 | blend_rot180 | diff_rot180
  - compare_summary.csv    : metrics for rot0 / rot180 + best scale/shift
  - triage_all.png         : overview strip

How to read metrics:
  - higher edge_NCC after rot180  => systematic 180° orientation issue
  - best_scale far from 1.0       => FOV / focal length mismatch
  - large shift_frac              => XYZ / aim offset
  - low NCC even after warp       => big pose error or different content/lighting
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
OUT_DIR = ROOT / "compare_out"

# Keep 1080x720 aspect (3:2). Working size:
W, H = 540, 360
SCALE_MIN, SCALE_MAX, SCALE_STEPS = 0.65, 1.45, 33
IDS = list(range(1, 10))


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((W, H), Image.Resampling.BILINEAR)


def to_gray(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]


def sobel_mag(g: np.ndarray) -> np.ndarray:
    # simple Sobel via numpy
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)
    # mild blur via 3x3 mean
    k = np.array([1, 2, 1], dtype=np.float32)
    k = k / k.sum()
    tmp = np.pad(mag, ((0, 0), (1, 1)), mode="edge")
    tmp = tmp[:, 0:-2] * k[0] + tmp[:, 1:-1] * k[1] + tmp[:, 2:] * k[2]
    tmp = np.pad(tmp, ((1, 1), (0, 0)), mode="edge")
    out = tmp[0:-2, :] * k[0] + tmp[1:-1, :] * k[1] + tmp[2:, :] * k[2]
    return out


def zscore(a: np.ndarray) -> np.ndarray:
    m = float(a.mean())
    s = float(a.std())
    if s < 1e-6:
        return a * 0.0
    return (a - m) / s


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = zscore(a).ravel()
    bb = zscore(b).ravel()
    return float(np.dot(aa, bb) / aa.size)


def fft_ncc_best_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, int, int]:
    r = zscore(ref)
    m = zscore(mov)
    corr = np.fft.ifft2(np.fft.fft2(r) * np.conj(np.fft.fft2(m))).real
    corr /= r.size
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(corr[iy, ix]), int(dy), int(dx)


def apply_scale(g: np.ndarray, scale: float) -> np.ndarray:
    h, w = g.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil = Image.fromarray(np.clip(g * (255.0 / (g.max() + 1e-6)), 0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (w, h), 0)
    canvas.paste(pil, ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(canvas, dtype=np.float32)


def search_scale_shift(ref_e: np.ndarray, mov_e: np.ndarray) -> dict:
    best = {"ncc": -1e9, "scale": 1.0, "dy": 0, "dx": 0}
    for s in np.linspace(SCALE_MIN, SCALE_MAX, SCALE_STEPS):
        mov_s = apply_scale(mov_e, float(s))
        score, dy, dx = fft_ncc_best_shift(ref_e, mov_s)
        if score > best["ncc"]:
            best = {"ncc": score, "scale": float(s), "dy": dy, "dx": dx}
    best["ncc_no_warp"] = ncc(ref_e, mov_e)
    return best


def rot180(img: Image.Image) -> Image.Image:
    return img.rotate(180)


def shift_rgb(img: Image.Image, dx: int, dy: int) -> Image.Image:
    w, h = img.size
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(img, (dx, dy))
    return out


def scale_rgb(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_panel(i: int, orig: Image.Image, unity: Image.Image, m0: dict, m180: dict) -> Image.Image:
    gap = 6
    header = 48
    footer = 78
    cols = 5
    canvas = Image.new("RGB", (cols * W + (cols + 1) * gap, header + H + footer), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(12)
    draw.text((gap, 10), f"pair {i}: original vs Unity (edge-NCC triage)", fill=(240, 240, 240), font=f)

    u180 = rot180(unity)
    u180_aligned = shift_rgb(scale_rgb(u180, m180["scale"]), m180["dx"], m180["dy"])

    o = np.asarray(orig, dtype=np.float32)
    u = np.asarray(u180_aligned, dtype=np.float32)
    blend = np.clip(0.5 * o + 0.5 * u, 0, 255).astype(np.uint8)
    diff = np.clip(np.abs(o - u), 0, 255).astype(np.uint8)

    tiles = [
        (orig, "original"),
        (unity, "unity raw"),
        (u180, "unity rot180"),
        (Image.fromarray(blend), "blend @rot180+warp"),
        (Image.fromarray(diff), "absdiff @rot180+warp"),
    ]
    y0 = header
    for c, (im, label) in enumerate(tiles):
        x = gap + c * (W + gap)
        canvas.paste(im, (x, y0))
        draw.text((x, y0 + H + 4), label, fill=(200, 200, 200), font=fs)

    line1 = (
        f"rot0:  edgeNCC={m0['ncc_no_warp']:.3f}  after_scale_shift={m0['ncc']:.3f}  "
        f"scale={m0['scale']:.3f}  shift=({m0['dx']},{m0['dy']})"
    )
    line2 = (
        f"rot180: edgeNCC={m180['ncc_no_warp']:.3f}  after_scale_shift={m180['ncc']:.3f}  "
        f"scale={m180['scale']:.3f}  shift=({m180['dx']},{m180['dy']})"
    )
    winner = "rot180" if m180["ncc"] >= m0["ncc"] else "rot0"
    line3 = f"winner_by_edgeNCC={winner}   |  scale~FOV proxy; shift~XYZ/aim; low NCC~pose/content mismatch"
    draw.text((gap, header + H + 28), line1, fill=(180, 220, 255), font=fs)
    draw.text((gap, header + H + 46), line2, fill=(255, 220, 140), font=fs)
    draw.text((gap, header + H + 64), line3, fill=(180, 255, 180), font=fs)
    return canvas


def classify(m0: dict, m180: dict) -> str:
    parts = []
    if m180["ncc"] > m0["ncc"] + 0.05:
        parts.append("prefers rot180 (orientation)")
    elif m0["ncc"] > m180["ncc"] + 0.05:
        parts.append("prefers rot0")
    else:
        parts.append("orient ambiguous")

    best = m180 if m180["ncc"] >= m0["ncc"] else m0
    if abs(best["scale"] - 1.0) > 0.08:
        parts.append(f"FOV-like scale={best['scale']:.2f}")
    shift = math.hypot(best["dx"], best["dy"]) / math.hypot(W, H)
    if shift > 0.04:
        parts.append(f"XYZ/aim shift={shift*100:.1f}%diag")
    if best["ncc"] < 0.20:
        parts.append("WEAK match even after warp")
    elif best["ncc"] > 0.45:
        parts.append("structure reasonably alignable")
    return "; ".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    thumbs = []

    for i in IDS:
        op = ORIG_DIR / f"{i}.jpg"
        up = UNITY_DIR / f"{i}.png"
        if not op.exists() or not up.exists():
            print(f"skip {i}")
            continue

        orig = load_rgb(op)
        unity = load_rgb(up)
        ref_e = sobel_mag(to_gray(orig))
        mov_e = sobel_mag(to_gray(unity))
        mov180_e = np.rot90(mov_e, 2)

        m0 = search_scale_shift(ref_e, mov_e)
        m180 = search_scale_shift(ref_e, mov180_e)
        hint = classify(m0, m180)

        panel = make_panel(i, orig, unity, m0, m180)
        outp = OUT_DIR / f"compare_{i}.png"
        panel.save(outp)
        thumbs.append(panel.resize((panel.width // 2, panel.height // 2), Image.Resampling.BILINEAR))

        row = {
            "id": i,
            "edgeNCC_rot0": f"{m0['ncc_no_warp']:.4f}",
            "edgeNCC_rot180": f"{m180['ncc_no_warp']:.4f}",
            "warped_rot0": f"{m0['ncc']:.4f}",
            "warped_rot180": f"{m180['ncc']:.4f}",
            "scale_rot0": f"{m0['scale']:.4f}",
            "scale_rot180": f"{m180['scale']:.4f}",
            "dx_rot180": m180["dx"],
            "dy_rot180": m180["dy"],
            "shift_frac_rot180": f"{math.hypot(m180['dx'], m180['dy']) / math.hypot(W, H):.4f}",
            "winner": "rot180" if m180["ncc"] >= m0["ncc"] else "rot0",
            "hint": hint,
            "panel": outp.name,
        }
        rows.append(row)
        print(
            f"[{i}] rot0={m0['ncc']:.3f}@{m0['scale']:.2f}  "
            f"rot180={m180['ncc']:.3f}@{m180['scale']:.2f} "
            f"shift180=({m180['dx']},{m180['dy']})  | {hint}"
        )

    summary = OUT_DIR / "compare_summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if thumbs:
        triage = Image.new("RGB", (thumbs[0].width, sum(t.height for t in thumbs)), (0, 0, 0))
        y = 0
        for t in thumbs:
            triage.paste(t, (0, y))
            y += t.height
        triage.save(OUT_DIR / "triage_all.png")

    scales = [float(r["scale_rot180"]) for r in rows]
    winners = [r["winner"] for r in rows]
    print("\n======== SUMMARY ========")
    print(f"panels: {OUT_DIR}")
    print(f"winner counts: {{{', '.join(f'{k}:{winners.count(k)}' for k in sorted(set(winners)))}}}")
    print(f"rot180 scale mean={np.mean(scales):.3f} std={np.std(scales):.3f}")
    print("Open compare_out/compare_*.png and compare_summary.csv to triage XYZ vs angle vs FOV.")


if __name__ == "__main__":
    main()
