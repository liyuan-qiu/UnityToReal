"""
Convert Unity 8-bit depth visualization to metric meters.

Assumes linear encoding used by PoseCsvAutoCapture depthMaterial:
  gray/255 ~= (z - min_depth) / (max_depth - min_depth)
  => z = min_depth + (gray/255) * (max_depth - min_depth)

Default range from user: min_depth=0.01 m, max_depth=0.2 m.
Invalid / masked-out pixels (gray==0) stay 0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def gray8_to_depth_m(
    gray: np.ndarray,
    min_depth: float,
    max_depth: float,
    invert: bool = False,
    invalid_zero: bool = True,
) -> np.ndarray:
    g = gray.astype(np.float32)
    if gray.ndim == 3:
        g = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = np.clip(g / 255.0, 0.0, 1.0)
    if invert:
        t = 1.0 - t
    depth = min_depth + t * (max_depth - min_depth)
    if invalid_zero:
        depth = np.where(g > 0, depth, 0.0).astype(np.float32)
    return depth.astype(np.float32)


def depth_m_to_mm16(depth_m: np.ndarray) -> np.ndarray:
    """Store millimeters as uint16 (0 = invalid)."""
    mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    mm[depth_m <= 0] = 0
    return mm


def colorize(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = depth_m > 0
    out = np.zeros((*depth_m.shape, 3), np.uint8)
    if not np.any(valid):
        return out
    t = np.zeros_like(depth_m, np.float32)
    t[valid] = np.clip((depth_m[valid] - min_depth) / max(max_depth - min_depth, 1e-6), 0, 1)
    u8 = (t * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    out[valid] = colored[valid]
    return out


def main():
    ap = argparse.ArgumentParser(description="Decode 8-bit Unity depth to meters.")
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=ROOT / "unity_depth",
    )
    ap.add_argument("--pattern", type=str, default="Photo{id}.png")
    ap.add_argument("--ids", type=str, default="1,2,3,4")
    ap.add_argument("--min-depth", type=float, default=0.01)
    ap.add_argument("--max-depth", type=float, default=0.2)
    ap.add_argument("--invert", action="store_true", help="If white means near instead of far")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "unity_depth_metric",
    )
    ap.add_argument(
        "--also-copy-to",
        type=Path,
        default=None,
        help="Optional second output directory for decoded depth copies.",
    )
    args = ap.parse_args()

    src_dir = args.src_dir if args.src_dir.is_absolute() else ROOT / args.src_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    also = None
    if args.also_copy_to is not None:
        also = (
            args.also_copy_to
            if args.also_copy_to.is_absolute()
            else ROOT / args.also_copy_to
        )

    for sub in ("depth_m", "depth_mm16", "depth_m_color"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if also is not None:
        also.mkdir(parents=True, exist_ok=True)

    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    meta = {
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "invert": bool(args.invert),
        "formula": (
            "z = max - (g/255)*(max-min)" if args.invert else "z = min + (g/255)*(max-min)"
        ),
        "note": "8-bit Unity depth viz decoded to meters; 0 = invalid/masked",
    }
    rows = []

    print(
        f"decode 8bit -> meters  range=[{args.min_depth},{args.max_depth}] "
        f"invert={args.invert}"
    )
    for i in ids:
        p = src_dir / args.pattern.format(id=i)
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"skip {i}: missing {p}")
            continue
        depth = gray8_to_depth_m(im, args.min_depth, args.max_depth, invert=args.invert)
        mm16 = depth_m_to_mm16(depth)
        col = colorize(depth, args.min_depth, args.max_depth)

        stem = f"Photo{i}"
        np.save(out_dir / "depth_m" / f"{stem}.npy", depth)
        cv2.imwrite(str(out_dir / "depth_mm16" / f"{stem}.png"), mm16)
        cv2.imwrite(str(out_dir / "depth_m_color" / f"{stem}.png"), col)

        if also is not None:
            np.save(also / f"{stem}_depth_m.npy", depth)
            cv2.imwrite(str(also / f"{stem}_depth_mm16.png"), mm16)
            cv2.imwrite(str(also / f"{stem}_depth_m_color.png"), col)

        sel = depth > 0
        rows.append(
            {
                "id": i,
                "src": p.name,
                "valid_px": int(sel.sum()),
                "depth_min_m": f"{float(depth[sel].min()):.5f}" if sel.any() else "",
                "depth_max_m": f"{float(depth[sel].max()):.5f}" if sel.any() else "",
                "depth_mean_m": f"{float(depth[sel].mean()):.5f}" if sel.any() else "",
                "npy": f"depth_m/{stem}.npy",
                "mm16": f"depth_mm16/{stem}.png",
            }
        )
        print(
            f"[{i}] valid={int(sel.sum())}  "
            f"z=[{float(depth[sel].min()):.4f},{float(depth[sel].max()):.4f}]m  "
            f"mean={float(depth[sel].mean()):.4f}m"
        )

    if not rows:
        raise SystemExit(f"No depth images decoded from {src_dir}")

    with (out_dir / "depth_metric_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "depth_metric_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    readme = out_dir / "README.md"
    extra = f"""

## Metric depth (decoded from 8-bit)

- Range: **[{args.min_depth}, {args.max_depth}] m**
- Formula: `{meta['formula']}`
- `depth_m/Photo{{i}}.npy` : float32 meters (0=invalid)
- `depth_mm16/Photo{{i}}.png` : uint16 millimeters (0=invalid) — handy for Depth-Anything loaders
- `depth_m_color/` : turbo preview
- See `depth_metric_meta.json` / `depth_metric_manifest.csv`
"""
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        if "## Metric depth" not in txt:
            readme.write_text(txt.rstrip() + "\n" + extra, encoding="utf-8")
    else:
        readme.write_text("# Depth pack\n" + extra, encoding="utf-8")

    print(f"OUT package: {out_dir}")
    if also is not None:
        print(f"also: {also}")


if __name__ == "__main__":
    main()
