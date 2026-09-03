# Depth-Anything training pairs (`trainingData20260818_2`)

Real NoTag RGB ↔ Unity NoTag RGB/Depth pairs for Depth-Anything.
**Before reconstruction / Depth-Anything inference on new Real images, always undistort first, then crop (or resize) back to `1080×720`.**

This folder is self-contained: calibration NPZ and helper scripts live here so other programs can run without looking up the repo root.

## Bundled tools (this folder)

| File | Role |
|------|------|
| `capsule_intrinsics.npz` | Camera `K` + `dist` for this dataset |
| `undistort_to_1080x720.py` | **Preferred** preprocess for new Real images |
| `undistort_images.py` | Undistort only (no forced 1080×720) |
| `build_depth_anything_training_pairs.py` | Rebuild this pack from parent Real/Unity dirs |
| `decode_unity_depth8_to_metric.py` | Optional: decode `unity_depth` uint8 → meters |

Example (from this folder):

```bash
python undistort_to_1080x720.py path/to/new_real.jpg path/to/out_1080x720.png
python decode_unity_depth8_to_metric.py
```

Defaults resolve relative to this folder (`capsule_intrinsics.npz`, `unity_depth/`, etc.).

> Note: re-running `build_depth_anything_training_pairs.py` may overwrite `README.md` / `metadata.json`. Keep a copy of this README if you customize it.

## Directory layout

| Path | Meaning |
|------|---------|
| `real_rgb_raw/PhotoN.png` | Original Real NoTag RGB (distorted) |
| `real_rgb/PhotoN.png` | **Undistorted** Real RGB — use this as network input |
| `unity_rgb/PhotoN.png` | Unity NoTag RGB, rotated 180° |
| `unity_depth/PhotoN.png` | Matching Unity depth (uint8 shader), rotated 180° |
| `preview/PhotoN.png` | Real \| Unity \| blend \| depth preview |
| `manifest.csv` | Pair index |
| `metadata.json` | Intrinsics + Unity camera settings |

No Tag-derived 2D warp (`sx/sy/tx/ty`, pitch/yaw) is applied in this pack.

## Preprocess before reconstruction (required)

Depth-Anything was trained on **undistorted** Real RGB at **1080×720**. New Real images must use the **same** camera model and output size.

### 1. Intrinsics

Use the NPZ **in this folder**:

```text
capsule_intrinsics.npz
```

Keys: `camera_matrix` (K), `dist_coeffs` (dist).  
Values are also recorded in `metadata.json`.

### 2–3. Undistort + crop to `1080×720`

Preferred one-shot:

```bash
python undistort_to_1080x720.py INPUT.jpg OUTPUT.png
# or a directory:
python undistort_to_1080x720.py INPUT_DIR OUTPUT_DIR
```

Equivalent manual steps:

```python
import cv2
import numpy as np

npz = np.load("capsule_intrinsics.npz")  # this folder
K = np.asarray(npz["camera_matrix"], np.float64)
dist = np.asarray(npz["dist_coeffs"], np.float64).reshape(-1)

img = cv2.imread("your_real.jpg")  # BGR
und = cv2.undistort(img, K, dist, None, K)
```

Do **not** use a different NPZ, and do **not** change `newCameraMatrix` unless you also retrain / recalibrate.

Then force **1080×720** (center crop after optional cover-scale):

```python
TARGET_W, TARGET_H = 1080, 720

h, w = und.shape[:2]
if (w, h) != (TARGET_W, TARGET_H):
    scale = max(TARGET_W / w, TARGET_H / h)
    if abs(scale - 1.0) > 1e-6:
        und = cv2.resize(
            und,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        h, w = und.shape[:2]
    x0 = max(0, (w - TARGET_W) // 2)
    y0 = max(0, (h - TARGET_H) // 2)
    und = und[y0 : y0 + TARGET_H, x0 : x0 + TARGET_W]
    if und.shape[1] != TARGET_W or und.shape[0] != TARGET_H:
        und = cv2.resize(und, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

assert und.shape[1] == 1080 and und.shape[0] == 720
cv2.imwrite("real_undistorted_1080x720.png", und)
```

### 4. Feed Depth-Anything

```text
input  = undistorted Real RGB @ 1080×720
target = unity_depth (training only; same basename)
```

Unity depth in this pack is **uint8 shader output**, not confirmed metric meters. See `metadata.json`.

## How this pack was built

Script in this folder: `build_depth_anything_training_pairs.py`

```text
Real:  undistort with K, dist (newCameraMatrix=K)
Unity: rotate 180° for RGB and depth
Size:  1080×720
```

Unity render settings used for this pack (already baked into Unity images):

```text
Gate Fit = Horizontal
Horizontal FOV = 71.25°
lensShift = [0.164, -0.081]
original = [-0.358725, -2.2282, 13.2305]
```

## Checklist for new Real images

1. Load `capsule_intrinsics.npz` from **this** folder.
2. Run `undistort_to_1080x720.py` (or `cv2.undistort(..., newCameraMatrix=K)` then crop to **1080×720**).
3. Run Depth-Anything / reconstruction on the result.
4. Do not apply Tag 2D warp for the no-warp training path.
