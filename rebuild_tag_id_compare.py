"""
Tag vs Unity, reliable 3-step compare:
  1) Detect ArUco 4x4 cells on undistorted real and Unity rot180
  2) Shared IDs: robust sx, sy, tx, ty (no rotation / pitch / yaw)
  3) Warp Unity onto real; panel 5 = blend, panel 6 = absdiff on overlap
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
REAL_DIR = ROOT / "trainingData20260818"
NPZ = REAL_DIR / "capsule_intrinsics.npz"
PHOTO_IDS = [8, 9, 10, 11, 12]
WORK_W, WORK_H = 540, 360
IMG_W, IMG_H = 1080, 720


def font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def undistort(bgr, K, dist):
    """Undistort real into the SAME pinhole K Unity uses. Do not change fx/cx."""
    h, w = bgr.shape[:2]
    return cv2.undistort(bgr, K, dist, None, K), K


def detect_markers(gray):
    """id -> (center 2, corners 4x2)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        pts = c.reshape(-1, 2).astype(np.float64)
        out[int(i)] = {"center": pts.mean(axis=0), "corners": pts}
    return out


def _axis_scale_shift(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """dst ≈ s * src + t, weighted 1D least squares."""
    src = np.asarray(src, dtype=np.float64).reshape(-1)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1)
    if w is None:
        w = np.ones(len(src), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
    sw = float(np.sum(w))
    if sw < 1e-12 or len(src) == 0:
        return 1.0, 0.0
    mx = float(np.dot(w, src) / sw)
    my = float(np.dot(w, dst) / sw)
    var = float(np.dot(w, (src - mx) ** 2))
    if var < 1e-9:
        return 1.0, float(my - mx)
    s = float(np.clip(np.dot(w, (src - mx) * (dst - my)) / var, 0.05, 3.0))
    return s, float(my - s * mx)


def _sxsy_t(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """dst_x ≈ sx*src_x + tx, dst_y ≈ sy*src_y + ty. No rotation / shear."""
    sx, tx = _axis_scale_shift(src[:, 0], dst[:, 0], w)
    sy, ty = _axis_scale_shift(src[:, 1], dst[:, 1], w)
    return sx, sy, tx, ty


def _apply_sxsy(src: np.ndarray, sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    return np.column_stack((sx * src[:, 0] + tx, sy * src[:, 1] + ty))


def _huber_irls(
    src: np.ndarray,
    dst: np.ndarray,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    delta: float = 10.0,
    iters: int = 16,
) -> tuple[float, float, float, float]:
    n = len(src)
    w = np.ones(n, dtype=np.float64)
    for _ in range(iters):
        err = np.linalg.norm(_apply_sxsy(src, sx, sy, tx, ty) - dst, axis=1)
        w = np.ones(n, dtype=np.float64)
        big = err > delta
        w[big] = delta / np.maximum(err[big], 1e-9)
        sx, sy, tx, ty = _sxsy_t(src, dst, w)
    return sx, sy, tx, ty


def _matrix(sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    return np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float64)


def fit_sxsy_xy(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    thresh_px: float = 16.0,
    iters: int = 800,
):
    """
    Map Unity -> real with independent sx, sy plus translation.
    RANSAC (2-point) then Huber IRLS. Rotation stays 0.
    """
    src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst_xy = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    n = len(src_xy)
    empty = (None, float("nan"), float("nan"), float("nan"), float("nan"), None)
    if n == 0:
        return empty
    if n == 1:
        t = dst_xy[0] - src_xy[0]
        return _matrix(1.0, 1.0, t[0], t[1]), 1.0, 1.0, float(t[0]), float(t[1]), np.ones(1, dtype=bool)

    rng = np.random.default_rng(0)
    best_mask = np.ones(n, dtype=bool)
    best_score = (-1, 1e18)
    min_span = 6.0
    for _ in range(iters):
        idx = rng.choice(n, size=2, replace=False)
        a, b = src_xy[idx]
        if abs(a[0] - b[0]) < min_span or abs(a[1] - b[1]) < min_span:
            continue
        sx, sy, tx, ty = _sxsy_t(src_xy[idx], dst_xy[idx])
        err = np.linalg.norm(_apply_sxsy(src_xy, sx, sy, tx, ty) - dst_xy, axis=1)
        mask = err <= thresh_px
        nin = int(mask.sum())
        med = float(np.median(err))
        score = (nin, -med)
        if score > best_score:
            best_score = score
            best_mask = mask

    if int(best_mask.sum()) < 2:
        best_mask = np.ones(n, dtype=bool)
    sx, sy, tx, ty = _sxsy_t(src_xy[best_mask], dst_xy[best_mask])
    sx, sy, tx, ty = _huber_irls(src_xy, dst_xy, sx, sy, tx, ty, delta=thresh_px)

    err = np.linalg.norm(_apply_sxsy(src_xy, sx, sy, tx, ty) - dst_xy, axis=1)
    med = float(np.median(err)) if n else thresh_px
    adapt = float(np.clip(2.5 * med, 8.0, 40.0))
    mask = err <= adapt
    if int(mask.sum()) >= 2:
        sx, sy, tx, ty = _sxsy_t(src_xy[mask], dst_xy[mask])
        sx, sy, tx, ty = _huber_irls(src_xy[mask], dst_xy[mask], sx, sy, tx, ty, delta=adapt)
        err = np.linalg.norm(_apply_sxsy(src_xy, sx, sy, tx, ty) - dst_xy, axis=1)
        mask = err <= adapt
    return _matrix(sx, sy, tx, ty), sx, sy, tx, ty, mask


def refine_tx_ty_edge(
    unity180: np.ndarray,
    und: np.ndarray,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    max_shift: int = 18,
    down: int = 2,
) -> tuple[float, float, float]:
    """Small bounded edge-NCC polish of translation only (won't jump a checker cell)."""
    h, w = und.shape[:2]
    M0 = _matrix(sx, sy, tx, ty)
    u = cv2.warpAffine(unity180, M0, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    ww, hh = w // down, h // down
    def ed(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (ww, hh), interpolation=cv2.INTER_AREA)
        e = cv2.Canny(g, 50, 140).astype(np.float32)
        return e
    eu, er = ed(u), ed(und)
    Hh, Ww = er.shape
    ms = max(1, int(round(max_shift / down)))
    best_ncc, best_dx, best_dy = -1e9, 0, 0
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            y0 = max(0, dy)
            y1 = min(Hh, Hh + dy)
            x0 = max(0, dx)
            x1 = min(Ww, Ww + dx)
            aa = eu[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            bb = er[y0:y1, x0:x1]
            if aa.size < 400:
                continue
            aa = aa - aa.mean()
            bb = bb - bb.mean()
            den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if den < 1e-9:
                continue
            ncc = float(np.dot(aa.ravel(), bb.ravel()) / den)
            if ncc > best_ncc:
                best_ncc, best_dx, best_dy = ncc, dx, dy
    return tx + best_dx * down, ty + best_dy * down, best_ncc


def apply_pts(M, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    return (M @ np.hstack([pts, ones]).T).T


def _Rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _Ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def H_pitch_yaw(K: np.ndarray, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    R = _Ry(yaw_deg) @ _Rx(pitch_deg)
    return K @ R @ np.linalg.inv(K)


def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = (H @ np.c_[pts, np.ones(len(pts))].T).T
    return q[:, :2] / np.maximum(q[:, 2:3], 1e-9)


def fit_pitch_yaw_then_affine(
    src: np.ndarray,
    dst: np.ndarray,
    K: np.ndarray,
    lo: float = -8.0,
    hi: float = 8.0,
    step: float = 0.5,
    refine: float = 0.25,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    """Search pitch/yaw first (H=K R K^{-1}), then closed-form sx,sy,tx,ty."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 2:
        H = np.eye(3, dtype=np.float64)
        return 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, H

    def eval_py(p: float, y: float):
        H = H_pitch_yaw(K, p, y)
        sh = apply_H(H, src)
        sx, sy, tx, ty = _sxsy_t(sh, dst)
        pred = _apply_sxsy(sh, sx, sy, tx, ty)
        err = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
        return err, sx, sy, tx, ty, H

    best_err, best = 1e18, None
    p = lo
    while p <= hi + 1e-9:
        y = lo
        while y <= hi + 1e-9:
            err, sx, sy, tx, ty, H = eval_py(p, y)
            if err < best_err:
                best_err, best = err, (p, y, sx, sy, tx, ty, H)
            y += step
        p += step
    bp, by_ = best[0], best[1]
    p = bp - step
    while p <= bp + step + 1e-9:
        y = by_ - step
        while y <= by_ + step + 1e-9:
            err, sx, sy, tx, ty, H = eval_py(p, y)
            if err < best_err:
                best_err, best = err, (p, y, sx, sy, tx, ty, H)
            y += refine
        p += refine
    pitch, yaw, sx, sy, tx, ty, H = best
    sh = apply_H(H, src)
    sx, sy, tx, ty = _huber_irls(sh, dst, sx, sy, tx, ty, delta=12.0)
    pred = _apply_sxsy(sh, sx, sy, tx, ty)
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return pitch, yaw, sx, sy, tx, ty, H


def draw_ids(bgr, real_det, unity_det, common, M=None, H=None, h_first=False):
    vis = bgr.copy()
    for k in common:
        pr = tuple(np.round(real_det[k]["center"]).astype(int))
        pu = np.asarray(unity_det[k]["center"], dtype=np.float64).reshape(1, 2)
        if h_first:
            if H is not None:
                pu = apply_H(H, pu)
            if M is not None:
                pu = apply_pts(M, pu)
        else:
            if M is not None:
                pu = apply_pts(M, pu)
            if H is not None:
                pu = apply_H(H, pu)
        pu = tuple(np.round(pu[0]).astype(int))
        cv2.line(vis, pr, pu, (0, 255, 255), 2)
        cv2.circle(vis, pr, 8, (0, 255, 0), 2)
        cv2.circle(vis, pu, 8, (0, 0, 255), 2)
        cv2.putText(vis, str(k), (pr[0] + 6, pr[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis


def rs(im):
    return cv2.resize(im, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)


def yellow_cyan(real: np.ndarray, unity: np.ndarray) -> np.ndarray:
    """Real contributes R, Unity contributes B → yellow vs cyan if they disagree."""
    r, u = real.astype(np.float32), unity.astype(np.float32)
    out = np.zeros_like(r)
    out[..., 0] = u[..., 0]
    out[..., 1] = 0.5 * r[..., 1] + 0.5 * u[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_mag(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_diff_vis(real: np.ndarray, unity: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Red=real edges, blue=Unity edges. Mean |Δmag| and edge NCC on overlap."""
    er = edge_mag(real)
    eu = edge_mag(unity)
    m = overlap if overlap.any() else np.ones(er.shape, dtype=bool)

    def stretch(mag: np.ndarray) -> np.ndarray:
        v = mag[m]
        p = float(np.percentile(v, 99)) if v.size else 1.0
        p = max(p, 1e-3)
        return np.clip(mag / p * 255.0, 0, 255)

    nr, nu = stretch(er), stretch(eu)
    vis = np.zeros((*er.shape, 3), dtype=np.uint8)
    vis[..., 2] = nr.astype(np.uint8)
    vis[..., 0] = nu.astype(np.uint8)
    vis[..., 1] = np.minimum(nr, nu).astype(np.uint8)
    vis[~m] = 0
    d = np.abs(nr - nu)
    mag_m = np.maximum(er[m], eu[m])
    thr = float(np.percentile(mag_m, 75)) if mag_m.size else 0.0
    strong = m & ((er >= thr) | (eu >= thr))
    if not strong.any():
        strong = m
    mean_edge = float(np.mean(d[strong]))
    a = er[strong] - float(er[strong].mean())
    b = eu[strong] - float(eu[strong].mean())
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    ncc = float(np.dot(a, b) / den) if den > 1e-9 else 0.0
    return vis, mean_edge, ncc


def rebuild(
    ver: str,
    real_dir: Path | None = None,
    npz_path: Path | None = None,
    photo_ids: list[int] | None = None,
    out_name: str | None = None,
    pitch_yaw: bool = False,
) -> None:
    real_dir = Path(real_dir) if real_dir is not None else REAL_DIR
    npz_path = Path(npz_path) if npz_path is not None else (real_dir / "capsule_intrinsics.npz")
    photo_ids = photo_ids if photo_ids is not None else PHOTO_IDS
    unity_dir = real_dir / ver
    out = ROOT / "compare_out" / (out_name or f"{real_dir.name}_tag_vs_{ver}")
    undist_dir = out / "undistorted_real"
    id_dir = out / "id_match"
    out.mkdir(parents=True, exist_ok=True)
    undist_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path)
    K = np.asarray(z["camera_matrix"], np.float64)
    dist = np.asarray(z["dist_coeffs"], np.float64).reshape(-1)

    rows, thumbs = [], []
    print(
        f"\n===== {ver}  detect IDs -> sx,sy + tx,ty"
        f"{' + pitch/yaw' if pitch_yaw else ''} -> pixel ====="
    )
    print(f"real={real_dir}  unity={unity_dir}  npz={npz_path.name}  ids={photo_ids}")

    for i in photo_ids:
        rp = real_dir / f"Photo{i}_tag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp))
        unity = cv2.imread(str(up))
        if raw is None or unity is None:
            print(f"skip {i}")
            continue
        native = f"{unity.shape[1]}x{unity.shape[0]}"
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        und, _new_K = undistort(raw, K, dist)
        unity180 = cv2.rotate(unity, cv2.ROTATE_180)
        cv2.imwrite(str(undist_dir / f"Photo{i}_tag_undist.jpg"), und)

        rdet = detect_markers(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY))
        udet = detect_markers(cv2.cvtColor(unity180, cv2.COLOR_BGR2GRAY))
        common = sorted(set(rdet) & set(udet))

        src_c, dst_c = [], []
        src_all, dst_all = [], []
        for k in common:
            src_c.append(udet[k]["center"])
            dst_c.append(rdet[k]["center"])
            src_all.append(udet[k]["corners"])
            dst_all.append(rdet[k]["corners"])
        src_c = np.asarray(src_c, dtype=np.float64) if src_c else np.zeros((0, 2))
        dst_c = np.asarray(dst_c, dtype=np.float64) if dst_c else np.zeros((0, 2))
        src_all = np.vstack(src_all) if src_all else np.zeros((0, 2))
        dst_all = np.vstack(dst_all) if dst_all else np.zeros((0, 2))

        # Prefer 4 corners per cell; fall back to centers
        pts_src, pts_dst = (src_all, dst_all) if len(src_all) >= 2 else (src_c, dst_c)
        M, sx, sy, tx, ty, mask = fit_sxsy_xy(pts_src, pts_dst)
        n_in = int(mask.sum()) if mask is not None else 0
        ncc_t = float("nan")
        if M is not None and np.isfinite(sx) and len(src_c):
            rmse0 = float(
                np.sqrt(np.mean(np.sum((_apply_sxsy(src_c, sx, sy, tx, ty) - dst_c) ** 2, axis=1)))
            )
            tx2, ty2, ncc_t = refine_tx_ty_edge(unity180, und, sx, sy, tx, ty)
            rmse1 = float(
                np.sqrt(np.mean(np.sum((_apply_sxsy(src_c, sx, sy, tx2, ty2) - dst_c) ** 2, axis=1)))
            )
            if rmse1 <= rmse0 + 1.0:
                tx, ty = tx2, ty2
            M = _matrix(sx, sy, tx, ty)
        scale = float(math.sqrt(max(sx, 1e-6) * max(sy, 1e-6))) if np.isfinite(sx) else float("nan")

        if M is not None and len(src_c):
            pred = apply_pts(M, src_c)
            rmse = float(np.sqrt(np.mean(np.sum((pred - dst_c) ** 2, axis=1))))
        else:
            rmse = float("nan")

        pitch_deg, yaw_deg, rmse_py = 0.0, 0.0, rmse
        Hpy = None
        Mpy = M
        sx_py, sy_py, tx_py, ty_py = sx, sy, tx, ty
        if pitch_yaw and M is not None and len(pts_src) >= 4:
            pitch_deg, yaw_deg, sx_py, sy_py, tx_py, ty_py, Hpy = fit_pitch_yaw_then_affine(
                pts_src, pts_dst, K
            )
            Mpy = _matrix(sx_py, sy_py, tx_py, ty_py)
            if len(src_c):
                pred_py = _apply_sxsy(apply_H(Hpy, src_c), sx_py, sy_py, tx_py, ty_py)
                rmse_py = float(np.sqrt(np.mean(np.sum((pred_py - dst_c) ** 2, axis=1))))

        if M is not None:
            aligned = cv2.warpAffine(
                unity180, M, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
            )
            if Hpy is not None:
                u_h = cv2.warpPerspective(
                    unity180, Hpy, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
                )
                aligned_py = cv2.warpAffine(
                    u_h, Mpy, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
                )
            else:
                aligned_py = aligned
            gray_a = cv2.cvtColor(aligned_py if pitch_yaw else aligned, cv2.COLOR_BGR2GRAY)
            overlap = gray_a > 8
        else:
            aligned = np.zeros_like(und)
            aligned_py = aligned
            overlap = np.zeros(und.shape[:2], dtype=bool)

        o = und.astype(np.float32)
        u180 = unity180.astype(np.float32)
        a = aligned.astype(np.float32)
        apy = aligned_py.astype(np.float32)
        blend_direct = np.clip(0.5 * o + 0.5 * u180, 0, 255).astype(np.uint8)
        blend_aligned = np.clip(0.5 * o + 0.5 * a, 0, 255).astype(np.uint8)
        blend_py = np.clip(0.5 * o + 0.5 * apy, 0, 255).astype(np.uint8)
        diff_py = np.clip(np.abs(o - apy), 0, 255).astype(np.uint8)
        diff_aligned = np.clip(np.abs(o - a), 0, 255).astype(np.uint8)
        if overlap.any():
            mean_abs = float(np.mean(np.abs(o[overlap] - apy[overlap])))
            diff_py[~overlap] = 0
        else:
            mean_abs = float(np.mean(np.abs(o - apy)))

        before = draw_ids(blend_direct, rdet, udet, common, M=None)
        after = draw_ids(blend_aligned, rdet, udet, common, M=M)
        after_py = draw_ids(blend_py, rdet, udet, common, M=Mpy, H=Hpy, h_first=True)
        cv2.imwrite(str(id_dir / f"Photo{i}_ids_before.jpg"), before)
        cv2.imwrite(str(id_dir / f"Photo{i}_ids_after_scaleXY.jpg"), after)
        cv2.imwrite(str(id_dir / f"Photo{i}_ids_after_pitchyaw.jpg"), after_py)

        if pitch_yaw:
            tiles = [
                (rs(raw), "1 real raw"),
                (rs(und), "2 real undistort(K, dist)"),
                (rs(unity180), "3 unity rot180"),
                (rs(blend_aligned), f"4 blend sx,sy+t  RMSE={rmse:.1f}px"),
                (rs(blend_py), f"5 blend pitch/yaw then sxsy ({pitch_deg:+.2f},{yaw_deg:+.2f}) RMSE={rmse_py:.1f}px"),
                (rs(diff_py), f"6 absdiff after pitch/yaw  mean={mean_abs:.1f}"),
            ]
            title = f"{ver} Photo{i}: panel4 = scale+XY only   panel5 = pitch/yaw THEN sx,sy,t"
        else:
            tiles = [
                (rs(raw), "1 real raw"),
                (rs(und), "2 real undistort(K, dist)"),
                (rs(unity), "3 unity raw"),
                (rs(unity180), "4 unity rot180  (before sx/sy/XY)"),
                (rs(blend_aligned), f"5 blend: REAL + Unity  sx,sy=({sx:.3f},{sy:.3f}) t=({tx:+.0f},{ty:+.0f})"),
                (rs(diff_aligned), f"6 absdiff after warp  mean={mean_abs:.1f}  ID RMSE={rmse:.1f}px"),
            ]
            title = f"{ver} Photo{i}: panel5 = undist REAL + Unity after sx,sy + tx,ty"
        gap, header, footer = 6, 44, 100
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        fnt, fs = font(15), font(12)
        draw.text(
            (gap, 8),
            f"{title}",
            fill=(240, 240, 240),
            font=fnt,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label[:70], fill=(200, 200, 200), font=fs)
        line1 = (
            f"real IDs={sorted(rdet)}   unity IDs={sorted(udet)}   SHARED={common}"
        )
        line2 = (
            f"sx,sy=({sx:.4f},{sy:.4f}) t=({tx:+.1f},{ty:+.1f})  RMSE_xy={rmse:.1f}px  "
            f"pitch,yaw=({pitch_deg:+.2f},{yaw_deg:+.2f}) then sx,sy=({sx_py:.3f},{sy_py:.3f}) "
            f"t=({tx_py:+.0f},{ty_py:+.0f})  RMSE_py={rmse_py:.1f}px  |d|={mean_abs:.2f}"
        )
        line3 = (
            f"SHARED={common}  inliers={n_in}  native {native}  "
            f"{'H=K Ry(yaw) Rx(pitch) K^-1 after affine' if pitch_yaw else 'no pitch/yaw'}"
        )
        draw.text((gap, header + WORK_H + 28), line1, fill=(160, 200, 255), font=fs)
        draw.text((gap, header + WORK_H + 48), line2, fill=(255, 220, 120), font=fs)
        draw.text((gap, header + WORK_H + 68), line3, fill=(180, 180, 180), font=fs)

        panel_path = out / f"compare_{i}.png"
        canvas.save(panel_path)
        cv2.imwrite(str(out / f"{i}_absdiff.png"), diff_py if pitch_yaw else diff_aligned)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))

        rows.append(
            {
                "id": i,
                "image_file": rp.name,
                "unity_native": native,
                "real_ids": " ".join(str(k) for k in sorted(rdet)),
                "unity_ids": " ".join(str(k) for k in sorted(udet)),
                "shared_ids": " ".join(str(k) for k in common),
                "n_shared": len(common),
                "align_scale": f"{scale:.5f}",
                "align_sx": f"{sx:.5f}",
                "align_sy": f"{sy:.5f}",
                "align_tx": f"{tx:.2f}",
                "align_ty": f"{ty:.2f}",
                "align_pitch_deg": f"{pitch_deg:.3f}",
                "align_yaw_deg": f"{yaw_deg:.3f}",
                "align_sx_py": f"{sx_py:.5f}",
                "align_sy_py": f"{sy_py:.5f}",
                "align_tx_py": f"{tx_py:.2f}",
                "align_ty_py": f"{ty_py:.2f}",
                "n_inliers": n_in,
                "edge_ncc": f"{ncc_t:.4f}",
                "id_rmse_px": f"{rmse:.2f}",
                "id_rmse_pitchyaw_px": f"{rmse_py:.2f}",
                "mean_absdiff_overlap": f"{mean_abs:.2f}",
                "panel": panel_path.name,
            }
        )
        print(
            f"[{i}] shared={common}  sx,sy=({sx:.3f},{sy:.3f})  t=({tx:+.0f},{ty:+.0f})  "
            f"RMSE={rmse:.1f}px  pitch,yaw=({pitch_deg:+.2f},{yaw_deg:+.2f})  RMSE_py={rmse_py:.1f}px"
        )

    if not rows:
        return
    csv_path = out / "compare_summary.csv"
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
        triage.save(out / "triage_all.png")
    print(f"Wrote {csv_path}")


def load_warp_csv(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            i = int(row["id"])
            has_pitch_yaw = all(
                row.get(name, "").strip()
                for name in (
                    "align_pitch_deg",
                    "align_yaw_deg",
                    "align_sx_py",
                    "align_sy_py",
                    "align_tx_py",
                    "align_ty_py",
                )
            )
            out[i] = {
                "sx": float(row["align_sx"]),
                "sy": float(row["align_sy"]),
                "tx": float(row["align_tx"]),
                "ty": float(row["align_ty"]),
                "pitch_deg": float(row["align_pitch_deg"]) if has_pitch_yaw else 0.0,
                "yaw_deg": float(row["align_yaw_deg"]) if has_pitch_yaw else 0.0,
                "sx_py": float(row["align_sx_py"]) if has_pitch_yaw else float(row["align_sx"]),
                "sy_py": float(row["align_sy_py"]) if has_pitch_yaw else float(row["align_sy"]),
                "tx_py": float(row["align_tx_py"]) if has_pitch_yaw else float(row["align_tx"]),
                "ty_py": float(row["align_ty_py"]) if has_pitch_yaw else float(row["align_ty"]),
                "has_pitch_yaw": float(has_pitch_yaw),
            }
    return out


def rebuild_notag(
    unity_ver: str,
    warp_csv: Path | None,
    real_dir: Path | None = None,
    npz_path: Path | None = None,
    photo_ids: list[int] | None = None,
    out_name: str | None = None,
    apply_tag_warp: bool = True,
) -> None:
    """Apply per-photo sx,sy,tx,ty from tagged compare onto NoTag real vs Unity."""
    real_dir = Path(real_dir) if real_dir is not None else REAL_DIR
    npz_path = Path(npz_path) if npz_path is not None else (real_dir / "capsule_intrinsics.npz")
    photo_ids = photo_ids if photo_ids is not None else PHOTO_IDS
    unity_dir = real_dir / unity_ver
    if apply_tag_warp:
        if warp_csv is None:
            raise ValueError("warp_csv is required when applying Tag warp")
        warps = load_warp_csv(warp_csv)
    else:
        warps = {
            photo_id: {
                "sx": 1.0,
                "sy": 1.0,
                "tx": 0.0,
                "ty": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
                "sx_py": 1.0,
                "sy_py": 1.0,
                "tx_py": 0.0,
                "ty_py": 0.0,
                "has_pitch_yaw": 0.0,
            }
            for photo_id in photo_ids
        }
    out = ROOT / "compare_out" / (out_name or f"{real_dir.name}_NoTag_vs_{unity_ver}")
    undist_dir = out / "undistorted_real"
    out.mkdir(parents=True, exist_ok=True)
    undist_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path)
    K = np.asarray(z["camera_matrix"], np.float64)
    dist = np.asarray(z["dist_coeffs"], np.float64).reshape(-1)

    rows, thumbs = [], []
    mode = f"Tag warp from {warp_csv}" if apply_tag_warp else "NO Tag 2D warp"
    print(f"\n===== NoTag vs {unity_ver}: {mode} =====")
    print(f"real={real_dir}  unity={unity_dir}  npz={npz_path.name}  ids={photo_ids}")

    for i in photo_ids:
        if i not in warps:
            print(f"skip {i}: no warp in csv")
            continue
        warp = warps[i]
        sx, sy, tx, ty = warp["sx"], warp["sy"], warp["tx"], warp["ty"]
        pitch_deg = warp["pitch_deg"] if apply_tag_warp else 0.0
        yaw_deg = warp["yaw_deg"] if apply_tag_warp else 0.0
        use_pitch_yaw = apply_tag_warp and bool(warp["has_pitch_yaw"])
        rp = real_dir / f"Photo{i}_NoTag.jpg"
        up = unity_dir / f"Photo{i}_tag_Unity.jpg"
        raw = cv2.imread(str(rp))
        unity = cv2.imread(str(up))
        if raw is None or unity is None:
            print(f"skip {i}: missing {rp.name if raw is None else up.name}")
            continue
        native = f"{unity.shape[1]}x{unity.shape[0]}"
        if unity.shape[1] != IMG_W or unity.shape[0] != IMG_H:
            unity = cv2.resize(unity, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        if raw.shape[1] != IMG_W or raw.shape[0] != IMG_H:
            raw = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        und, _ = undistort(raw, K, dist)
        unity180 = cv2.rotate(unity, cv2.ROTATE_180)
        cv2.imwrite(str(undist_dir / f"Photo{i}_NoTag_undist.jpg"), und)

        if use_pitch_yaw:
            Hpy = H_pitch_yaw(K, pitch_deg, yaw_deg)
            unity_warped = cv2.warpPerspective(
                unity180, Hpy, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
            )
            applied_sx, applied_sy = warp["sx_py"], warp["sy_py"]
            applied_tx, applied_ty = warp["tx_py"], warp["ty_py"]
        elif not apply_tag_warp:
            unity_warped = unity180
            applied_sx, applied_sy, applied_tx, applied_ty = 1.0, 1.0, 0.0, 0.0
        else:
            unity_warped = unity180
            applied_sx, applied_sy, applied_tx, applied_ty = sx, sy, tx, ty
        M = _matrix(applied_sx, applied_sy, applied_tx, applied_ty)
        aligned = cv2.warpAffine(
            unity_warped, M, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        overlap = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) > 8
        o = und.astype(np.float32)
        a = aligned.astype(np.float32)
        blend_aligned = np.clip(0.5 * o + 0.5 * a, 0, 255).astype(np.uint8)
        yc = yellow_cyan(und, aligned)
        yc[~overlap] = 0
        edge_vis, mean_edge, edge_ncc = edge_diff_vis(und, aligned, overlap)

        tiles = [
            (rs(raw), "1 real raw NoTag"),
            (rs(und), "2 real undistort(K, dist)"),
            (rs(unity), "3 unity raw (notagUnity)"),
            (rs(unity180), "4 unity rot180  (before sx/sy/XY)"),
            (
                rs(blend_aligned),
                f"5 blend: NoTag REAL + Unity  sx,sy=({applied_sx:.3f},{applied_sy:.3f}) "
                f"t=({applied_tx:+.0f},{applied_ty:+.0f})",
            ),
            (rs(edge_vis), f"6 edge: red=real  blue=Unity  |d|={mean_edge:.1f}  NCC={edge_ncc:.3f}"),
            (rs(yc), "7 yellow/cyan (real=R, Unity=B)"),
        ]
        gap, header, footer = 6, 44, 90
        canvas = Image.new(
            "RGB",
            (len(tiles) * WORK_W + (len(tiles) + 1) * gap, header + WORK_H + footer),
            (22, 22, 22),
        )
        draw = ImageDraw.Draw(canvas)
        fnt, fs = font(15), font(12)
        draw.text(
            (gap, 8),
            (
                f"{unity_ver} Photo{i} NoTag: "
                f"{'warp copied from tagged ID fit' if apply_tag_warp else 'NO Tag 2D warp; Unity rot180 only'}"
            ),
            fill=(240, 240, 240),
            font=fnt,
        )
        for ti, (bgr, label) in enumerate(tiles):
            x = gap + ti * (WORK_W + gap)
            canvas.paste(bgr_to_pil(bgr), (x, header))
            draw.text((x, header + WORK_H + 4), label[:70], fill=(200, 200, 200), font=fs)
        draw.text(
            (gap, header + WORK_H + 28),
            f"{'warp from tagged fit' if apply_tag_warp else 'no image-space alignment'}: "
            f"pitch,yaw=({pitch_deg:+.2f},{yaw_deg:+.2f})  "
            f"sx,sy=({applied_sx:.4f},{applied_sy:.4f})  "
            f"tx,ty=({applied_tx:+.1f},{applied_ty:+.1f})",
            fill=(160, 200, 255),
            font=fs,
        )
        draw.text(
            (gap, header + WORK_H + 48),
            f"edge |d mag|={mean_edge:.2f}  edgeNCC={edge_ncc:.3f}   panel6 red=real Sobel, blue=Unity   "
            f"panel7 yellow/cyan  native {native}",
            fill=(255, 220, 120),
            font=fs,
        )
        panel_path = out / f"compare_{i}.png"
        canvas.save(panel_path)
        cv2.imwrite(str(out / f"{i}_edgediff.png"), edge_vis)
        cv2.imwrite(str(out / f"{i}_yellow_cyan.png"), yc)
        thumbs.append(canvas.resize((canvas.width // 3, canvas.height // 3), Image.Resampling.BILINEAR))
        rows.append(
            {
                "id": i,
                "image_file": rp.name,
                "unity_file": up.name,
                "align_sx": f"{applied_sx:.5f}",
                "align_sy": f"{applied_sy:.5f}",
                "align_tx": f"{applied_tx:.2f}",
                "align_ty": f"{applied_ty:.2f}",
                "align_pitch_deg": f"{pitch_deg:.3f}",
                "align_yaw_deg": f"{yaw_deg:.3f}",
                "tag_warp_applied": apply_tag_warp,
                "warp_source": str(warp_csv.as_posix()) if warp_csv is not None else "",
                "mean_edge_abs": f"{mean_edge:.2f}",
                "edge_ncc": f"{edge_ncc:.4f}",
                "panel": panel_path.name,
            }
        )
        print(
            f"[{i}] pitch,yaw=({pitch_deg:+.2f},{yaw_deg:+.2f})  "
            f"sx,sy=({applied_sx:.3f},{applied_sy:.3f})  "
            f"t=({applied_tx:+.0f},{applied_ty:+.0f})  "
            f"edgeAbs={mean_edge:.1f}  edgeNCC={edge_ncc:.3f}"
        )

    if not rows:
        return
    csv_path = out / "compare_summary.csv"
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
        triage.save(out / "triage_all.png")
    print(f"Wrote {csv_path}")


def parse_ids(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("versions", nargs="*", default=["tagUnity1", "tagUnity2", "tagUnity3", "tagUnity5"])
    p.add_argument("--real-dir", type=Path, default=None)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--ids", type=str, default="")
    p.add_argument("--out-name", type=str, default="")
    p.add_argument("--notag-unity", type=str, default="", help="e.g. notagUnity1; uses Photo*_NoTag.jpg")
    p.add_argument("--warp-csv", type=Path, default=None, help="tagged compare_summary.csv with align_sx/sy/tx/ty")
    p.add_argument("--no-tag-warp", action="store_true", help="For NoTag compare, use Unity rot180 only")
    p.add_argument("--pitch-yaw", action="store_true", help="After sx/sy/t, fit image pitch/yaw homography")
    args = p.parse_args()
    photo_ids = parse_ids(args.ids) if args.ids.strip() else None
    if args.notag_unity.strip():
        warp_csv = args.warp_csv
        if warp_csv is None and not args.no_tag_warp:
            rd = Path(args.real_dir) if args.real_dir is not None else REAL_DIR
            warp_csv = ROOT / "compare_out" / f"{rd.name}_tag_vs_tagUnity1" / "compare_summary.csv"
        if warp_csv is not None and not warp_csv.is_absolute():
            warp_csv = ROOT / warp_csv
        rebuild_notag(
            args.notag_unity.strip(),
            warp_csv,
            real_dir=args.real_dir,
            npz_path=args.npz,
            photo_ids=photo_ids,
            out_name=args.out_name.strip() or None,
            apply_tag_warp=not args.no_tag_warp,
        )
        return
    for ver in args.versions:
        rebuild(
            ver,
            real_dir=args.real_dir,
            npz_path=args.npz,
            photo_ids=photo_ids,
            out_name=args.out_name.strip() or None,
            pitch_yaw=args.pitch_yaw,
        )


if __name__ == "__main__":
    main()
