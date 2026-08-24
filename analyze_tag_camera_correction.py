"""Physical camera correction analysis for trainingData20260818_2.

Pipeline:
1. Undistort each real tag image into the original calibrated K.
2. Rotate each Unity tag render by 180 degrees.
3. Detect sub-pixel ChArUco corners in both images.
4. Estimate one global pinhole K for all Unity renders (zero distortion).
5. Solve tag->camera pose independently for real and Unity.
6. Report the camera-in-tag delta pose needed to move Unity onto real.

The delta poses are reported in the OpenCV ChArUco board frame. They are
diagnostic values, not Unity Transform values; coordinate conversion is needed
before applying them in Unity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
DEFAULT_REAL_DIR = ROOT / "trainingData20260818_2"
DEFAULT_UNITY_DIR = DEFAULT_REAL_DIR / "tagUnity2"
DEFAULT_INTRINSICS = DEFAULT_REAL_DIR / "capsule_intrinsics.npz"
DEFAULT_OUT = ROOT / "compare_out" / "training20260818_2_physical_camera_analysis_tagUnity2"
IDS = (1, 2, 3, 4)
IMAGE_SIZE = (1080, 720)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 5), 0.010, 0.0075, ARUCO_DICT)


def create_detector() -> cv2.aruco.CharucoDetector:
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.CharucoDetector(
        BOARD, cv2.aruco.CharucoParameters(), detector_params
    )


def detect_charuco(
    image: np.ndarray, detector: cv2.aruco.CharucoDetector
) -> dict | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if marker_ids is None or marker_corners is None or len(marker_ids) < 2:
        return None

    board_ids = BOARD.getIds().reshape(-1)
    board_obj = BOARD.getObjPoints()
    marker_object_by_id = {
        int(marker_id): np.asarray(corners, np.float32).reshape(4, 3)
        for marker_id, corners in zip(board_ids, board_obj)
    }
    marker_obj, marker_img, marker_feature_ids = [], [], []
    for corners, marker_id in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id not in marker_object_by_id:
            continue
        image_corners = np.asarray(corners, np.float32).reshape(4, 2)
        marker_obj.append(marker_object_by_id[marker_id])
        marker_img.append(image_corners)
        marker_feature_ids.extend(marker_id * 4 + np.arange(4, dtype=np.int32))
    if not marker_obj:
        return None
    marker_obj_array = np.vstack(marker_obj)
    marker_img_array = np.vstack(marker_img)
    marker_feature_ids_array = np.asarray(marker_feature_ids, np.int32)

    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 6:
        pose_obj, pose_img = BOARD.matchImagePoints(charuco_corners, charuco_ids)
        pose_obj = np.asarray(pose_obj, np.float32).reshape(-1, 3)
        pose_img = np.asarray(pose_img, np.float32).reshape(-1, 2)
        source = "charuco"
        charuco_count = len(charuco_corners)
    else:
        pose_obj, pose_img = marker_obj_array, marker_img_array
        source = "aruco_fallback"
        charuco_count = 0 if charuco_corners is None else len(charuco_corners)

    return {
        "pose_obj": pose_obj,
        "pose_img": pose_img,
        "pose_source": source,
        "charuco_count": int(charuco_count),
        "marker_count": int(len(marker_ids)),
        "marker_obj": marker_obj_array,
        "marker_img": marker_img_array,
        "marker_feature_ids": marker_feature_ids_array,
    }


def solve_pose(
    obj: np.ndarray, img: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    residual = projected.reshape(-1, 2) - img
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rvec.reshape(3, 1), tvec.reshape(3, 1), rmse


def calibrate_unity_K(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    initial_K: np.ndarray,
) -> tuple[np.ndarray, float, list[np.ndarray], list[np.ndarray]]:
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )
    rms, K, _dist, rvecs, tvecs = cv2.calibrateCamera(
        [p.astype(np.float32) for p in object_points],
        [p.reshape(-1, 1, 2).astype(np.float32) for p in image_points],
        IMAGE_SIZE,
        initial_K.copy(),
        np.zeros((8, 1), np.float64),
        flags=flags,
    )
    return K, float(rms), rvecs, tvecs


def pose_camera_in_board(
    rvec_tag_to_cam: np.ndarray, tvec_tag_to_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R_tag_to_cam, _ = cv2.Rodrigues(rvec_tag_to_cam)
    R_cam_to_board = R_tag_to_cam.T
    p_cam_in_board = -R_cam_to_board @ tvec_tag_to_cam
    return R_cam_to_board, p_cam_in_board.reshape(3)


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R_delta).magnitude()))


def shared_points(
    ids_a: np.ndarray,
    pts_a: np.ndarray,
    ids_b: np.ndarray,
    pts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_a = {int(i): p for i, p in zip(ids_a, pts_a)}
    map_b = {int(i): p for i, p in zip(ids_b, pts_b)}
    ids = np.array(sorted(set(map_a) & set(map_b)), np.int32)
    return (
        ids,
        np.asarray([map_a[int(i)] for i in ids], np.float64),
        np.asarray([map_b[int(i)] for i in ids], np.float64),
    )


def point_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def draw_correspondence(
    base: np.ndarray,
    real_points: np.ndarray,
    other_points: np.ndarray,
    ids: np.ndarray,
    title: str,
) -> np.ndarray:
    out = base.copy()
    for marker_id, p_real, p_other in zip(ids, real_points, other_points):
        a = tuple(np.round(p_real).astype(int))
        b = tuple(np.round(p_other).astype(int))
        cv2.line(out, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, a, 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, b, 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(int(marker_id)),
            (a[0] + 5, a[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(
        out, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--unity-dir", type=Path, default=DEFAULT_UNITY_DIR)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ids", default="1,2,3,4")
    args = parser.parse_args()

    real_dir = args.real_dir.resolve()
    unity_dir = args.unity_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = out_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    calibration = np.load(args.intrinsics)
    K_real = np.asarray(calibration["camera_matrix"], np.float64)
    distortion = np.asarray(calibration["dist_coeffs"], np.float64).reshape(-1)
    detector = create_detector()
    photo_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    samples: list[dict] = []
    for photo_id in photo_ids:
        real_path = real_dir / f"Photo{photo_id}_tag.jpg"
        unity_path = unity_dir / f"Photo{photo_id}_tag_Unity.jpg"
        raw_real = cv2.imread(str(real_path))
        raw_unity = cv2.imread(str(unity_path))
        if raw_real is None or raw_unity is None:
            print(f"skip Photo{photo_id}: missing input")
            continue
        if raw_real.shape[1::-1] != IMAGE_SIZE:
            raw_real = cv2.resize(raw_real, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        if raw_unity.shape[1::-1] != IMAGE_SIZE:
            raw_unity = cv2.resize(raw_unity, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

        real_und = cv2.undistort(raw_real, K_real, distortion, None, K_real)
        unity_180 = cv2.rotate(raw_unity, cv2.ROTATE_180)
        real_detection = detect_charuco(real_und, detector)
        unity_detection = detect_charuco(unity_180, detector)
        if real_detection is None or unity_detection is None:
            print(
                f"skip Photo{photo_id}: real={real_detection is not None}, "
                f"unity={unity_detection is not None}"
            )
            continue

        samples.append(
            {
                "id": photo_id,
                "real_path": real_path,
                "unity_path": unity_path,
                "real_und": real_und,
                "unity_180": unity_180,
                "real_obj": real_detection["pose_obj"],
                "real_img": real_detection["pose_img"],
                "real_source": real_detection["pose_source"],
                "real_charuco_count": real_detection["charuco_count"],
                "real_marker_count": real_detection["marker_count"],
                "real_marker_obj": real_detection["marker_obj"],
                "real_marker_img": real_detection["marker_img"],
                "real_marker_feature_ids": real_detection["marker_feature_ids"],
                "unity_obj": unity_detection["pose_obj"],
                "unity_img": unity_detection["pose_img"],
                "unity_source": unity_detection["pose_source"],
                "unity_charuco_count": unity_detection["charuco_count"],
                "unity_marker_count": unity_detection["marker_count"],
                "unity_marker_obj": unity_detection["marker_obj"],
                "unity_marker_img": unity_detection["marker_img"],
                "unity_marker_feature_ids": unity_detection["marker_feature_ids"],
            }
        )

    if len(samples) < 3:
        raise SystemExit(f"Need at least 3 valid views; found {len(samples)}")

    K_unity, unity_calib_rms, _cal_rvecs, _cal_tvecs = calibrate_unity_K(
        [s["unity_obj"] for s in samples],
        [s["unity_img"] for s in samples],
        K_real,
    )
    unity_K_leave_one_out = []
    if len(samples) >= 4:
        for held_out in range(len(samples)):
            train = [s for j, s in enumerate(samples) if j != held_out]
            K_loo, rms_loo, _, _ = calibrate_unity_K(
                [s["unity_obj"] for s in train],
                [s["unity_img"] for s in train],
                K_real,
            )
            unity_K_leave_one_out.append(
                {
                    "held_out_photo": samples[held_out]["id"],
                    "fx": float(K_loo[0, 0]),
                    "fy": float(K_loo[1, 1]),
                    "cx": float(K_loo[0, 2]),
                    "cy": float(K_loo[1, 2]),
                    "rms_px": rms_loo,
                }
            )

    rows: list[dict] = []
    delta_rotvecs: list[np.ndarray] = []
    delta_positions_mm: list[np.ndarray] = []
    delta_translations_mm: list[np.ndarray] = []
    pose_records: list[dict] = []
    for sample in samples:
        real_rvec, real_tvec, real_pose_rmse = solve_pose(
            sample["real_obj"], sample["real_img"], K_real
        )
        unity_rvec, unity_tvec, unity_pose_rmse = solve_pose(
            sample["unity_obj"], sample["unity_img"], K_unity
        )
        unity_rvec_sameK, unity_tvec_sameK, unity_sameK_rmse = solve_pose(
            sample["unity_obj"], sample["unity_img"], K_real
        )

        R_real_c2b, p_real_b = pose_camera_in_board(real_rvec, real_tvec)
        R_unity_c2b, p_unity_b = pose_camera_in_board(unity_rvec, unity_tvec)
        R_delta = R_real_c2b @ R_unity_c2b.T
        t_delta_mm = (
            p_real_b - R_delta @ p_unity_b
        ) * 1000.0
        delta_euler_xyz = Rotation.from_matrix(R_delta).as_euler("xyz", degrees=True)
        delta_rotvec = Rotation.from_matrix(R_delta).as_rotvec()
        delta_p_mm = (p_real_b - p_unity_b) * 1000.0
        delta_rotvecs.append(delta_rotvec)
        delta_positions_mm.append(delta_p_mm)
        delta_translations_mm.append(t_delta_mm)

        shared_ids, real_shared, unity_shared = shared_points(
            sample["real_marker_feature_ids"],
            sample["real_marker_img"],
            sample["unity_marker_feature_ids"],
            sample["unity_marker_img"],
        )
        raw_shared_rmse = point_rmse(real_shared, unity_shared)

        real_marker_obj_by_feature = {
            int(i): p
            for i, p in zip(
                sample["real_marker_feature_ids"], sample["real_marker_obj"]
            )
        }
        shared_obj = np.asarray(
            [real_marker_obj_by_feature[int(i)] for i in shared_ids], np.float32
        )
        unity_pose_projected_to_real, _ = cv2.projectPoints(
            shared_obj, unity_rvec, unity_tvec, K_real, None
        )
        unity_pose_projected_to_real = unity_pose_projected_to_real.reshape(-1, 2)
        physical_before_rmse = point_rmse(real_shared, unity_pose_projected_to_real)

        corrected_projected, _ = cv2.projectPoints(
            shared_obj, real_rvec, real_tvec, K_real, None
        )
        corrected_projected = corrected_projected.reshape(-1, 2)
        physical_after_rmse = point_rmse(real_shared, corrected_projected)

        raw_panel = draw_correspondence(
            sample["real_und"],
            real_shared,
            unity_shared,
            shared_ids,
            f"Photo{sample['id']} raw pixels: green=real, red=Unity rot180",
        )
        physical_panel = draw_correspondence(
            sample["real_und"],
            real_shared,
            unity_pose_projected_to_real,
            shared_ids,
            (
                f"Photo{sample['id']} physical baseline: Unity pose projected with real K "
                f"(RMSE {physical_before_rmse:.2f}px)"
            ),
        )
        corrected_panel = draw_correspondence(
            sample["real_und"],
            real_shared,
            corrected_projected,
            shared_ids,
            (
                f"Photo{sample['id']} per-photo corrected camera "
                f"(RMSE {physical_after_rmse:.2f}px)"
            ),
        )
        diagnostic = cv2.hconcat(
            [
                cv2.resize(raw_panel, (540, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(physical_panel, (540, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(corrected_panel, (540, 360), interpolation=cv2.INTER_AREA),
            ]
        )
        cv2.imwrite(str(diagnostic_dir / f"Photo{sample['id']}_physical.png"), diagnostic)

        row = {
            "id": sample["id"],
            "real_pose_feature_source": sample["real_source"],
            "unity_pose_feature_source": sample["unity_source"],
            "real_charuco_corners": sample["real_charuco_count"],
            "unity_charuco_corners": sample["unity_charuco_count"],
            "real_aruco_markers": sample["real_marker_count"],
            "unity_aruco_markers": sample["unity_marker_count"],
            "shared_aruco_corners": len(shared_ids),
            "real_pnp_rmse_px": real_pose_rmse,
            "unity_pnp_rmse_px": unity_pose_rmse,
            "unity_same_realK_pnp_rmse_px": unity_sameK_rmse,
            "raw_shared_corner_rmse_px": raw_shared_rmse,
            "physical_before_correction_rmse_px": physical_before_rmse,
            "physical_after_per_photo_correction_rmse_px": physical_after_rmse,
            "real_camera_x_mm": p_real_b[0] * 1000.0,
            "real_camera_y_mm": p_real_b[1] * 1000.0,
            "real_camera_z_mm": p_real_b[2] * 1000.0,
            "unity_camera_x_mm": p_unity_b[0] * 1000.0,
            "unity_camera_y_mm": p_unity_b[1] * 1000.0,
            "unity_camera_z_mm": p_unity_b[2] * 1000.0,
            "delta_position_x_mm": delta_p_mm[0],
            "delta_position_y_mm": delta_p_mm[1],
            "delta_position_z_mm": delta_p_mm[2],
            "delta_rigid_translation_x_mm": t_delta_mm[0],
            "delta_rigid_translation_y_mm": t_delta_mm[1],
            "delta_rigid_translation_z_mm": t_delta_mm[2],
            "delta_rotation_angle_deg": rotation_angle_deg(R_delta),
            "delta_roll_x_deg": delta_euler_xyz[0],
            "delta_pitch_y_deg": delta_euler_xyz[1],
            "delta_yaw_z_deg": delta_euler_xyz[2],
        }
        rows.append(row)
        pose_records.append(
            {
                "R_unity_c2b": R_unity_c2b,
                "p_unity_b": p_unity_b,
                "real_shared": real_shared,
                "shared_obj": shared_obj,
            }
        )

    delta_rotations = Rotation.from_rotvec(np.asarray(delta_rotvecs))
    mean_rotation = delta_rotations.mean()
    mean_rotation_errors_deg = np.degrees(
        (mean_rotation.inv() * delta_rotations).magnitude()
    )
    position_deltas = np.asarray(delta_positions_mm)
    mean_position_delta = position_deltas.mean(axis=0)
    position_spread = np.linalg.norm(position_deltas - mean_position_delta, axis=1)
    rigid_translations = np.asarray(delta_translations_mm)
    mean_rigid_translation = rigid_translations.mean(axis=0)
    rigid_translation_spread = np.linalg.norm(
        rigid_translations - mean_rigid_translation, axis=1
    )

    # Leave-one-out test: estimate one rigid correction from the other photos,
    # then predict the held-out photo. This is the non-tautological transfer test.
    loo_rmses = []
    for held_out, (row, pose) in enumerate(zip(rows, pose_records)):
        train_indices = [j for j in range(len(rows)) if j != held_out]
        R_loo = Rotation.from_rotvec(
            np.asarray([delta_rotvecs[j] for j in train_indices])
        ).mean().as_matrix()
        t_loo_mm = rigid_translations[train_indices].mean(axis=0)
        R_corrected_c2b = R_loo @ pose["R_unity_c2b"]
        p_corrected_b = (
            R_loo @ pose["p_unity_b"] + t_loo_mm / 1000.0
        )
        R_tag_to_cam_corrected = R_corrected_c2b.T
        t_tag_to_cam_corrected = (
            -R_tag_to_cam_corrected @ p_corrected_b.reshape(3, 1)
        )
        rvec_corrected, _ = cv2.Rodrigues(R_tag_to_cam_corrected)
        predicted, _ = cv2.projectPoints(
            pose["shared_obj"],
            rvec_corrected,
            t_tag_to_cam_corrected,
            K_real,
            None,
        )
        loo_rmse = point_rmse(
            pose["real_shared"], predicted.reshape(-1, 2)
        )
        row["loo_global_pose_correction_rmse_px"] = loo_rmse
        loo_rmses.append(loo_rmse)

    fieldnames = list(rows[0].keys())
    with (out_dir / "per_photo_camera_correction.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    W, H = IMAGE_SIZE
    summary = {
        "dataset": real_dir.name,
        "unity_folder": unity_dir.name,
        "photos_analyzed": [row["id"] for row in rows],
        "image_size_px": [W, H],
        "real_K": K_real.tolist(),
        "real_distortion": distortion.tolist(),
        "unity_effective_K_zero_distortion": K_unity.tolist(),
        "unity_multiview_calibration_rms_px": unity_calib_rms,
        "unity_K_leave_one_out": unity_K_leave_one_out,
        "unity_K_leave_one_out_std_px": {
            key: float(
                np.std([entry[key] for entry in unity_K_leave_one_out], ddof=1)
            )
            if len(unity_K_leave_one_out) > 1
            else None
            for key in ("fx", "fy", "cx", "cy")
        },
        "intrinsic_delta": {
            "fx_px": float(K_unity[0, 0] - K_real[0, 0]),
            "fy_px": float(K_unity[1, 1] - K_real[1, 1]),
            "cx_px": float(K_unity[0, 2] - K_real[0, 2]),
            "cy_px": float(K_unity[1, 2] - K_real[1, 2]),
            "fx_ratio_unity_over_real": float(K_unity[0, 0] / K_real[0, 0]),
            "fy_ratio_unity_over_real": float(K_unity[1, 1] / K_real[1, 1]),
        },
        "unity_effective_fov_deg": {
            "horizontal": float(
                math.degrees(2.0 * math.atan(W / (2.0 * K_unity[0, 0])))
            ),
            "vertical": float(
                math.degrees(2.0 * math.atan(H / (2.0 * K_unity[1, 1])))
            ),
        },
        "real_fov_deg": {
            "horizontal": float(
                math.degrees(2.0 * math.atan(W / (2.0 * K_real[0, 0])))
            ),
            "vertical": float(
                math.degrees(2.0 * math.atan(H / (2.0 * K_real[1, 1])))
            ),
        },
        "delta_pose_consistency": {
            "mean_delta_position_mm": mean_position_delta.tolist(),
            "per_photo_position_deviation_from_mean_mm": position_spread.tolist(),
            "max_position_deviation_from_mean_mm": float(position_spread.max()),
            "mean_delta_rotation_xyzw": mean_rotation.as_quat().tolist(),
            "per_photo_rotation_deviation_from_mean_deg": mean_rotation_errors_deg.tolist(),
            "max_rotation_deviation_from_mean_deg": float(mean_rotation_errors_deg.max()),
            "mean_rigid_translation_mm": mean_rigid_translation.tolist(),
            "per_photo_rigid_translation_deviation_from_mean_mm": rigid_translation_spread.tolist(),
            "max_rigid_translation_deviation_from_mean_mm": float(
                rigid_translation_spread.max()
            ),
        },
        "leave_one_out_global_correction": {
            "per_photo_rmse_px": loo_rmses,
            "mean_rmse_px": float(np.mean(loo_rmses)),
            "max_rmse_px": float(np.max(loo_rmses)),
            "meaning": (
                "Each photo is predicted using a single rigid delta pose averaged "
                "from the other three photos."
            ),
        },
        "interpretation_note": (
            "Delta poses use the OpenCV ChArUco board frame and cannot be copied "
            "directly into a Unity Transform without axis/handedness conversion."
        ),
    }
    with (out_dir / "camera_correction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Analyzed {len(rows)} photos")
    print("Real K:")
    print(K_real)
    print("Estimated Unity effective K (zero distortion):")
    print(K_unity)
    print(f"Unity multiview calibration RMS: {unity_calib_rms:.4f} px")
    if unity_K_leave_one_out:
        print(
            "Unity K leave-one-out std: "
            + ", ".join(
                f"{key}={np.std([entry[key] for entry in unity_K_leave_one_out], ddof=1):.2f}px"
                for key in ("fx", "fy", "cx", "cy")
            )
        )
    print(
        "Delta-pose spread: "
        f"max rigid translation={rigid_translation_spread.max():.3f} mm, "
        f"max rotation={mean_rotation_errors_deg.max():.3f} deg"
    )
    for row in rows:
        print(
            f"Photo{row['id']}: shared marker corners={row['shared_aruco_corners']} "
            f"raw={row['raw_shared_corner_rmse_px']:.2f}px "
            f"physical-before={row['physical_before_correction_rmse_px']:.2f}px "
            f"corrected={row['physical_after_per_photo_correction_rmse_px']:.2f}px "
            f"LOO-global={row['loo_global_pose_correction_rmse_px']:.2f}px "
            f"dpos=({row['delta_position_x_mm']:+.2f},"
            f"{row['delta_position_y_mm']:+.2f},"
            f"{row['delta_position_z_mm']:+.2f})mm "
            f"drot={row['delta_rotation_angle_deg']:.2f}deg"
        )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
