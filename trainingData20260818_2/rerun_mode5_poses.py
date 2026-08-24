"""Offline Mode 5 pose log + Unity CSV, same math as capsule_calibration.run_tag_pose_logger."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MODE5_CHARUCO_BOARD = cv2.aruco.CharucoBoard((5, 5), 0.010, 0.0075, ARUCO_DICT)
BOARD_CENTER_OPENCV_M = np.array([0.025, 0.025, 0.0], np.float64).reshape(3, 1)
TAG_FROM_OPENCV = np.diag([1.0, -1.0, -1.0])
MAX_REPROJECTION_RMSE_PX = 1.0


def create_detector() -> cv2.aruco.CharucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.CharucoDetector(
        MODE5_CHARUCO_BOARD, cv2.aruco.CharucoParameters(), params
    )


def solve_pose(charuco_corners, charuco_ids, K, dist, min_corners):
    count = 0 if charuco_corners is None else len(charuco_corners)
    if charuco_corners is None or charuco_ids is None or count < min_corners:
        return None, f"need {min_corners} ChArUco corners; found {count}"

    obj_pts, img_pts = MODE5_CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, "PnP solve failed"
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    residual = img_pts.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if rmse > MAX_REPROJECTION_RMSE_PX:
        return None, f"RMSE {rmse:.2f}px exceeds {MAX_REPROJECTION_RMSE_PX:.2f}px"

    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    tvec_center = tvec + R_board_to_cam @ BOARD_CENTER_OPENCV_M
    R_tag_to_cam = R_board_to_cam @ TAG_FROM_OPENCV
    rvec_tag, _ = cv2.Rodrigues(R_tag_to_cam)
    return (rvec_tag, tvec_center, rmse, len(obj_pts)), None

ORIGINAL = np.array([-0.358725, -2.2282, 13.2305], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Mode 5 poses using a dataset's intrinsics NPZ.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--min-corners",
        type=int,
        default=8,
        help="Minimum detected ChArUco corners for offline single-image PnP.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    npz_path = data_dir / "capsule_intrinsics.npz"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else data_dir / "camera_pose_relative_to_tag.csv"
    )

    data = np.load(npz_path)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    detector = create_detector()
    photos = sorted(data_dir.glob("Photo*_tag.jpg"), key=lambda p: int(p.stem[5:-4]))
    rows = []
    print("Mode 5 board: 5x5 squares x 10 mm, origin = physical center, tag = diag(1,-1,-1)@OpenCV")
    print(f"intrinsics = {npz_path}")
    print()
    for photo in photos:
        frame = cv2.imread(str(photo))
        if frame is None:
            print(f"skip {photo.name}: unreadable")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _mc, _mi = detector.detectBoard(gray)
        solved, reason = solve_pose(
            charuco_corners, charuco_ids, K, dist, args.min_corners
        )
        if solved is None:
            print(f"{photo.name}: REJECTED ({reason})")
            continue
        rvec, tvec, rmse, n_corners = solved
        R_tag_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_tag = R_tag_to_cam.T
        P_mm = (-R_cam_to_tag @ tvec).flatten() * 1000.0
        euler = R.from_matrix(R_cam_to_tag).as_euler("xyz", degrees=True)
        quat = R.from_matrix(R_cam_to_tag).as_quat()  # xyzw
        rows.append(
            {
                "image_file": photo.name,
                "tag_id": "charuco_50mm_5x5",
                "camera_x_mm": float(P_mm[0]),
                "camera_y_mm": float(P_mm[1]),
                "camera_z_mm": float(P_mm[2]),
                "camera_orientation_reference_frame": "camera axes relative to 50 mm ChArUco board center",
                "euler_rotation_convention": "extrinsic XYZ; scipy Rotation.as_euler('xyz')",
                "camera_roll_deg": float(euler[0]),
                "camera_pitch_deg": float(euler[1]),
                "camera_yaw_deg": float(euler[2]),
                "camera_qx": float(quat[0]),
                "camera_qy": float(quat[1]),
                "camera_qz": float(quat[2]),
                "camera_qw": float(quat[3]),
            }
        )
        n_c = 0 if charuco_corners is None else len(charuco_corners)
        print(
            f"{photo.name:18s} corners={n_c:2d} rmse={rmse:.3f}px  "
            f"P_mm=({P_mm[0]:+7.3f},{P_mm[1]:+7.3f},{P_mm[2]:+7.3f})  "
            f"rpy=({euler[0]:+7.3f},{euler[1]:+7.3f},{euler[2]:+7.3f})"
        )

    if not rows:
        print("No poses.")
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
