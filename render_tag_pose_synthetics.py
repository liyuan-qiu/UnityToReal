"""Render mode-5 tag-pose measurements as synthetic ArUco images.

The pose CSV stores the camera pose in the tag coordinate system. This script
inverts that pose, projects a marker ID using the saved capsule intrinsics, and
optionally creates a real/synthetic/overlay comparison for every source photo.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CSV = Path("camera_pose_relative_to_tag.csv")
DEFAULT_INTRINSICS = Path("capsule_intrinsics.npz")
DEFAULT_OUTPUT_DIRECTORY = Path("tag_pose_synthetic_output")
MARKER_SIZE_MM = 17.0
DEFAULT_IMAGE_WIDTH = 1080
DEFAULT_IMAGE_HEIGHT = 720
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Mode-5 camera pose CSV.")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS,
                        help="NPZ containing camera_matrix and dist_coeffs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="Directory for rendered images.")
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM,
                        help="Black ArUco marker side length in millimetres.")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help="Output width when a row's source photo is unavailable.")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help="Output height when a row's source photo is unavailable.")
    return parser.parse_args()


def load_calibration(path):
    calibration = np.load(path)
    return calibration["camera_matrix"], calibration["dist_coeffs"]


def make_marker_image(tag_id, side_pixels=800):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side_pixels)
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def tag_to_camera_pose(row):
    """Convert the logged camera-in-tag pose into OpenCV tag-to-camera pose."""
    camera_position_mm = np.array([
        float(row["camera_x_mm"]),
        float(row["camera_y_mm"]),
        float(row["camera_z_mm"]),
    ])
    camera_euler_deg = np.array([
        float(row["camera_roll_deg"]),
        float(row["camera_pitch_deg"]),
        float(row["camera_yaw_deg"]),
    ])

    rotation_camera_to_tag = Rotation.from_euler(
        "xyz", camera_euler_deg, degrees=True
    ).as_matrix()
    rotation_tag_to_camera = rotation_camera_to_tag.T
    translation_tag_to_camera_m = (
        -rotation_tag_to_camera @ camera_position_mm.reshape(3, 1) / 1000.0
    )
    rvec, _ = cv2.Rodrigues(rotation_tag_to_camera)
    return rvec, translation_tag_to_camera_m


def projected_marker_corners(rvec, tvec, camera_matrix, dist_coeffs, marker_size_mm):
    half_size_m = marker_size_mm / 2000.0
    # This order exactly matches the object points used by mode 5.
    object_corners = np.array([
        [-half_size_m, half_size_m, 0.0],
        [half_size_m, half_size_m, 0.0],
        [half_size_m, -half_size_m, 0.0],
        [-half_size_m, -half_size_m, 0.0],
    ], dtype=np.float32)
    image_corners, _ = cv2.projectPoints(
        object_corners, rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_corners.reshape(4, 2).astype(np.float32)


def render_marker(image_shape, marker, projected_corners):
    height, width = image_shape[:2]
    rendered = np.full((height, width, 3), 255, dtype=np.uint8)
    marker_size = marker.shape[0]
    source_corners = np.array([
        [0, 0],
        [marker_size - 1, 0],
        [marker_size - 1, marker_size - 1],
        [0, marker_size - 1],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source_corners, projected_corners)
    warped_marker = cv2.warpPerspective(
        marker, homography, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    marker_mask = cv2.warpPerspective(
        np.full((marker_size, marker_size), 255, dtype=np.uint8), homography,
        (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    rendered[marker_mask > 0] = warped_marker[marker_mask > 0]
    return rendered


def comparison_image(real_image, synthetic):
    overlay = cv2.addWeighted(real_image, 0.5, synthetic, 0.5, 0.0)
    label_height = 34
    labelled = []
    for image, label in ((real_image, "Real photo (rotated for mode 5)"),
                         (synthetic, "Synthetic from logged pose"),
                         (overlay, "50% overlay")):
        panel = cv2.copyMakeBorder(image, label_height, 0, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(panel)
    return cv2.hconcat(labelled)


def source_image_for_comparison(image_path):
    if not image_path.is_file():
        return None
    source = cv2.imread(str(image_path))
    if source is None:
        return None
    # The logger ran solvePnP on load_capsule_frame(), which rotates 180 degrees.
    return cv2.rotate(source, cv2.ROTATE_180)


def main():
    arguments = parse_arguments()
    if not arguments.csv.is_file():
        raise FileNotFoundError(f"Pose CSV was not found: {arguments.csv}")
    if not arguments.intrinsics.is_file():
        raise FileNotFoundError(f"Intrinsics file was not found: {arguments.intrinsics}")
    camera_matrix, dist_coeffs = load_calibration(arguments.intrinsics)
    arguments.output.mkdir(parents=True, exist_ok=True)
    fallback_shape = (arguments.height, arguments.width, 3)
    rendered_count = 0
    comparison_count = 0

    with arguments.csv.open(newline="", encoding="utf-8") as pose_file:
        for row_index, row in enumerate(csv.DictReader(pose_file), start=1):
            image_path = Path(row["image_file"])
            real_image = source_image_for_comparison(image_path)
            image_shape = real_image.shape if real_image is not None else fallback_shape
            tag_id = int(row["tag_id"])
            marker = make_marker_image(tag_id)
            rvec, tvec = tag_to_camera_pose(row)
            corners = projected_marker_corners(
                rvec, tvec, camera_matrix, dist_coeffs, arguments.marker_size_mm
            )
            synthetic = render_marker(image_shape, marker, corners)
            stem = f"{row_index:02d}_{image_path.stem or f'tag_{tag_id}'}"
            cv2.imwrite(str(arguments.output / f"{stem}_synthetic.png"), synthetic)
            rendered_count += 1

            if real_image is not None:
                comparison = comparison_image(real_image, synthetic)
                cv2.imwrite(str(arguments.output / f"{stem}_comparison.png"), comparison)
                comparison_count += 1

    print(f"Rendered {rendered_count} synthetic image(s) to: {arguments.output.resolve()}")
    print(f"Created {comparison_count} real/synthetic comparison image(s).")


if __name__ == "__main__":
    main()