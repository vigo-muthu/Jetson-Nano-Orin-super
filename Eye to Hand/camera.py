#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import json

# ============================================================
# CONFIGURATION
# ============================================================

# 7x7 checkerboard squares -> 6x6 inner corners
CHECKERBOARD = (4, 3)

# 2.82.8 cm square size
SQUARE_SIZE = 0.030   # meters

SAVE_DIR = "eye_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# REALSENSE D415 SETUP
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

intrinsics = color_profile.get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(
    intrinsics.coeffs,
    dtype=np.float64
).reshape(-1, 1)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs.flatten())

# ============================================================
# CHECKERBOARD 3D POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ============================================================
# FIND NEXT FILE NUMBER
# ============================================================

existing = []

for f in os.listdir(SAVE_DIR):
    if f.startswith("eye_") and f.endswith(".json"):
        try:
            existing.append(
                int(f.split("_")[1].split(".")[0])
            )
        except:
            pass

sample_id = 0 if len(existing) == 0 else max(existing) + 1

print(f"\nStarting sample index: {sample_id}")

# ============================================================
# MAIN LOOP
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

print("\n")
print("===================================")
print("s  -> save pose")
print("ESC -> exit")
print("===================================")

while True:

    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    image = np.asanyarray(
        color_frame.get_data()
    )

    display = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    pose_valid = False

    if found:

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(
            display,
            CHECKERBOARD,
            corners,
            found
        )

        success, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            pose_valid = True

            R, _ = cv2.Rodrigues(rvec)

            T = np.eye(4)

            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            x = tvec[0][0]
            y = tvec[1][0]
            z = tvec[2][0]

            cv2.putText(
                display,
                f"X={x:.3f} m",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Y={y:.3f} m",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Z={z:.3f} m",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Samples: {sample_id}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    cv2.imshow(
        "Eye-to-Hand Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    if key == ord('s'):

        if not pose_valid:
            print(
                "\nCheckerboard not detected. "
                "Move board and try again."
            )
            continue

        data = {
            "sample_id": sample_id,
            "R": R.tolist(),
            "t": tvec.tolist(),
            "T": T.tolist(),
            "rvec": rvec.tolist()
        }

        filename = os.path.join(
            SAVE_DIR,
            f"eye_{sample_id:03d}.json"
        )

        with open(filename, "w") as f:
            json.dump(
                data,
                f,
                indent=4
            )

        print(
            f"\nSaved: {filename}"
        )

        print(
            f"Translation (m): "
            f"{x:.4f}, {y:.4f}, {z:.4f}"
        )

        sample_id += 1

    # ========================================================
    # EXIT
    # ========================================================

    elif key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
