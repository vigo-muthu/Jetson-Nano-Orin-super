"""
Step 1: Data collection using a standard CHECKERBOARD.
Configured for: 7x7 Squares (6x6 Internal Corners), 6mm square size.
Bypasses ROS2 by utilizing manual free-move mode.
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from pymycobot.mycobot import MyCobot

# ---------------- CONFIG ----------------
SAVE_DIR = "./calib_data"
NUM_POSES = 20
ROBOT_PORT = "/dev/ttyUSB0"
ROBOT_BAUD = 1000000

# ---- CALIB.IO CHECKERBOARD CONFIG (7x7 Squares -> 6x6 Inner Corners) ----
PATTERN_X = 6      
PATTERN_Y = 6      
SQUARE_LEN = 0.006  # 6mm checker width converted to meters

os.makedirs(SAVE_DIR, exist_ok=True)

# Generate static 3D target points
objp = np.zeros((PATTERN_X * PATTERN_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_X, 0:PATTERN_Y].T.reshape(-1, 2) * SQUARE_LEN
np.save(os.path.join(SAVE_DIR, "chessboard_objp.npy"), objp)

# ---------------- INIT REALSENSE ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)
np.save(os.path.join(SAVE_DIR, "camera_matrix.npy"), camera_matrix)
np.save(os.path.join(SAVE_DIR, "dist_coeffs.npy"), dist_coeffs)

# ---------------- INIT ROBOT ----------------
print("Connecting to robot arm...")
mc = MyCobot(ROBOT_PORT, ROBOT_BAUD)
time.sleep(2)

print("Releasing servos... You can now move the robot gently by hand!")
mc.release_all_servos()
time.sleep(1)

robot_poses = []
img_count = 0

print(f"\nMove the robot hand-guided to {NUM_POSES} diverse poses.")
print("Ensure the ENTIRE checkerboard is in view. Press ENTER to capture, 'q' to quit.")

try:
    while img_count < NUM_POSES:
        for _ in range(5):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (PATTERN_X, PATTERN_Y), None)
        
        display_img = img.copy()
        if ret:
            cv2.drawChessboardCorners(display_img, (PATTERN_X, PATTERN_Y), corners, ret)
            cv2.putText(display_img, f"MATCH [{img_count}/{NUM_POSES}] - Press Enter", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "BOARD NOT FULLY VISIBLE", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Calibration capture", display_img)
        key = cv2.waitKey(1)

        if key == 13:  # Enter Key
            if not ret:
                print("Cannot save: Grid pattern is not fully visible to the camera loop.")
                continue
                
            coords = mc.get_coords()
            
            # Robust verification of data packet
            if not coords or coords == -1 or isinstance(coords, int) or len(coords) < 6:
                print(f"Serial read glitch (received: {coords}). Hold still and retry hitting ENTER.")
                continue

            fname = os.path.join(SAVE_DIR, f"img_{img_count:02d}.png")
            cv2.imwrite(fname, img)  
            robot_poses.append(coords)
            print(f"[{img_count}] Saved {fname}, pose={coords}")
            img_count += 1

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Safely handle potential file structure complaints from numpy
    if len(robot_poses) > 0:
        valid_poses = np.array(robot_poses, dtype=np.float32)
        np.save(os.path.join(SAVE_DIR, "robot_poses.npy"), valid_poses)
        print(f"Successfully saved {len(valid_poses)} poses.")
    else:
        print("No data collected.")
