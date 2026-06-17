"""
Step 2: Solve hand-to-eye (eye-to-hand) calibration.
Configured for: 7x7 Squares (6x6 Internal Corners).
Reads images + robot_poses.npy saved by collect_calibration_data.py.
Outputs T_cam2base.npy (4x4): camera frame -> robot base frame.
"""

import os
import glob
import numpy as np
import cv2

SAVE_DIR = "./calib_data"

PATTERN_X = 6
PATTERN_Y = 6

camera_matrix = np.load(os.path.join(SAVE_DIR, "camera_matrix.npy"))
dist_coeffs = np.load(os.path.join(SAVE_DIR, "dist_coeffs.npy"))
robot_poses = np.load(os.path.join(SAVE_DIR, "robot_poses.npy"))  # [N,6] x,y,z(mm), rx,ry,rz(deg)
obj_points = np.load(os.path.join(SAVE_DIR, "chessboard_objp.npy"))

img_files = sorted(glob.glob(os.path.join(SAVE_DIR, "img_*.png")))
assert len(img_files) == len(robot_poses), "Mismatch between number of images and poses"


def euler_xyz_deg_to_R(rx, ry, rz):
    """MyCobot get_coords() Euler angles in degrees conversion."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

for fpath, pose in zip(img_files, robot_poses):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Process standard checkerboard grid layout
    ret, corners = cv2.findChessboardCorners(gray, (PATTERN_X, PATTERN_Y), None)

    if not ret:
        print(f"Skipping {fpath}: Chessboard pattern extraction failed.")
        continue

    img_points = corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not ok:
        print(f"Skipping {fpath}: solvePnP failed")
        continue

    R_target2cam.append(cv2.Rodrigues(rvec)[0])
    t_target2cam.append(tvec.flatten())

    x, y, z, rx, ry, rz = pose
    R_gripper2base.append(euler_xyz_deg_to_R(rx, ry, rz))
    t_gripper2base.append(np.array([x, y, z]) / 1000.0)  # mm -> meters

print(f"Using {len(R_target2cam)} / {len(img_files)} valid pose pairs for calibration")

# --- Eye-to-hand inversion pipeline ---
R_base2gripper = [R.T for R in R_gripper2base]
t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]

R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_base2gripper, t_base2gripper,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_cam2base = np.eye(4)
T_cam2base[:3, :3] = R_cam2base
T_cam2base[:3, 3] = t_cam2base.flatten()

print("\nT_cam2base (camera frame -> robot base frame):")
print(T_cam2base)

np.save(os.path.join(SAVE_DIR, "T_cam2base.npy"), T_cam2base)
print(f"\nSaved matrix to {SAVE_DIR}/T_cam2base.npy")
