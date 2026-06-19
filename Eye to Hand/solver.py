#!/usr/bin/env python3

import os
import json
import numpy as np
import cv2

HAND_DIR = "hand_data"
EYE_DIR = "eye_data"

# We will store the INVERTED robot poses here
R_base2gripper_list = []
t_base2gripper_list = []
R_target2cam_list = []
t_target2cam_list = []

# 1. Find matching sample IDs
print("Looking for paired calibration data...")
hand_files = sorted([f for f in os.listdir(HAND_DIR) if f.endswith('.json')])
eye_files = sorted([f for f in os.listdir(EYE_DIR) if f.endswith('.json')])

hand_ids = set([int(f.split('_')[1].split('.')[0]) for f in hand_files])
eye_ids = set([int(f.split('_')[1].split('.')[0]) for f in eye_files])

common_ids = sorted(list(hand_ids.intersection(eye_ids)))
print(f"Found {len(common_ids)} perfectly paired hand and eye samples.")

if len(common_ids) < 3:
    print("Error: You need at least 3 pairs for calibration.")
    exit()

# 2. Load and INVERT the data
for idx in common_ids:
    # --- Robot Data ---
    with open(os.path.join(HAND_DIR, f"hand_{idx:03d}.json"), 'r') as f:
        hand_data = json.load(f)
        
        # This is Gripper-in-Base (T_base_to_gripper)
        R_g2b = np.array(hand_data["R"], dtype=np.float64)
        t_g2b = np.array(hand_data["t"], dtype=np.float64).reshape(3, 1)
        
        # TRICK OPENCV FOR EYE-TO-HAND: WE MUST INVERT IT (Base-in-Gripper)
        R_b2g = R_g2b.T
        t_b2g = -R_b2g @ t_g2b
        
        R_base2gripper_list.append(R_b2g)
        t_base2gripper_list.append(t_b2g)

    # --- Camera Data ---
    with open(os.path.join(EYE_DIR, f"eye_{idx:03d}.json"), 'r') as f:
        eye_data = json.load(f)
        
        # This is Target-in-Camera (T_cam_to_target). We do NOT invert this.
        R_target2cam_list.append(np.array(eye_data["R"], dtype=np.float64))
        t_target2cam_list.append(np.array(eye_data["t"], dtype=np.float64).reshape(3, 1))

# 3. Hand-Eye Calibration (Eye-to-Hand)
print("\nProcessing mathematical calibration (Tsai method)...")

# By passing the INVERTED robot poses first, OpenCV correctly computes Camera -> Base
R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_gripper2base=R_base2gripper_list,
    t_gripper2base=t_base2gripper_list,
    R_target2cam=R_target2cam_list,
    t_target2cam=t_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

# Build the final 4x4 Homogeneous Transformation Matrix
T_cam2base = np.eye(4)
T_cam2base[:3, :3] = R_cam2base
T_cam2base[:3, 3] = t_cam2base.flatten()

print("\n================ FINAL CALIBRATION MATRIX ================")
print("Homogeneous Transformation Matrix (T_cam2base):")
np.set_printoptions(suppress=True, precision=5)
print(T_cam2base)

print("\nPhysical Distance from Robot Base to Camera Lens:")
print(f"  X offset: {T_cam2base[0, 3]:.4f} meters")
print(f"  Y offset: {T_cam2base[1, 3]:.4f} meters")
print(f"  Z offset: {T_cam2base[2, 3]:.4f} meters")

np.save("T_cam2base.npy", T_cam2base)
print("\nSaved calibration matrix to T_cam2base.npy")
