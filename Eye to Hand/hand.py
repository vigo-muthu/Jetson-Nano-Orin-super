#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import json
import time
import numpy as np
from pymycobot.mycobot import MyCobot
from scipy.spatial.transform import Rotation

# ============================================================
# HELPER TO LOCK / HOLD POSITION
# ============================================================
def hold_position(robot_obj):
    """Re-engages all joint motors to lock the arm in its current pose."""
    robot_obj.power_on()
    time.sleep(0.1)
    for idx in range(1, 7):
        try:
            robot_obj.focus_servo(idx)
            time.sleep(0.05)
        except AttributeError:
            pass
    time.sleep(0.1)

# ============================================================
# CONFIGURATION & INITIALIZATION
# ============================================================
SAVE_DIR = "hand_data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Connecting to JetCobot...")
mc = MyCobot('/dev/ttyUSB0', 1000000)
time.sleep(1)

print("Initializing motor states...")
hold_position(mc)

print("============================================================")
print("MANUAL HAND-GUIDED CALIBRATION OPERATING MODE:")
print("  Type 'f' + ENTER -> Free Drive (Relax arm motors to move it)")
print("  Type 'h' + ENTER -> Hold Position (Lock motors to hold pose)")
print("  Type 's' + ENTER -> Save current coordinates to JSON")
print("  Type 'q' + ENTER -> Quit script safely")
print("============================================================")

sample_id = 0

while True:
    print(f"\n--- [Sample Progress: {sample_id} saved files] ---")
    
    # Using standard input prevents terminal buffer skipping
    cmd = input("Enter Command (f / h / s / q): ").strip().lower()
    
    if cmd == 'q':
        print("\nExiting and re-engaging motors for safety...")
        hold_position(mc)
        break
        
    elif cmd == 'f':
        print("\n[FREE DRIVE] Relaxing motors. Move the arm manually by hand...")
        mc.release_all_servos()
        
    elif cmd == 'h':
        print("\n[HOLD POSITION] Locking motors in place.")
        hold_position(mc)
        
    elif cmd == 's':
        print("\n[SAVE] Attempting to read stable pose data...")
        print("Ensuring serial line is settled...")
        time.sleep(0.5)
        
        coords = None
        for i in range(5):  # Retry loop for serial stability
            raw_coords = mc.get_coords()
            
            if isinstance(raw_coords, (list, tuple, np.ndarray)) and len(raw_coords) == 6:
                coords = raw_coords
                break
                
            print(f"  ...Serial coordinate read returned unexpected type or failed. Attempt {i+1}/5. Received: {raw_coords}")
            time.sleep(0.2)

        if coords is None:
            print("[ERROR] Could not read valid coordinates from the robot. Try saving [s] again!")
            continue

        # Read joint angles for safety verification logging
        angles = mc.get_angles() or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Extract positions and convert mm to meters
        x = coords[0] / 1000.0
        y = coords[1] / 1000.0
        z = coords[2] / 1000.0

        rx = coords[3]
        ry = coords[4]
        rz = coords[5]

        # Process rotation matrix
        R = Rotation.from_euler(
            'xyz',
            [rx, ry, rz],
            degrees=True
        ).as_matrix()

        t = np.array([[x], [y], [z]])

        # Format payload JSON
        data = {
            "sample_id": sample_id,
            "read_joints": angles,
            "read_coords": coords,
            "R": R.tolist(),
            "t": t.tolist()
        }

        filename = os.path.join(
            SAVE_DIR,
            f"hand_{sample_id:03d}.json"
        )

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"[SUCCESS] Saved file: {filename}")
        print(f"Captured Position: X={coords[0]:.1f}mm, Y={coords[1]:.1f}mm, Z={coords[2]:.1f}mm")
        
        sample_id += 1
        
    elif cmd == '':
        continue
    else:
        print(f"[INVALID] '{cmd}' is not a valid command. Use f, h, s, or q.")

print("\n============================================================")
print(f"Manual session completed! Saved {sample_id} total configurations in '{SAVE_DIR}'.")
