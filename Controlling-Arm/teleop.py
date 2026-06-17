import cv2
import numpy as np
import math
import time
from pymycobot import MyCobot

print("Connecting to JetCobot...")
mc = MyCobot('/dev/ttyUSB0', 1000000)
time.sleep(2)

# Smaller step for safe testing
STEP = 5.0      # mm
SPEED = 40

# Linux/OpenCV key codes
UP_ARROW    = 65362
DOWN_ARROW  = 65364
LEFT_ARROW  = 65361
RIGHT_ARROW = 65363
PAGE_UP     = 65365
PAGE_DOWN   = 65366


def rotation_matrix_from_rpy(roll_deg, pitch_deg, yaw_deg):

    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [0,                1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1]
    ])

    # Assume rx=roll, ry=pitch, rz=yaw
    return Rz @ Ry @ Rx


def move_tool(dx_tool, dy_tool, dz_tool):

    coords = mc.get_coords()

    if not coords or len(coords) < 6:
        print("Failed to read coordinates")
        return

    x, y, z, rx, ry, rz = coords

    R = rotation_matrix_from_rpy(rx, ry, rz)

    delta_tool = np.array([
        dx_tool,
        dy_tool,
        dz_tool
    ])

    # Tool frame -> Base frame
    delta_base = R @ delta_tool

    x += float(delta_base[0])
    y += float(delta_base[1])
    z += float(delta_base[2])

    target = [x, y, z, rx, ry, rz]

    print("\n--------------------------------")
    print("Current :", coords)
    print("Tool Δ  :", delta_tool)
    print("Base Δ  :", delta_base)
    print("Target  :", target)

    mc.send_coords(target, SPEED, 1)


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        return

    print("\n==========================================")
    print("END-EFFECTOR FRAME TELEOP")
    print("==========================================")
    print("↑  : +Tool Z")
    print("↓  : -Tool Z")
    print("←  : +Tool X")
    print("→  : -Tool X")
    print("PgUp   : +Tool Y")
    print("PgDn   : -Tool Y")
    print("O : Open Gripper")
    print("C : Close Gripper")
    print("H : Home")
    print("Q : Quit")
    print("==========================================\n")

    while True:

        ret, frame = cap.read()

        if ret:
            cv2.imshow("JetCobot Teleop", frame)

        key = cv2.waitKeyEx(1)

        if key == ord('q'):
            break

        elif key == ord('h'):
            print("Moving Home...")
            mc.send_angles([0, 0, 0, 0, 0, -45], 50)

        elif key == ord('o'):
            print("Open Gripper")
            mc.set_gripper_value(100, 50)

        elif key == ord('c'):
            print("Close Gripper")
            mc.set_gripper_value(20, 50)

        # ← = +Tool X
        elif key == LEFT_ARROW:
            move_tool(STEP, 0, 0)

        # → = -Tool X
        elif key == RIGHT_ARROW:
            move_tool(-STEP, 0, 0)

        # PgUp = +Tool Y
        elif key == PAGE_UP:
            move_tool(0, STEP, 0)

        # PgDn = -Tool Y
        elif key == PAGE_DOWN:
            move_tool(0, -STEP, 0)

        # ↑ = +Tool Z
        elif key == UP_ARROW:
            move_tool(0, 0, STEP)

        # ↓ = -Tool Z
        elif key == DOWN_ARROW:
            move_tool(0, 0, -STEP)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
