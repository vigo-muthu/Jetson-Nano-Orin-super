import cv2
import numpy as np
import time
from pymycobot import MyCobot

# 1. Initialize physical connection to the robot arm
# Note: Double check if your specific OS uses '/dev/ttyUSB0' or '/dev/ttyTHS1'
PORT = '/dev/ttyUSB0' 
BAUD = 115200
mc = MyCobot(PORT, BAUD)

# Power on the joints and initialize the gripper hardware
mc.power_on()
time.sleep(1)
mc.set_gripper_mode(0) 
mc.init_gripper()
mc.set_gripper_value(100, 50) # Fully open the gripper claw (100% open at 50% speed)

# Move the robot to a safe, predictable home configuration
HOME_ANGLES = [0, 0, 0, 0, 0, 0]
mc.send_angles(HOME_ANGLES, 50)
time.sleep(3)

# 2. Camera Sensor Initialization
cap = cv2.VideoCapture(0) # Grabs the integrated USB workspace camera

# Define target color thresholds in HSV space (Example: A bright yellow block)
lower_target = np.array([20, 100, 100])
upper_target = np.array([30, 255, 255])

# Spatial Calibration Metrics (Tweak these parameters to fit your desk setup)
CAMERA_X_OFFSET = 145.0  # Measured physical distance (mm) from robot base center to camera view center
PIXEL_TO_MM_RATIO = 0.65  # Scale factor: roughly how many physical mm does 1 pixel occupy?

print("System initialized. Scanning for target objects...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break
            
        # Isolate the targeted color profile
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_target, upper_target)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Focus entirely on the largest object matching that color
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 400: # Disregard minor pixel noise
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Isolate object's exact center pixel
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Visual feedback: Draw a tracking circle on the feed screen
                    cv2.circle(frame, (cX, cY), 8, (0, 255, 0), -1)
                    
                    # Convert pixel positions into absolute physical coordinates (mm)
                    # Assumes a baseline 640x480 video frame profile
                    target_x = CAMERA_X_OFFSET + ((240 - cY) * PIXEL_TO_MM_RATIO)
                    target_y = (320 - cX) * PIXEL_TO_MM_RATIO
                    target_z = 55.0  # Safe physical Z-height of your desk surface
                    
                    print(f"Target located! Calculated positions -> X: {target_x:.1f}mm, Y: {target_y:.1f}mm")
                    
                    # --- HARDWARE PICK & PLACE SEQUENCE ---
                    
                    # Approach: Hover 50mm squarely above the object to prevent accidental collisions
                    mc.send_coords([target_x, target_y, target_z + 50, -180, 0, 0], 40, 1)
                    time.sleep(2)
                    
                    # Descend: Drop directly down onto the item
                    mc.send_coords([target_x, target_y, target_z, -180, 0, 0], 30, 1)
                    time.sleep(1.5)
                    
                    # Grasp: Securely close the gripper around the target 
                    mc.set_gripper_value(15, 60) # 15% open (closed tight) at 60% speed
                    time.sleep(1.5)
                    
                    # Ascend: Lift the item clear of the table surface
                    mc.send_coords([target_x, target_y, target_z + 70, -180, 0, 0], 40, 1)
                    time.sleep(1.5)
                    
                    # Transfer: Travel to the designated physical drop location
                    dest_x, dest_y, dest_z = 130.0, -120.0, 75.0
                    mc.send_coords([dest_x, dest_y, dest_z, -180, 0, 0], 40, 1)
                    time.sleep(2.5)
                    
                    # Release: Drop the item into the designated box/zone
                    mc.set_gripper_value(100, 50)
                    time.sleep(1.5)
                    
                    # Reset: Return back to safety home configuration before scanning again
                    mc.send_angles(HOME_ANGLES, 40)
                    time.sleep(3)
                    print("Cycle completed successfully. Scanning for new targets...")
        
        # Display the live execution window
        cv2.imshow("JetCobot Real-Time Tracking Pipeline", frame)
        
        # Break execution loop immediately if 'q' is pressed on keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safely release camera hardware resource and tear down windows upon exit
    cap.release()
    cv2.destroyAllWindows()
