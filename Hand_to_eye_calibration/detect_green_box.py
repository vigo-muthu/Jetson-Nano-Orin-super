import os
import cv2
import numpy as np
import pyrealsense2 as rs

SAVE_DIR = "./calib_data"

# 1. Load the Calibration Matrix
try:
    T_cam2base = np.load(os.path.join(SAVE_DIR, "T_cam2base.npy"))
except FileNotFoundError:
    print("Error: T_cam2base.npy not found! Run Step 2 first.")
    exit()

def get_robot_coordinates(x_cam, y_cam, z_cam):
    """Transforms Camera Frame (m) to Robot Base Frame (mm)"""
    point_cam = np.array([[x_cam], [y_cam], [z_cam], [1.0]])
    point_base = T_cam2base @ point_cam
    return [point_base[0, 0] * 1000.0, point_base[1, 0] * 1000.0, point_base[2, 0] * 1000.0]

# 2. Initialize RealSense Pipeline (Color + Depth)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

profile = pipeline.start(config)
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

print("Looking for green objects... Press 'q' to quit.")

frame_counter = 0  # Added to prevent terminal flooding

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
            
        img = np.asanyarray(color_frame.get_data())
        
        # 3. Find the Green Box using HSV color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Green HSV bounds in OpenCV (Hue is 0-179)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Create a single mask for green
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find the contours of the green object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest green object
            c = max(contours, key=cv2.contourArea)
            
            # Only proceed if the object is reasonably large
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 4. Get the 3D depth of that center pixel
                    depth = depth_frame.get_distance(cx, cy)
                    
                    # If depth is valid (not 0)
                    if depth > 0:
                        # Convert 2D pixel + Depth into 3D Camera Coordinate (Meters)
                        camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                        
                        # 5. Transform to Robot Coordinates (Millimeters)
                        robot_coords = get_robot_coordinates(camera_point[0], camera_point[1], camera_point[2])
                        
                        # --- Draw visual feedback ---
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
                        
                        cam_text = f"Cam (m): X:{camera_point[0]:.2f} Y:{camera_point[1]:.2f} Z:{camera_point[2]:.2f}"
                        rob_text = f"Rob(mm): X:{robot_coords[0]:.0f} Y:{robot_coords[1]:.0f} Z:{robot_coords[2]:.0f}"
                        
                        cv2.putText(img, cam_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(img, rob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # --- TERMINAL OUTPUT ---
                        # Print to the terminal every 15 frames (~twice a second)
                        frame_counter += 1
                        if frame_counter % 15 == 0:
                            print(f"TARGET DETECTED | {cam_text}  |  {rob_text}")
        
        cv2.imshow("Green Object Tracking", img)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
