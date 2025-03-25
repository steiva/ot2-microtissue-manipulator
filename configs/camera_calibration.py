import sys
import os

upstream_dir = os.path.abspath(os.path.join(os.getcwd()))
if upstream_dir not in sys.path:
    sys.path.insert(0, upstream_dir)

import cv2
import numpy as np
from microtissue_manipulator import utils
from configs import paths
import json

# Define the dimensions of the ChArUco board
# Create the ChArUco board

squares_x=7
squares_y=5 
square_length=0.0154
marker_length=0.0077
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# Create the detector parameters
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Initialize lists to store the corners and ids from all images
all_corners = []
all_ids = []
image_size = None

# Capture images from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cv2.namedWindow('Charuco Board',  cv2.WINDOW_NORMAL)
cv2.resizeWindow('Charuco Board', 1348, 1011)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Interpolate the Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

        if retval > 0:
            # Draw the Charuco board
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)


            if image_size is None:
                image_size = gray.shape[::-1]

    # Display the image
    cv2.imshow('Charuco Board', frame)

    k = cv2.waitKey(1) & 0xFF
    # Break the loop on 'q' key press
    if k == ord('q'):
        break

    elif k == ord('s'):
        if charuco_corners is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

cap.release()
cv2.destroyAllWindows()

print("Captured images:", len(all_corners))

# Select each tenth of the all_corners list
selected_corners = all_corners
selected_ids = all_ids

for i, corners in enumerate(selected_corners):
    if len(corners) < 4:
        selected_corners.pop(i)
        selected_ids.pop(i)

print("Selected corners:", len(selected_corners))

# Calibrate the camera using the collected corners and ids
profile_name = 'bottomDeck'
if len(all_corners) > 0:
    print('Calibrating camera ...')
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(selected_corners, selected_ids, charuco_board, image_size, None, None)

    if ret:
        utils.check_camera_config(profile_name)
        config_path = os.path.join(paths.PROFILES_DIR, profile_name, 'camera_intrinsics.json')
        with open(config_path, 'r') as json_file:
            camera_data = json.load(json_file)

        camera_data['camera_mtx'] = camera_matrix.tolist()
        camera_data['dist_coeffs'] = dist_coeffs.tolist()

        with open(config_path, 'w') as json_file:
            json.dump(camera_data, json_file, indent=4)

        print("Calibration successful ...")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)
    else:
        print("Calibration failed")
else:
    print("No corners were detected")