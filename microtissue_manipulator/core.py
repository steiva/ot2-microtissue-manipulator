import cv2
import threading
import numpy as np
from configs import paths
import json

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                self.last_frame = frame

    def read(self):
        if self.last_frame is not None:
            return True, self.last_frame
        else: 
            return False, None
        
    def release(self):
        self.camera.release()

class Camera():
    def __init__(self, index = 0, no_buffer = True):
        self.cap = cv2.VideoCapture(index)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

        with open(paths.CAMERA_INTRINSICS_PATH, 'r') as f:
            camera_data = json.load(f)

        self.camera_matrix = np.array(camera_data['camera_mtx'])
        self.distortion_coefficients = np.array(camera_data['dist_coeffs'])
        self.no_buffer = no_buffer

        if self.no_buffer:
            print('Using camera without buffer ...')
            self.cap = CameraBufferCleanerThread(self.cap)
        print('Camera initialized ...')
        self.window_name = 'frame'

    def get_frame(self, undist: bool = False, gray: bool = False) -> np.ndarray:
        """
        Function for reading frames from capture with direct undistortion
        and settings for getting grayscale images directly.

        Args:
            undist (bool, optional): Boolean wether to undistort the image or not.
            Defaults to True.
            gray (bool, optional): Boolean wether to give grayscale images directly.
            Defaults to True.

        Returns:
            np.ndarray: A captured frame represented by a numpy array.
        """
        if self.cap.read() is not None:         
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_frame()
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if undist:
                frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients)
            return frame
        else:
            self.get_frame()

    def get_window(self) -> None:
        """
        Function for opening a window that fits the screen properly
        for 1920x1080 resolution monitor.
        """        
        cv2.namedWindow(self.window_name,  cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1348, 1011)

    def release_camera(self) -> None:
        """
        Release camera capture.
        """        
        print('Releasing capture ...')
        self.cap.release()