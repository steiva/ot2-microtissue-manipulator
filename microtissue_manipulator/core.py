import cv2
import threading
import numpy as np
import paths
import json
import os
from scipy.spatial import KDTree
import pandas as pd

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
    def __init__(self, index = 0, config_profile = 'standardDeck', no_buffer = True, use_new_cam_mtx = False) -> None:
        self.w, self.h = 2592, 1944
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

        config_path = os.path.join(paths.PROFILES_DIR, config_profile, 'camera_intrinsics.json')
        with open(config_path, 'r') as f:
            camera_data = json.load(f)

        self.camera_matrix = np.array(camera_data['camera_mtx'])
        self.distortion_coefficients = np.array(camera_data['dist_coeffs'])
        self.no_buffer = no_buffer
        
        if use_new_cam_mtx:
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients, (self.w, self.h), 1, (self.w, self.h))
        else:
            self.new_camera_matrix = None

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
                # self.get_frame()
                print('Frame not captured ...')
                return
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if undist:
                if self.new_camera_matrix is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients, None, self.new_camera_matrix)
                else:
                    frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients)
            return frame
        # else:
        #     self.get_frame()

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


class Core():
    def __init__(self) -> None:
        self.cuboids = None
        self.cuboid_df = None
        self.selected = []
        self.best_circ = None
        self.pickup_offset = 30
        self.initial_offset = 40
        self.locked = False

    def preprocess_frame(self, frame: np.ndarray) -> None:
        self.gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bil_fr = cv2.bilateralFilter(self.gray_fr, 5, 175, 175)
        self.thresh_fr = cv2.adaptiveThreshold(self.gray_fr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,29,5)

    def find_contours(self, frame: np.ndarray, offset: int = 0) -> None:
        """
        General contour finding pipeline of objects inside a circular contour. In our case
        we are looking for objects in a Petri dish.

        Args:
            frame (np.ndarray): frame taken from camera cap, or just an image.
            offset (int, optional): offset for radius of the circle where cuboids
            are detected. Offset is counted inwards. Defaults to 0.
        """
        self.preprocess_frame(frame)
        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = cv2.bilateralFilter(frame, 5, 175, 175)
        thresh = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,29,5) 
        kernel = np.ones((3,3),np.uint8)
        res = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find all the contours in the resulting image.
        contours, hierarchy = cv2.findContours(
            res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.cuboids = contours

    def get_circles(self, frame: np.ndarray) -> None:
        """
        Function looks for a petri dish in the frame, and assigns the smallest one
        to a class variable for storage. 

        Args:
            frame (np.ndarray): frame in which to detect the petri dish.
        """
        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        blur = cv2.GaussianBlur(frame,(3,3),0)
        # if not self.inv:
        #     ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # else:
        ret, thresh = cv2.threshold(blur,125,255,cv2.THRESH_BINARY_INV)

        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(thresh,kernel,iterations = 3)

        blur2 = cv2.blur(dilation, (7, 7))
        detected_circles = cv2.HoughCircles(image=blur2,
                                            method=cv2.HOUGH_GRADIENT,
                                            dp=1.2,
                                            minDist=500,
                                            param1=100,
                                            param2=50,
                                            minRadius=700,
                                            maxRadius=900
                                            )

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

            pt = detected_circles[0][0]
            a, b, r = pt
            
            if self.best_circ is None or r < self.best_circ[2]:
                self.best_circ = pt

            best_center = np.array(self.best_circ[:2])
            curr_center = np.array([a, b])
            if np.sqrt(np.sum((best_center - curr_center)**2)) > 30:
                self.best_circ = pt

    # def size_conversion(self, cuboid_size_px: float) -> float:
    #     """Size conversion from pixels to microns and then to 
    #     cuboid diameter. In this calculation, cuboids are assumed to 
    #     look circular. If we regard the cuboids as squares (top down view),
    #     just a square root of cuboid_size_micron2 should be sufficient.

    #     Args:
    #         cuboid_size_px (float): cuboid size in pixels as seen by the camera.

    #     Returns:
    #         float: cuboid diameter in microns.
    #     """    
    #     cuboid_size_micron2 = cuboid_size_px * self.size_conversion_ratio * 1000000
    #     cuboid_diameter = 2 * np.sqrt(cuboid_size_micron2 / np.pi)
    #     return cuboid_diameter

    def contour_aspect_ratio(self, contour: np.ndarray) -> float:
        """
        Function calculates the aspect ratio of a contour.

        Args:
            contour (np.ndarray): A contour for which the aspect ratio is to be calculated.

        Returns:
            float: The aspect ratio of the contour.
        """
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return aspect_ratio
    
    def contour_circularity(self, contour: np.ndarray) -> float:
        """
        Function calculates the circularity of a contour.

        Args:
            contour (np.ndarray): A contour for which the circularity is to be calculated.

        Returns:
            float: The circularity of the contour.
        """
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        return circularity

    def cuboid_dataframe(self, contours: list, filter_thresh: int = None) -> None:
        """
        Function creates dataframe with all necessary information about the cuboids:
        The area of the individual cuboids, the coordinates of their center, distance
        to closest neighbor, and a boolean status if it is pickable or not, based on 
        wether it is located in a pickable region.

        Args:
            contours (list): a list of detected contours.
            filter_thresh (int): filter out the contours based on size. For example,
            if filter_thresh = 10, all contours that are smaller than 10 are filtered out.
        """        
        df_columns = ['contour', 'area', 'cX', 'cY', 'min_dist', 'aspect_ratio', 'circularity']

        if not contours:
            self.cuboid_df = pd.DataFrame(columns=df_columns)
            return
        
        cuboid_df = pd.DataFrame({'contour':contours})
        cuboid_df['area'] = cuboid_df.apply(lambda row : cv2.contourArea(row.values[0]), axis=1)
        if filter_thresh:
            cuboid_df = cuboid_df.loc[cuboid_df.area > filter_thresh]
            if len(cuboid_df) == 0:
                self.cuboid_df = pd.DataFrame(columns=df_columns)
                return
            
        centers = cuboid_df.apply(lambda row : self.contour_centers([row.values[0]])[0], axis=1)
        cuboid_df[['cX','cY']] = pd.DataFrame(centers.tolist(),index=cuboid_df.index)
        cuboid_df.dropna(inplace=True)

        T = KDTree(cuboid_df[['cX', 'cY']].to_numpy())
        cuboid_df['min_dist'] = cuboid_df.apply(lambda row: T.query((row.values[2], row.values[3]), k = 2)[0][-1], axis = 1)
        cuboid_df['aspect_ratio'] = cuboid_df.apply(lambda row: self.contour_aspect_ratio(row.values[0]), axis=1)
        cuboid_df['circularity'] = cuboid_df.apply(lambda row: self.contour_circularity(row.values[0]), axis=1)
        self.cuboid_df = cuboid_df

    def contour_centers(self, contours: tuple) -> list:
        """
        Function calculates the centers of the inputed contours.

        Args:
            contours (tuple): A tuple of contours to be filtered, normally outputed 
            by cv2.findContours() function.

        Returns:
            list: outputs list of coordinates of the contour centers.
        """
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
            else:
                centers.append((None,None))
        return centers
    

def mask_frame(frame: np.ndarray, pt: tuple, offset: int) -> np.ndarray:
    """Function creates a circular mask and applies it to an image. In our case this is
    used to select the area in the petri dish only and find contours there.

    Args:
        frame (np.ndarray): frame that needs to be masked.
        pt (tuple): circle parameters, center coordinates a,b and radius r.
        offset (int): an offset for mask application. Useful if circle is too large.

    Returns:
        np.ndarray: returns a masked image.
    """
    a, b, r = pt
    # Create mask to isolate the information in the petri dish.
    mask = np.zeros_like(frame)
    mask = cv2.circle(mask, (a, b), r-offset, (255, 255, 255), -1)
    # Apply the mask to the image.
    result = cv2.bitwise_and(frame.astype('uint8'), mask.astype('uint8'))
    return result