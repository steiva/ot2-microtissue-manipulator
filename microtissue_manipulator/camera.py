import cv2
import multiprocessing as mp
import numpy as np
import time
from typing import Optional, Tuple, Union
from pygrabber.dshow_graph import FilterGraph
import paths
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union
import queue

CAMERA_LABELS_FILE = paths.CAM_CONFIGS_DIR + "/camera_labels.json"
CAMERA_CONFIG_DIR = paths.CAM_CONFIGS_DIR

class CameraManagerWindows:

    def __init__(self):
        self.graph = FilterGraph()
        self.devices = self.graph.get_input_devices()
        self.device_to_index = {name: idx for idx, name in enumerate(self.devices)}
        os.makedirs(CAMERA_CONFIG_DIR, exist_ok=True)
        self.label_map = self.load_labels()

    def load_labels(self):
        if os.path.exists(CAMERA_LABELS_FILE):
            with open(CAMERA_LABELS_FILE, "r") as f:
                return json.load(f)
        else:
            with open(CAMERA_LABELS_FILE, "w") as f:
                json.dump({}, f, indent=4)
            return {}

    def save_labels(self):
        with open(CAMERA_LABELS_FILE, "w") as f:
            json.dump(self.label_map, f, indent=4)

    def assign_label(self, device_name, label):
        if device_name not in self.devices:
            raise ValueError(f"Device '{device_name}' not found.")
        self.label_map[label] = device_name
        self.save_labels()
        self.load_resolution_config(label)

    def list_devices(self):
        return self.devices

    def list_labels(self):
        return self.label_map

    def get_camera_index_by_label(self, label):
        device_name = self.label_map.get(label)
        if device_name is None:
            raise ValueError(f"Label '{label}' not found.")
        index = self.device_to_index.get(device_name)
        if index is None:
            raise RuntimeError(f"Device '{device_name}' is not connected.")
        return index

    def get_config_path(self, label):
        return os.path.join(CAMERA_CONFIG_DIR, f"{label}.json")

    def load_resolution_config(self, label):
        config_path = self.get_config_path(label)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            empty = {"resolutions": [], "default_resolution": None}
            with open(config_path, "w") as f:
                json.dump(empty, f, indent=4)
            return empty

    def save_resolution_config(self, label, resolutions):
        config_path = self.get_config_path(label)
        with open(config_path, "w") as f:
            json.dump({"resolutions": resolutions}, f, indent=4)

class MultiprocessVideoCapture:
    """
    A class that captures video frames in a separate process and allows the main process
    to retrieve the most recent frame without blocking.
    """
    
    def __init__(self, camera_id: Union[int, str], width: int = 640, height: int = 480, fps: int = 30, focus: int = None):
        """
        Initialize the MultiprocessVideoCapture.
        
        Args:
            camera_id: Camera index or path to video file
            width: Frame width
            height: Frame height
            fps: Desired frames per second
        """
        # Create a shared array for the frame data
        self.shape = (height, width, 3)
        self.frame_size = int(np.prod(self.shape))
        self.shared_array = mp.Array('B', self.frame_size)
        
        # Create shared value for frame ready flag and error status
        self.frame_ready = mp.Value('i', 0)
        self.has_error = mp.Value('i', 0)
        self.error_msg = mp.Array('c', 256)
        self.running = mp.Value('i', 1)
        
        # Create process for video capture
        self.process = mp.Process(
            target=self._capture_process, 
            args=(camera_id, width, height, fps, focus)
        )
        self.process.daemon = True
        self.process.start()
    
    def _capture_process(self, camera_id: Union[int, str], width: int, height: int, fps: int, focus: int) -> None:
        """
        The process function that captures frames from the camera.
        
        Args:
            camera_id: Camera index or path to video file
            width: Frame width
            height: Frame height
            fps: Desired frames per second
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                with self.has_error.get_lock():
                    self.has_error.value = 1
                    error_msg = f"Could not open camera {camera_id}"
                    self.error_msg.value = error_msg.encode()
                return
                
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if focus is not None:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
            
            # For timing to maintain fps
            frame_time = 1.0 / fps
            
            while self.running.value:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    with self.has_error.get_lock():
                        self.has_error.value = 1
                        error_msg = "Failed to grab frame from camera"
                        self.error_msg.value = error_msg.encode()
                    break
                
                # Ensure frame size matches expected dimensions
                if frame.shape != self.shape:
                    frame = cv2.resize(frame, (width, height))
                
                # Copy the frame data to shared memory
                frame_flat = frame.flatten()
                with self.shared_array.get_lock():
                    shared_array_np = np.frombuffer(self.shared_array.get_obj(), dtype='B')
                    np.copyto(shared_array_np, frame_flat)
                
                # Set frame as ready
                with self.frame_ready.get_lock():
                    self.frame_ready.value = 1
                
                # Sleep to maintain desired fps
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            # Handle any exceptions
            with self.has_error.get_lock():
                self.has_error.value = 1
                error_msg = f"Exception in capture process: {str(e)}"
                self.error_msg.value = error_msg[:255].encode()
        finally:
            # Clean up
            if 'cap' in locals() and cap.isOpened():
                cap.release()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the most recent frame from the video capture process.
        
        Returns:
            Tuple containing:
            - Boolean indicating if frame was retrieved successfully
            - The frame as a numpy array or None if no frame is available
        """
        # Check for errors
        if self.has_error.value:
            error_msg = self.error_msg.value.decode()
            print(f"Error in capture process: {error_msg}")
            return False, None
        
        # Check if frame is ready
        if not self.frame_ready.value:
            return False, None
        
        # Get the frame from shared memory
        with self.shared_array.get_lock():
            frame_flat = np.frombuffer(self.shared_array.get_obj(), dtype='B')
            frame = frame_flat.reshape(self.shape).copy()
        
        return True, frame
    
    def is_opened(self) -> bool:
        """
        Check if the video capture is open and running.
        
        Returns:
            Boolean indicating if the capture is open
        """
        return self.process.is_alive() and not self.has_error.value
    
    def release(self) -> None:
        """
        Release the video capture resources.
        """
        self.running.value = 0
        if self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()

def open_capture(label: str, cam_manager: CameraManagerWindows, resolution: Union[list[int, int], str] = 'default', focus: int = None) -> MultiprocessVideoCapture:
    index = cam_manager.get_camera_index_by_label(label)
    if resolution == 'default':
        resolution = cam_manager.load_resolution_config(label)['default_resolution']
    elif isinstance(resolution, list) and len(resolution) == 2:
        config = cam_manager.load_resolution_config(label)
        if resolution not in config['resolutions']:
            print(f"Warning: Resolution {resolution} not found in saved configurations for camera '{label}'. Using default resolution.")
            resolution = cam_manager.load_resolution_config(label)['default_resolution']
    return MultiprocessVideoCapture(index, width=resolution[0], height=resolution[1], focus=focus)

class frameOperations():
    def __init__(self, width, height):
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.new_camera_matrix = None
        self.w = width
        self.h = height

    def load_camera_intrinsics(self, config_profile: str, use_new_cam_mtx: bool = True) -> None:
        """
        Load camera intrinsics from a JSON configuration file.
        
        Args:
            config_profile: The name of the profile to load the camera intrinsics from.
            use_new_cam_mtx: Whether to compute a new camera matrix using cv2.getOptimalNewCameraMatrix.
        """
        config_path = os.path.join(paths.PROFILES_DIR, config_profile, 'camera_intrinsics.json')
        with open(config_path, 'r') as f:
            camera_data = json.load(f)

        if not camera_data:
            raise ValueError(f"No camera intrinsics found in {config_path}. Please calibrate the camera first.")
            return
        
        self.camera_matrix = np.array(camera_data['camera_mtx'])
        self.distortion_coefficients = np.array(camera_data['dist_coeffs'])
        
        if use_new_cam_mtx:
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients, (self.w, self.h), 1, (self.w, self.h))
        else:
            self.new_camera_matrix = None

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a frame using the loaded camera intrinsics.
        
        Args:
            frame: The input frame to undistort.
        
        Returns:
            The undistorted frame.
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsics not loaded. Please load camera intrinsics before undistorting frames.")
        
        if self.new_camera_matrix is not None:
            return cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients, newCameraMatrix=self.new_camera_matrix)
        else:
            return cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients)


@dataclass
class DisplayCommand:
    """Commands that can be sent to the display process"""
    command: str  # 'frame', 'annotation', 'status', 'quit'
    data: Any = None
    timestamp: float = None

class FrameDisplayProcess:
    """Separate process for handling OpenCV frame display"""
    
    def __init__(self, window_name="Robot Vision", window_size=(1348, 1011)):
        self.window_name = window_name
        self.window_size = window_size
        self.frame_queue = mp.Queue(maxsize=5)  # Limit queue size to prevent memory buildup
        self.command_queue = mp.Queue(maxsize=10)
        self.status_queue = mp.Queue(maxsize=5)
        self.process = None
        self.running = False
        
    def start(self):
        """Start the display process"""
        if self.process is not None and self.process.is_alive():
            print("Display process already running")
            return
            
        self.running = True
        self.process = mp.Process(target=self._display_worker, daemon=True)
        self.process.start()
        print(f"Display process started with PID: {self.process.pid}")
        
    def stop(self):
        """Stop the display process"""
        if self.process is None or not self.process.is_alive():
            return
            
        self.running = False
        try:
            self.command_queue.put(DisplayCommand("quit"), timeout=1)
            self.process.join(timeout=3)
            if self.process.is_alive():
                print("Force terminating display process...")
                self.process.terminate()
                self.process.join(timeout=1)
        except Exception as e:
            print(f"Error stopping display process: {e}")
        finally:
            self.process = None
            
    def send_frame(self, frame: np.ndarray, annotations: Dict[str, Any] = None):
        """Send a frame to be displayed (non-blocking)"""
        if not self.running or self.process is None or not self.process.is_alive():
            return False
            
        try:
            # Clear old frames to prevent lag
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                    
            frame_data = {
                'frame': frame.copy(),
                'annotations': annotations or {},
                'timestamp': time.time()
            }
            self.frame_queue.put(frame_data, timeout=0.001)  # Very short timeout
            return True
        except queue.Full:
            # Skip frame if queue is full (prevents blocking)
            return False
        except Exception as e:
            print(f"Error sending frame: {e}")
            return False
            
    def send_status(self, status_text: str, color=(0, 255, 0)):
        """Send status text to be displayed"""
        try:
            status_data = {
                'text': status_text,
                'color': color,
                'timestamp': time.time()
            }
            self.status_queue.put(status_data, timeout=0.001)
        except queue.Full:
            pass  # Skip if queue is full
        except Exception as e:
            print(f"Error sending status: {e}")
            
    def _display_worker(self):
        """Worker function that runs in the separate process"""
        try:
            # Initialize OpenCV in the separate process
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, *self.window_size)
            
            current_frame = None
            status_text = ""
            status_color = (0, 255, 0)
            last_frame_time = time.time()
            
            print(f"Display worker started in process {mp.current_process().pid}")
            
            while True:
                try:
                    # Check for quit command
                    try:
                        cmd = self.command_queue.get_nowait()
                        if cmd.command == "quit":
                            print("Display worker received quit command")
                            break
                    except queue.Empty:
                        pass
                    
                    # Get latest frame (non-blocking)
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        current_frame = frame_data['frame']
                        last_frame_time = time.time()
                    except queue.Empty:
                        pass
                    
                    # Get latest status (non-blocking)
                    try:
                        status_data = self.status_queue.get_nowait()
                        status_text = status_data['text']
                        status_color = status_data['color']
                    except queue.Empty:
                        pass
                    
                    # Display frame if available
                    if current_frame is not None:
                        display_frame = current_frame.copy()
                        
                        # Add status text overlay
                        if status_text:
                            cv2.putText(display_frame, status_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                        
                        # Add timestamp if frame is old
                        if time.time() - last_frame_time > 1.0:
                            cv2.putText(display_frame, "NO NEW FRAMES", (10, 70), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        cv2.imshow(self.window_name, display_frame)
                    
                    # Handle OpenCV events
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        print("ESC pressed in display window")
                        break
                        
                    # Small delay to prevent high CPU usage
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error in display worker: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Fatal error in display worker: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Display worker shutting down")