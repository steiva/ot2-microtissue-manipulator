import keyboard
import paths
import os
import json
import numpy as np

class ManualRobotMovement:
    def __init__(self, openapi, block_thread = False):
        self.openapi = openapi
        self.positions = []
        self.possible_steps = [0.01, 0.1, 0.5, 1, 3, 5, 10, 30, 50]
        self.step = 1

        keyboard.add_hotkey('up', self.move_forward)
        keyboard.add_hotkey('down', self.move_backward)
        keyboard.add_hotkey('left', self.move_left)
        keyboard.add_hotkey('right', self.move_right)
        keyboard.add_hotkey('pagedown', self.move_z_down)
        keyboard.add_hotkey('pageup', self.move_z_up)
        keyboard.add_hotkey('+', self.increase_step)
        keyboard.add_hotkey('-', self.decrease_step)
        keyboard.add_hotkey('s', self.save_position)
        # Keep the program running until 'q' is pressed
        if block_thread:
            keyboard.wait('q')
            keyboard.unhook_all()

    def position_safeguard(self, position):
        x, y, z = position
        can_move = False
        x_condition = x >=0 and x <= 380
        y_condition = y >=0 and y <= 350
        z_condition = z >=0.1 and z <= 205

        if x_condition and y_condition and z_condition:
            can_move = True
        return can_move

    def move_z_down(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x, y, z-self.step)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('z', -self.step)

    def move_z_up(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x, y, z+self.step)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('z', self.step)

    def move_forward(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x, y+self.step, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('y', self.step)

    def move_backward(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x, y-self.step, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('y', -self.step)

    def move_left(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x-self.step, y, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('x', -self.step)

    def move_right(self):
        x,y,z = self.openapi.get_position(verbose = False)[0].values()
        potential_position = (x+self.step, y, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('x', self.step)

    def increase_step(self):
        current_index = self.possible_steps.index(self.step)
        if current_index == len(self.possible_steps) - 1:
            return
        else:
            self.step = self.possible_steps[current_index + 1]

    def decrease_step(self):
        current_index = self.possible_steps.index(self.step)
        if current_index == 0:
            return
        else:
            self.step = self.possible_steps[current_index - 1]

    def save_position(self):
        position = self.openapi.get_position(verbose = False)[0]
        self.positions.append((position['x'], position['y'], position['z']))
        print(f"Saved position: {position}")

def create_configuration_profile(profile_name):
    profile_path = os.path.join(paths.PROFILES_DIR, profile_name)
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        check_calibration_config(profile_name)
        check_camera_config(profile_name)
        print(f"Profile '{profile_name}' created at {profile_path}.")
    else:
        print(f"Profile '{profile_name}' already exists at {profile_path}.")

def check_calibration_config(profile_name):
    config_path = os.path.join(paths.PROFILES_DIR, profile_name, 'calibration.json')
    if os.path.exists(config_path):
        print("calibration.json file exists.")
    else:
        print("calibration.json file does not exist.")
        data = {
            "robot_coords": [],  # List of robot coordinates
            "camera_coords": [],  # List of camera coordinates
            "tf_mtx": [],  # Transformation matrix
            "calib_origin": [],  # Calibration origin
            "offset": [],
            "size_conversion_ratio": 1.0,
            "one_d_ratio": 1.0
        }
        with open(config_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("calibration.json file created.")

def load_calibration_config(profile_name):
    config_path = os.path.join(paths.PROFILES_DIR, profile_name, 'calibration.json')
    with open(config_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_calibration_config(profile_name, data):
    config_path = os.path.join(paths.PROFILES_DIR, profile_name, 'calibration.json')
    with open(config_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def check_camera_config(profile_name):
    config_path = os.path.join(paths.PROFILES_DIR, profile_name, 'camera_intrinsics.json')
    if os.path.exists(config_path):
        print("camera_intrinsics.json file exists.")
    else:
        print("camera_intrinsics.json file does not exist.")
        data = {
            "camera_mtx": [],
            "dist_coeffs": []
        }
        with open(config_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("camera_intrinsics.json file created.")

def sort_coordinates(coords, reverse_y = False):
    """
    Sorts a list of 2D coordinates by their y-coordinate, divides them into two groups based on the y-coordinate,
    and sorts each group by the x-coordinate.

    Args:
        coords (list): List of tuples representing 2D coordinates.

    Returns:
        list: Sorted list of coordinates.
    """
    
    sorted_coords = sorted(coords, key=lambda coord: coord[1])
    mid_y = (sorted_coords[0][1] + sorted_coords[-1][1]) / 2
    if reverse_y:
        group1 = [coord for coord in sorted_coords if coord[1] > mid_y]
        group2 = [coord for coord in sorted_coords if coord[1] <= mid_y]
    else:
        group1 = [coord for coord in sorted_coords if coord[1] <= mid_y]
        group2 = [coord for coord in sorted_coords if coord[1] > mid_y]
    group1_sorted = sorted(group1, key=lambda coord: coord[0])
    group2_sorted = sorted(group2, key=lambda coord: coord[0])
    return group1_sorted + group2_sorted


def compute_tf_mtx(mm2pix_dict: dict) -> np.ndarray:
    """Function computes the transformation matrix between real-world
    coordinates and pixel coordinates in an image.

    Args:
        mm2pix_dict (dict): Dictionary mapping real-world coordinates
        to pixel coordinates. Example for four points:
        {(382.76, -113.37): (499, 412),
        (225.27, 94.68): (240, 103),
        (386.5, 91.55): (492, 98),
        (221.25, -110.62): (248, 419)}

    Returns:
        np.ndarray: array that represents the transformation matrix.
    """
    A = np.zeros((2 * len(mm2pix_dict), 6), dtype=float)
    b = np.zeros((2 * len(mm2pix_dict), 1), dtype=float)
    index = 0
    for XY, xy in mm2pix_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    tf_mtx = np.zeros((3, 3))
    tf_mtx[0, :] = np.squeeze(x[:3])
    tf_mtx[1, :] = np.squeeze(x[3:])
    tf_mtx[-1, -1] = 1
    return tf_mtx