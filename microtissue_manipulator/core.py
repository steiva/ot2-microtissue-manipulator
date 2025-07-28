import os
from scipy.spatial import KDTree
import numpy as np 
import cv2
import paths
import datetime
import string
import pandas as pd
from dataclasses import dataclass, fields, asdict, MISSING
from typing import get_type_hints, get_origin, get_args, Any, Union

class Destination:
    WELL_PLATE_PRESETS = {
        6: (2, 3),   # 2 rows × 3 cols
        24: (4, 6),  # 4 rows × 6 cols
        48: (6, 8),  # 6 rows × 8 cols
        96: (8, 12),  # 8 rows × 12 cols
        384: (16, 24)  # 16 rows × 24 cols
    }

    def __init__(self, plate_type=None, custom_positions=None):
        """
        Defines a destination, which can be a standard well plate or custom locations.

        :param plate_type: Integer for a standard well plate (6, 24, 48, 96, 384).
        :param custom_positions: List of arbitrary locations if not using a well plate.
        """
        self.plate_type = plate_type
        self.layout = self.WELL_PLATE_PRESETS.get(plate_type, None)
        self.custom_positions = custom_positions
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Generates well names based on plate type or uses custom positions."""
        if self.custom_positions:
            return self.custom_positions  # Use provided custom locations
        
        if not self.layout:
            raise ValueError("Invalid well plate type or missing custom positions.")

        rows, cols = self.layout
        row_labels = string.ascii_uppercase[:rows]  # First N letters for rows
        return [f"{row}{col}" for row in row_labels for col in range(1, cols + 1)]

    def get_well_index(self, well_label):
        """Returns the index of a well label like 'A1'."""
        if well_label in self.positions:
            return self.positions.index(well_label)
        return None

    def __repr__(self):
        return f"Destination(plate_type={self.plate_type}, positions={self.positions})"


class Routine:
    def __init__(self, destination, well_plan, fill_strategy="well_by_well"):
        """
        Routine class for controlling how a well plate or location is filled.

        :param destination: Destination object defining well plate/grid.
        :param well_plan: Dictionary {well_label: target_count} defining objects per well.
        :param fill_strategy: How the wells should be filled.
                              Options: "vertical", "horizontal", "well_by_well", "spread_out"
        """
        self.destination = destination
        self.well_plan = well_plan  # {well_label: target_count}
        self.fill_strategy = fill_strategy
        self.filled_wells = {k: 0 for k in well_plan}
        self.miss_counts = {k: 0 for k in well_plan}
        self.completed = False
        self.current_well = None

    def get_fill_order(self):
        """Returns the order in which wells should be filled based on strategy."""
        wells = list(self.well_plan.keys())

        if self.fill_strategy == "vertical":
            return sorted(wells, key=lambda well: int(well[1:]))  # Sort by column number
        elif self.fill_strategy == "horizontal":
            return sorted(wells, key=lambda well: well[0])  # Sort by row letter
        elif self.fill_strategy == "spread_out":
            return sorted(wells, key=lambda well: self.well_plan[well])  # Spread out based on needs
        else:  # Default: well_by_well
            return wells

    def get_next_well(self):
        """Returns the next well to be filled based on the strategy."""
        for well in self.get_fill_order():
            if self.filled_wells[well] < self.well_plan[well]:
                self.current_well = well
                return well
        self.completed = True
        return None

    def update_well(self, success=True):
        """Updates well status after an attempt."""
        if self.current_well is not None:
            if success:
                self.filled_wells[self.current_well] += 1
            else:
                self.miss_counts[self.current_well] += 1

    def is_done(self):
        """Checks if routine is completed."""
        return self.completed

def create_well_plan(plate_type):
    """Creates an empty DataFrame for well input based on the plate size."""
    rows, cols = Destination.WELL_PLATE_PRESETS[plate_type]
    row_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:rows])
    col_labels = list(range(1, cols + 1))

    well_df = pd.DataFrame(np.zeros((rows, cols), dtype=int), index=row_labels, columns=col_labels)
    return well_df

def is_instance_of_type(value: Any, expected_type: Any) -> bool:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        return isinstance(value, expected_type)

    if origin is Union:
        return any(is_instance_of_type(value, arg) for arg in args)

    if origin is tuple:
        if len(args) == 2 and args[1] is ...:  # Tuple[int, ...]
            return isinstance(value, tuple) and all(is_instance_of_type(v, args[0]) for v in value)
        return (
            isinstance(value, tuple)
            and len(value) == len(args)
            and all(is_instance_of_type(v, t) for v, t in zip(value, args))
        )

    if origin is list:
        return isinstance(value, list) and all(is_instance_of_type(v, args[0]) for v in value)

    if origin is dict:
        return (
            isinstance(value, dict)
            and all(is_instance_of_type(k, args[0]) and is_instance_of_type(v, args[1]) for k, v in value.items())
        )

    return isinstance(value, expected_type)

class MarkdownLogger:
    def __init__(self, log_dir=paths.LOGS_DIR, experiment_name=None, settings: dict = None, well_plate: pd.DataFrame = None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = f"experiment_{timestamp}"
        self.log_file = os.path.join(log_dir, f"{experiment_name}_log_{timestamp}.md")
        self._start_log(experiment_name, settings, well_plate)

    def _start_log(self, experiment_name, settings, well_plate):
        with open(self.log_file, 'w') as f:
            f.write(f"# Log for {experiment_name}\n")
            f.write(f"_Started on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            
            if settings:
                f.write("## Settings\n\n")
                f.write("| Key | Value |\n")
                f.write("| --- | ----- |\n")
                for key, value in settings.items():
                    f.write(f"| `{key}` | `{value}` |\n")
                f.write("\n")

            if well_plate is not None:
                f.write("## Well Plate Plan\n\n")
                f.write(well_plate.to_markdown(index=True))
                f.write("\n\n")

    def log_table(self, df: pd.DataFrame, title: str = "Table"):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"- **[{timestamp}]** {title}\n\n")
            f.write(df.to_markdown(index=False) + '\n\n')

    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"- **[{timestamp}]** {message}\n")

    def log_section(self, title):
        with open(self.log_file, 'a') as f:
            f.write(f"\n## {title}\n\n")

@dataclass
class PickingConfig:
    vol: float = 10.0
    dish_bottom: float = 66.1 #10.60 for 300ul, 9.5 for 200ul
    pickup_offset: float = 0.5
    pickup_height: float = dish_bottom + pickup_offset
    flow_rate: float = 50.0
    cuboid_size_theshold: tuple[int, int] = (250, 500)
    failure_threshold: float = 0.5
    minimum_distance: float = 1.7
    wait_time_after_deposit: float = 0.5
    one_by_one: bool = False

    # ----------------------Deposit configs-----------------------
    well_offset_x: float = -0.3 #384 well plate
    well_offset_y: float = -0.9 #384 well plate
    deposit_offset_z: float = 0.5
    destination_slot: int = 5

    # ----------------------Video configs-----------------------
    circle_center: tuple[int, int] = (1296, 972)
    circle_radius: int = 900
    contour_filter_window: tuple[int, int] = (30, 1000)  # min and max area for contour filtering
    aspect_ratio_window: tuple[float, float] = (0.75, 1.25)  # min and max aspect ratio for contour filtering
    circularity_window: tuple[float, float] = (0.6, 0.9)  # circularity range for contour filtering

    @classmethod
    def from_dict(cls, data: dict) -> "PickingConfig":
        type_hints = get_type_hints(cls)
        init_args = {}

        for f in fields(cls):
            name = f.name
            if name in data:
                value = data[name]
            elif f.default is not MISSING:
                value = f.default
            else:
                raise ValueError(f"Missing required field: {name}")

            expected_type = type_hints[name]
            if not is_instance_of_type(value, expected_type):
                raise TypeError(f"Field '{name}' is expected to be {expected_type}, got {type(value)}")

            init_args[name] = value

        # Recalculate pickup_height after loading
        obj = cls(**init_args)
        obj.pickup_height = obj.dish_bottom + obj.pickup_offset
        return obj

    def to_dict(self):
        return asdict(self)

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