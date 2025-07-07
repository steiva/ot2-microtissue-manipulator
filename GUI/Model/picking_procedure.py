# Imports
import sys
import os

# Add the upstream directory to sys.path
# upstream_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# if upstream_dir not in sys.path:
#     sys.path.insert(0, upstream_dir)

# Now you can import the module
from opentrons_api import ot2_api
from microtissue_manipulator import core, utils
import numpy as np 
import cv2
import time
import threading
import queue
import string
import pandas as pd
from dataclasses import dataclass, asdict, fields
from typeguard import typechecked

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

@typechecked
def test_func(x: int):
    return x

@typechecked
@dataclass
class PickingConfig:
    vol: float
    dish_bottom: float = 10.3 #10.60 for 300ul, 9.5 for 200ul
    pickup_offset: float = 0.5
    pickup_height: float = dish_bottom + pickup_offset
    flow_rate: float = 50.0
    cuboid_size_theshold: tuple[int] = (250, 500)
    failure_threshold: float = 0.5
    minimum_distance: float = 1.7
    wait_time_after_deposit: float = 0.5

    # ----------------------Deposit configs-----------------------
    well_offset_x: float = -0.3 #384 well plate
    well_offset_y: float = -0.9 #384 well plate

    # ----------------------Video configs-----------------------
    circle_center: tuple[int] = (1296, 972)
    circle_radius: int = 900
    contour_filter_window: tuple[int] = (30, 1000)  # min and max area for contour filtering
    aspect_ratio_window: tuple[float] = (0.75, 1.25)  # min and max aspect ratio for contour filtering
    min_circularity: float = 0.6  # min circularity for contour filtering

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**{k: v for k, v in cfg.items() if k in cls.__annotations__})

    def to_dict(self):
        return asdict(self)

class SharedSettings:
    def __init__(self, routine: Routine):
        self.lock = threading.Lock()
        self.cuboid_chosen = False
        self.local_timer_set = False
        self.routine = routine
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()

class PickingProcedure():
    def __init__(self, shared_settings: SharedSettings, 
                       picking_config: PickingConfig,
                       calibration_profile: str, 
                       cap: core.Camera, 
                       openapi: ot2_api.OpentronsAPI):
        
        self.coord_queue = queue.Queue()
        self.cr = core.Core()
        self.shared_settings_inst = shared_settings
        self.config = picking_config
        self.cap = cap
        self.openapi = openapi

        # ----------------------Robot configs-----------------------
        self.calibration_data = utils.load_calibration_config(calibration_profile)
        self.tf_mtx = np.array(self.calibration_data['tf_mtx'])
        self.calib_origin = np.array(self.calibration_data['calib_origin'])[:2]
        self.offset = np.array(self.calibration_data['offset'])
        self.size_conversion_ratio = self.calibration_data['size_conversion_ratio']
        self.one_d_ratio = self.calibration_data['one_d_ratio']

        self.isolated = []
        self.pickable_cuboids = []

    def cv_pipeline(self, frame):
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask, self.config.circle_center, self.config.circle_radius + int(self.config.minimum_distance / self.one_d_ratio), (255, 255, 255), -1)
        masked_frame = cv2.bitwise_and(frame, mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,2) 
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_inv = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.bitwise_and(thresh, mask_inv)

        # Find contours in the masked frame
        contours, hei = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if self.config.contour_filter_window[0] < cv2.contourArea(contour) < self.config.contour_filter_window[1]]
        self.cr.cuboids = filtered_contours
        self.cr.cuboid_dataframe(self.cr.cuboids)

        cuboid_size_micron2 = self.cr.cuboid_df.area * self.size_conversion_ratio * 10e6
        cuboid_diameter = 2 * np.sqrt(cuboid_size_micron2 / np.pi)
        dist_mm = self.cr.cuboid_df.min_dist * self.one_d_ratio
        self.cr.cuboid_df['diameter_microns'] = cuboid_diameter
        self.cr.cuboid_df['min_dist_mm'] = dist_mm
        
        # Filter out elongated contours
        self.pickable_cuboids = self.cr.cuboid_df.loc[(self.config.cuboid_size_theshold[0] < self.cr.cuboid_df['diameter_microns']) & 
                                            (self.cr.cuboid_df['diameter_microns'] < self.config.cuboid_size_theshold[1]) &
                                            ((self.cr.cuboid_df['aspect_ratio'] > self.config.aspect_ratio_window[0]) | 
                                             (self.cr.cuboid_df['aspect_ratio'] < self.config.aspect_ratio_window[1])) &
                                            (self.cr.cuboid_df['circularity'] > self.config.min_circularity)].copy()

        # Check if cuboid centers are within the circle radius from the current circle center
        self.pickable_cuboids['distance_to_center'] = self.pickable_cuboids.apply(
            lambda row: np.sqrt((row['cX'] - self.config.circle_center[0])**2 + (row['cY'] - self.config.circle_center[1])**2), axis=1
        )
        self.pickable_cuboids = self.pickable_cuboids[self.pickable_cuboids['distance_to_center'] <= self.config.circle_radius]
        self.isolated = self.pickable_cuboids.loc[self.pickable_cuboids.min_dist_mm > self.config.minimum_distance]

    def draw_annotations(self, frame, coords_tuple):
        cv2.circle(frame, self.config.circle_center, self.config.circle_radius + int(self.config.minimum_distance / self.one_d_ratio), (0, 0, 255), 2)
        cv2.circle(frame, self.config.circle_center, self.config.circle_radius, (0, 255, 0), 2)
        x,y,z = coords_tuple
        cv2.putText(frame, f"Robot coords: ({x:.2f}, {y:.2f}, {z:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.routine.current_well is None:
            self.routine.get_next_well()
        cv2.putText(frame, f"Filling well: {self.routine.current_well}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.shared_settings_inst.pause_event.is_set():
            cv2.putText(frame, "Paused", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with self.shared_settings_inst.lock:
            cuboid_chosen = self.shared_settings_inst.cuboid_chosen
        if not cuboid_chosen:
            cv2.drawContours(frame, self.cr.cuboids, -1, (0, 0, 255), 2)
            cv2.drawContours(frame, self.pickable_cuboids.contour.values.tolist(), -1, (0, 255, 255), 2)
            cv2.drawContours(frame, self.isolated.contour.values.tolist(), -1, (0, 255, 0), 2)
        return frame
    
    def video(self):
        window = self.cap.get_window()
        cuboid_choice = None

        while not self.shared_settings_inst.stop_event.is_set():
            frame = self.cap.get_frame(undist=True)
            plot_frame = frame.copy()

            with self.shared_settings_inst.lock:
                cuboid_chosen = self.shared_settings_inst.cuboid_chosen
                local_timer_set = self.shared_settings_inst.local_timer_set
                # self.current_idx = self.shared_settings_inst.idx
                self.routine = self.shared_settings_inst.routine

            x, y, z = self.openapi.get_position(verbose=False)[0].values()
            if not cuboid_chosen:
                self.cv_pipeline(frame)
            self.draw_annotations(plot_frame, (x, y, z))

            if not self.coord_queue.full() and not cuboid_chosen and not self.shared_settings_inst.pause_event.is_set() and len(self.isolated) > 0:
                if cuboid_choice is not None:
                    prev_x, prev_y = cuboid_choice[['cX', 'cY']].values[0]
                    
                    cv2.circle(plot_frame, (int(prev_x), int(prev_y)), int(round(self.config.failure_threshold / self.one_d_ratio)), (255, 0, 0), 2)
                    distances = self.cr.cuboid_df.apply(lambda row: np.sqrt((row['cX'] - prev_x)**2 + (row['cY'] - prev_y)**2), axis=1).to_numpy()
                    distances *= self.one_d_ratio
                    if any(distances <= self.config.failure_threshold):
                        with self.shared_settings_inst.lock:
                            self.shared_settings_inst.routine.update_well(success=False)
                        print(f"Miss detected at well {self.routine.current_well}.")
                    else:
                        with self.shared_settings_inst.lock:
                            self.shared_settings_inst.routine.update_well(success=True)
                            # print(f"Filled well {self.shared_settings_inst.routine.current_well}.")
                            self.shared_settings_inst.routine.get_next_well()
                            # print(f"Next well: {self.shared_settings_inst.routine.current_well}")

                if self.shared_settings_inst.routine.is_done():
                    self.shared_settings_inst.stop_event.set()
                    break

                cuboid_choice = self.isolated.sample(n=1) 
                cv2.drawContours(frame, cuboid_choice.contour.values.tolist(), -1, (255, 0, 0), 2)
                # cv2.imwrite(str(paths.BASE_DIR)+'\\outputs\\images\\'+f"frame_{idx}.png", frame)

                cX, cY = cuboid_choice[['cX', 'cY']].values[0]
                X_init, Y_init, _ = self.tf_mtx @ (cX, cY, 1)
                x, y, _ = self.openapi.get_position(verbose=False)[0].values()
                diff = np.array([x,y]) - np.array(self.calibration_data['calib_origin'])[:2]
                X = X_init + diff[0] + self.offset[0]
                Y = Y_init + diff[1] + self.offset[1]

                next_well = self.shared_settings_inst.routine.get_next_well()
                self.coord_queue.put((X, Y, next_well))
                with self.shared_settings_inst.lock:
                    self.shared_settings_inst.cuboid_chosen = True
            elif len(self.isolated) == 0 and not self.shared_settings_inst.pause_event.is_set():
                self.shared_settings_inst.pause_event.set()
                print("No cuboids found in the selected region. Pausing...")

            cv2.imshow(self.cap.window_name, plot_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.shared_settings_inst.stop_event.set()
            elif key == ord('p'):
                if self.shared_settings_inst.pause_event.is_set():
                    self.shared_settings_inst.pause_event.clear()
                    print("Resuming movement...")
                else:
                    print("Pausing movement...")
                    self.shared_settings_inst.pause_event.set()
        cv2.destroyAllWindows()

    def robot_movement(self):
        self.openapi.move_to_coordinates((self.calib_origin[0],self.calib_origin[1],100), min_z_height=self.config.dish_bottom, verbose=False)
        while not self.shared_settings_inst.stop_event.is_set():
            if self.shared_settings_inst.pause_event.is_set():
                time.sleep(0.1)  # Small sleep to prevent excessive CPU usage
                continue  # Skip to next iteration while paused
            # print("Moving robot...")
            try:
                # with self.shared_settings_inst.lock:
                # well = self.shared_settings_inst.routine.get_next_well()
                # Get latest coordinates from the queue (non-blocking)
                x, y, well = self.coord_queue.get(timeout=1)  # Timeout prevents indefinite blocking
                self.openapi.move_to_coordinates((x, y, self.config.pickup_height+20), min_z_height=self.config.dish_bottom, verbose=False, force_direct=True)
                self.openapi.move_to_coordinates((x, y, self.config.pickup_height), min_z_height=self.config.dish_bottom, verbose=False, force_direct=True)
                self.openapi.aspirate_in_place(flow_rate = self.config.flow_rate, volume = self.config.vol)
                self.openapi.move_relative('z', 20)

                # print(f'actually filling well {well}')
                self.openapi.move_to_well(self.openapi.labware_dct['6'], well, well_location='top', offset=(self.config.well_offset_x, self.config.well_offset_y, 5), verbose = False, force_direct = True)
                self.openapi.dispense(self.openapi.labware_dct['6'], well, well_location='bottom', offset=(self.config.well_offset_x, self.config.well_offset_y, 0), volume = self.config.vol, flow_rate = self.config.flow_rate)
                time.sleep(self.config.wait_time_after_deposit)
                self.openapi.move_to_well(self.openapi.labware_dct['6'], well, well_location='top', offset=(self.config.well_offset_x, self.config.well_offset_y, 5), verbose=False)
                self.openapi.move_to_coordinates((self.calib_origin[0],self.calib_origin[1],100), min_z_height=self.config.dish_bottom, verbose=False, force_direct=True)
                time.sleep(0.5)
                with self.shared_settings_inst.lock:
                    self.shared_settings_inst.cuboid_chosen = False
                    
            except queue.Empty:
                pass  # No new coordinates, continue looping