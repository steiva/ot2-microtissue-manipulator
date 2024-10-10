from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = os.path.join(BASE_DIR, 'configs', 'calibration.json')
CAMERA_INTRINSICS_PATH = os.path.join(BASE_DIR, 'configs', 'camera_intrinsics.json')