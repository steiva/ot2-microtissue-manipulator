from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent#.parent
PROFILES_DIR = os.path.join(BASE_DIR, 'configs')
CAM_CONFIGS_DIR = os.path.join(BASE_DIR, 'cam_configs')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')