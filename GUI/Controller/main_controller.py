"""
Main controller for the microtissue manipulator GUI.
Manages communication between models and views.
"""

from typing import Dict, Any, List, Optional
from Model.camera import CameraManagerWindows, MultiprocessVideoCapture
from Model.settings import SettingsModel
from Model.labware import LabwareModel


class MainController:
    """Main controller that coordinates between models and views."""
    
    def __init__(self):
        # Initialize models
        self.camera_manager = CameraManagerWindows()
        self.settings_model = SettingsModel()
        self.labware_model = LabwareModel()
        # Active camera captures
        self.active_cameras: Dict[str, MultiprocessVideoCapture] = {}
        
        # View references (will be set by the main view)
        self.main_view = None
        self.settings_view = None
        self.labware_view = None
        self.camera_view = None
    
    def set_views(self, main_view, settings_view, labware_view, camera_view):
        """Set references to view components."""
        self.main_view = main_view
        self.settings_view = settings_view
        self.labware_view = labware_view
        self.camera_view = camera_view
    
    # Camera control methods
    def get_available_cameras(self) -> List[tuple]:
        """Get list of available cameras."""
        return self.camera_manager.get_available_cameras()
    
    def start_camera_capture(self, camera_name: str, camera_index: int, 
                           width: int = 640, height: int = 480, focus: int = None) -> bool:
        """Start capturing from a specific camera."""
        try:
            if camera_name in self.active_cameras:
                self.stop_camera_capture(camera_name)
            
            capture = MultiprocessVideoCapture(camera_index, width, height, focus=focus)
            self.active_cameras[camera_name] = capture
            return True
        except Exception as e:
            print(f"Error starting camera capture: {e}")
            return False
    
    def stop_camera_capture(self, camera_name: str) -> bool:
        """Stop capturing from a specific camera."""
        try:
            if camera_name in self.active_cameras:
                self.active_cameras[camera_name].release()
                del self.active_cameras[camera_name]
            return True
        except Exception as e:
            print(f"Error stopping camera capture: {e}")
            return False
    
    def get_camera_frame(self, camera_name: str):
        """Get the latest frame from a specific camera."""
        if camera_name in self.active_cameras:
            return self.active_cameras[camera_name].read()
        return False, None
    
    def set_camera_focus(self, camera_name: str, focus_value: int) -> bool:
        """Set focus for a specific camera."""
        try:
            if camera_name in self.active_cameras:
                self.active_cameras[camera_name].set_focus(focus_value)
                return True
            return False
        except Exception as e:
            print(f"Error setting camera focus: {e}")
            return False
    
    def is_camera_active(self, camera_name: str) -> bool:
        """Check if a camera is actively capturing."""
        return camera_name in self.active_cameras
    
    # Settings control methods
    def initialize_robot(self) -> bool:
        """Initialize robot connection."""
        return self.settings_model.initialize_robot()
    
    def add_slot_offsets(self, x: float, y: float, z: float) -> bool:
        """Add slot offsets."""
        return self.settings_model.add_slot_offsets(x, y, z)
    
    def toggle_lights(self) -> bool:
        """Toggle robot lights."""
        return self.settings_model.toggle_lights()
    
    def home_robot(self) -> bool:
        """Home the robot."""
        return self.settings_model.home_robot()
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get current run information."""
        return self.settings_model.get_run_info()
    
    def retract_axis(self, axis: str) -> bool:
        """Retract a specific axis."""
        return self.settings_model.retract_axis(axis)
    
    def create_run(self, run_config: Dict[str, Any]) -> bool:
        """Create a new run."""
        return self.settings_model.create_run(run_config)
    
    def load_pipette(self, pipette_type: str, mount: str) -> bool:
        """Load a pipette."""
        return self.settings_model.load_pipette(pipette_type, mount)
    
    def placeholder_function_1(self) -> bool:
        """Placeholder function 1."""
        return self.settings_model.placeholder_function_1()
    
    def placeholder_function_2(self) -> bool:
        """Placeholder function 2."""
        return self.settings_model.placeholder_function_2()
    
    def placeholder_function_3(self) -> bool:
        """Placeholder function 3."""
        return self.settings_model.placeholder_function_3()
    
    def get_robot_status(self) -> Dict[str, Any]:
        """Get current robot status."""
        return {
            "initialized": self.settings_model.is_robot_initialized(),
            "lights_on": self.settings_model.get_lights_status(),
            "run_info": self.settings_model.current_run_info
        }
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.settings_model.settings
    
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a specific setting."""
        try:
            self.settings_model.set_setting(key, value)
            return True
        except Exception as e:
            print(f"Error updating setting: {e}")
            return False
    
    # Labware control methods
    def get_available_labware(self) -> List[Dict[str, Any]]:
        """Get list of available labware."""
        return self.labware_model.get_available_labware()
    
    def get_deck_layout(self) -> Dict[str, Any]:
        """Get current deck layout."""
        return self.labware_model.get_deck_layout()
    
    def set_slot_labware(self, slot: str, labware_type: str, labware_name: str = None) -> bool:
        """Set labware for a specific slot."""
        success = self.labware_model.set_slot_configuration(slot, labware_type, labware_name)
        if success and self.labware_view:
            self.labware_view.update_deck_display()
        return success
    
    def clear_slot(self, slot: str) -> bool:
        """Clear labware from a specific slot."""
        success = self.labware_model.clear_slot(slot)
        if success and self.labware_view:
            self.labware_view.update_deck_display()
        return success
    
    def clear_deck(self) -> bool:
        """Clear all labware from deck."""
        success = self.labware_model.clear_deck()
        if success and self.labware_view:
            self.labware_view.update_deck_display()
        return success
    
    def validate_deck_layout(self) -> tuple:
        """Validate current deck layout."""
        return self.labware_model.validate_deck_layout()
    
    def export_deck_layout(self, filename: str) -> bool:
        """Export deck layout to file."""
        return self.labware_model.export_deck_layout(filename)
    
    def import_deck_layout(self, filename: str) -> bool:
        """Import deck layout from file."""
        success = self.labware_model.import_deck_layout(filename)
        if success and self.labware_view:
            self.labware_view.update_deck_display()
        return success
    
    def add_custom_labware(self, name: str, labware_type: str, dimensions: Dict[str, int], 
                          description: str = "") -> bool:
        """Add custom labware definition."""
        success = self.labware_model.add_custom_labware(name, labware_type, dimensions, description)
        if success and self.labware_view:
            self.labware_view.update_labware_list()
        return success
    
    def get_slot_info(self, slot: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific slot."""
        return self.labware_model.get_slot_configuration(slot)
    
    def get_occupied_slots(self) -> List[str]:
        """Get list of occupied slots."""
        return self.labware_model.get_occupied_slots()
    
    def get_empty_slots(self) -> List[str]:
        """Get list of empty slots."""
        return self.labware_model.get_empty_slots()
    
    # Cleanup methods
    def cleanup(self):
        """Cleanup resources when closing application."""
        # Stop all active cameras
        for camera_name in list(self.active_cameras.keys()):
            self.stop_camera_capture(camera_name)
        
        # Save settings
        self.settings_model.save_settings()
        self.labware_model.save_labware_config()
        
        print("Application cleanup completed")
