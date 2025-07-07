"""
Labware model for the microtissue manipulator GUI.
Contains functionality for managing labware declarations and configurations.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple


class LabwareModel:
    """Model for handling labware declarations and configurations."""
    
    def __init__(self, labware_file: str = "labware_config.json"):
        self.labware_file = labware_file
        self.labware_config = self.load_labware_config()
        self.available_labware = self.get_available_labware()
        
    def load_labware_config(self) -> Dict[str, Any]:
        """Load labware configuration from JSON file."""
        if os.path.exists(self.labware_file):
            try:
                with open(self.labware_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self.get_default_labware_config()
        return self.get_default_labware_config()
    
    def save_labware_config(self) -> None:
        """Save current labware configuration to JSON file."""
        try:
            with open(self.labware_file, 'w') as f:
                json.dump(self.labware_config, f, indent=2)
        except IOError as e:
            print(f"Error saving labware config: {e}")
    
    def get_default_labware_config(self) -> Dict[str, Any]:
        """Get default labware configuration."""
        return {
            "deck_layout": {
                "slot_1": None,
                "slot_2": None,
                "slot_3": None,
                "slot_4": None,
                "slot_5": None,
                "slot_6": None,
                "slot_7": None,
                "slot_8": None,
                "slot_9": None,
                "slot_10": None,
                "slot_11": None,
                "slot_12": None
            },
            "custom_labware": {},
            "current_protocol": None
        }
    
    def get_available_labware(self) -> List[Dict[str, Any]]:
        """Get list of available labware types."""
        return [
            {
                "name": "96-well plate",
                "type": "96_well_plate",
                "description": "Standard 96-well microplate",
                "dimensions": {"rows": 8, "columns": 12}
            },
            {
                "name": "384-well plate",
                "type": "384_well_plate", 
                "description": "High-density 384-well microplate",
                "dimensions": {"rows": 16, "columns": 24}
            },
            {
                "name": "12-well reservoir",
                "type": "12_well_reservoir",
                "description": "12-well reagent reservoir",
                "dimensions": {"rows": 1, "columns": 12}
            },
            {
                "name": "200μL tip rack",
                "type": "tip_rack_200ul",
                "description": "200μL pipette tip rack",
                "dimensions": {"rows": 8, "columns": 12}
            },
            {
                "name": "1000μL tip rack",
                "type": "tip_rack_1000ul",
                "description": "1000μL pipette tip rack",
                "dimensions": {"rows": 8, "columns": 12}
            },
            {
                "name": "15mL tube rack",
                "type": "tube_rack_15ml",
                "description": "15mL conical tube rack",
                "dimensions": {"rows": 3, "columns": 5}
            },
            {
                "name": "50mL tube rack",
                "type": "tube_rack_50ml",
                "description": "50mL conical tube rack",
                "dimensions": {"rows": 2, "columns": 3}
            },
            {
                "name": "PCR plate",
                "type": "pcr_plate_96",
                "description": "96-well PCR plate",
                "dimensions": {"rows": 8, "columns": 12}
            }
        ]
    
    def get_slot_configuration(self, slot: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific deck slot."""
        return self.labware_config["deck_layout"].get(slot)
    
    def set_slot_configuration(self, slot: str, labware_type: str, labware_name: str = None) -> bool:
        """Set labware configuration for a specific deck slot."""
        try:
            if slot not in self.labware_config["deck_layout"]:
                print(f"Invalid slot: {slot}")
                return False
            
            # Find labware info
            labware_info = None
            for labware in self.available_labware:
                if labware["type"] == labware_type:
                    labware_info = labware
                    break
            
            if not labware_info:
                print(f"Unknown labware type: {labware_type}")
                return False
            
            self.labware_config["deck_layout"][slot] = {
                "labware_type": labware_type,
                "labware_name": labware_name or labware_info["name"],
                "labware_info": labware_info
            }
            
            self.save_labware_config()
            return True
        except Exception as e:
            print(f"Error setting slot configuration: {e}")
            return False
    
    def clear_slot(self, slot: str) -> bool:
        """Clear labware from a specific deck slot."""
        try:
            if slot not in self.labware_config["deck_layout"]:
                print(f"Invalid slot: {slot}")
                return False
            
            self.labware_config["deck_layout"][slot] = None
            self.save_labware_config()
            return True
        except Exception as e:
            print(f"Error clearing slot: {e}")
            return False
    
    def get_deck_layout(self) -> Dict[str, Any]:
        """Get the current deck layout configuration."""
        return self.labware_config["deck_layout"]
    
    def clear_deck(self) -> bool:
        """Clear all labware from the deck."""
        try:
            for slot in self.labware_config["deck_layout"]:
                self.labware_config["deck_layout"][slot] = None
            self.save_labware_config()
            return True
        except Exception as e:
            print(f"Error clearing deck: {e}")
            return False
    
    def add_custom_labware(self, name: str, labware_type: str, dimensions: Dict[str, int], 
                          description: str = "") -> bool:
        """Add a custom labware definition."""
        try:
            custom_labware = {
                "name": name,
                "type": labware_type,
                "description": description,
                "dimensions": dimensions,
                "custom": True
            }
            
            self.labware_config["custom_labware"][labware_type] = custom_labware
            self.available_labware.append(custom_labware)
            self.save_labware_config()
            return True
        except Exception as e:
            print(f"Error adding custom labware: {e}")
            return False
    
    def remove_custom_labware(self, labware_type: str) -> bool:
        """Remove a custom labware definition."""
        try:
            if labware_type in self.labware_config["custom_labware"]:
                del self.labware_config["custom_labware"][labware_type]
                
                # Remove from available labware list
                self.available_labware = [lw for lw in self.available_labware 
                                        if lw["type"] != labware_type]
                
                self.save_labware_config()
                return True
            return False
        except Exception as e:
            print(f"Error removing custom labware: {e}")
            return False
    
    def validate_deck_layout(self) -> Tuple[bool, List[str]]:
        """Validate the current deck layout and return any issues."""
        issues = []
        
        try:
            # Check for required labware
            has_tip_rack = False
            has_sample_plate = False
            
            for slot, config in self.labware_config["deck_layout"].items():
                if config is None:
                    continue
                    
                labware_type = config.get("labware_type", "")
                
                if "tip_rack" in labware_type:
                    has_tip_rack = True
                elif "plate" in labware_type:
                    has_sample_plate = True
            
            if not has_tip_rack:
                issues.append("No tip rack found on deck")
            
            if not has_sample_plate:
                issues.append("No sample plate found on deck")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Error validating deck layout: {e}")
            return False, issues
    
    def export_deck_layout(self, filename: str) -> bool:
        """Export current deck layout to a file."""
        try:
            export_data = {
                "deck_layout": self.labware_config["deck_layout"],
                "custom_labware": self.labware_config["custom_labware"],
                "timestamp": "2025-01-01 12:00:00"
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting deck layout: {e}")
            return False
    
    def import_deck_layout(self, filename: str) -> bool:
        """Import deck layout from a file."""
        try:
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                return False
            
            with open(filename, 'r') as f:
                import_data = json.load(f)
            
            if "deck_layout" in import_data:
                self.labware_config["deck_layout"] = import_data["deck_layout"]
            
            if "custom_labware" in import_data:
                self.labware_config["custom_labware"] = import_data["custom_labware"]
                # Update available labware with custom ones
                for labware in import_data["custom_labware"].values():
                    if labware not in self.available_labware:
                        self.available_labware.append(labware)
            
            self.save_labware_config()
            return True
        except Exception as e:
            print(f"Error importing deck layout: {e}")
            return False
    
    def get_labware_by_type(self, labware_type: str) -> Optional[Dict[str, Any]]:
        """Get labware information by type."""
        for labware in self.available_labware:
            if labware["type"] == labware_type:
                return labware
        return None
    
    def get_occupied_slots(self) -> List[str]:
        """Get list of slots that have labware assigned."""
        occupied = []
        for slot, config in self.labware_config["deck_layout"].items():
            if config is not None:
                occupied.append(slot)
        return occupied
    
    def get_empty_slots(self) -> List[str]:
        """Get list of empty slots on the deck."""
        empty = []
        for slot, config in self.labware_config["deck_layout"].items():
            if config is None:
                empty.append(slot)
        return empty
