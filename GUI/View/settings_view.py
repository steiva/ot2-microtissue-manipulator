"""
Settings view for the microtissue manipulator GUI.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                           QPushButton, QLabel, QGroupBox, QMessageBox, QProgressBar,
                           QDoubleSpinBox, QLineEdit, QComboBox, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class SettingsView(QWidget):
    """Settings view widget for robot control and configuration."""
    
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setup_ui()
        self.setup_status_timer()
        self.update_robot_status()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create scroll area for the main content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        main_layout = QVBoxLayout(scroll_widget)
        
        # Robot Control Section
        robot_group = self.create_robot_control_group()
        main_layout.addWidget(robot_group)
        
        # Robot Status Section
        status_group = self.create_status_group()
        main_layout.addWidget(status_group)
        
        # Configuration Section
        config_group = self.create_configuration_group()
        main_layout.addWidget(config_group)
        
        # Additional Functions Section
        additional_group = self.create_additional_functions_group()
        main_layout.addWidget(additional_group)
        
        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)
    
    def create_robot_control_group(self):
        """Create robot control group."""
        group = QGroupBox("Robot Control")
        layout = QGridLayout()
        
        # Initialize Robot
        self.init_robot_btn = QPushButton("Initialize Robot")
        self.init_robot_btn.clicked.connect(self.on_initialize_robot)
        layout.addWidget(self.init_robot_btn, 0, 0)
        
        # Home Robot
        self.home_robot_btn = QPushButton("Home Robot")
        self.home_robot_btn.clicked.connect(self.on_home_robot)
        layout.addWidget(self.home_robot_btn, 0, 1)
        
        # Toggle Lights
        self.toggle_lights_btn = QPushButton("Toggle Lights")
        self.toggle_lights_btn.clicked.connect(self.on_toggle_lights)
        layout.addWidget(self.toggle_lights_btn, 0, 2)
        
        # Retract Axis
        self.retract_axis_btn = QPushButton("Retract Axis")
        self.retract_axis_btn.clicked.connect(self.on_retract_axis)
        layout.addWidget(self.retract_axis_btn, 1, 0)
        
        # Get Run Info
        self.get_run_info_btn = QPushButton("Get Run Info")
        self.get_run_info_btn.clicked.connect(self.on_get_run_info)
        layout.addWidget(self.get_run_info_btn, 1, 1)
        
        # Create Run
        self.create_run_btn = QPushButton("Create Run")
        self.create_run_btn.clicked.connect(self.on_create_run)
        layout.addWidget(self.create_run_btn, 1, 2)
        
        # Load Pipette
        self.load_pipette_btn = QPushButton("Load Pipette")
        self.load_pipette_btn.clicked.connect(self.on_load_pipette)
        layout.addWidget(self.load_pipette_btn, 2, 0)
        
        # Placeholder buttons
        self.placeholder_btn_1 = QPushButton("Placeholder 1")
        self.placeholder_btn_1.clicked.connect(self.on_placeholder_1)
        layout.addWidget(self.placeholder_btn_1, 2, 1)
        
        self.placeholder_btn_2 = QPushButton("Placeholder 2")
        self.placeholder_btn_2.clicked.connect(self.on_placeholder_2)
        layout.addWidget(self.placeholder_btn_2, 2, 2)
        
        self.placeholder_btn_3 = QPushButton("Placeholder 3")
        self.placeholder_btn_3.clicked.connect(self.on_placeholder_3)
        layout.addWidget(self.placeholder_btn_3, 3, 0)
        
        group.setLayout(layout)
        return group
    
    def create_status_group(self):
        """Create robot status group."""
        group = QGroupBox("Robot Status")
        layout = QVBoxLayout()
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        group.setLayout(layout)
        return group
    
    def create_configuration_group(self):
        """Create configuration group."""
        group = QGroupBox("Configuration")
        layout = QGridLayout()
        
        # Slot Offsets
        layout.addWidget(QLabel("Slot Offsets:"), 0, 0)
        
        layout.addWidget(QLabel("X:"), 1, 0)
        self.offset_x_spinbox = QDoubleSpinBox()
        self.offset_x_spinbox.setMinimum(-1000.0)
        self.offset_x_spinbox.setMaximum(1000.0)
        self.offset_x_spinbox.setDecimals(2)
        layout.addWidget(self.offset_x_spinbox, 1, 1)
        
        layout.addWidget(QLabel("Y:"), 1, 2)
        self.offset_y_spinbox = QDoubleSpinBox()
        self.offset_y_spinbox.setMinimum(-1000.0)
        self.offset_y_spinbox.setMaximum(1000.0)
        self.offset_y_spinbox.setDecimals(2)
        layout.addWidget(self.offset_y_spinbox, 1, 3)
        
        layout.addWidget(QLabel("Z:"), 1, 4)
        self.offset_z_spinbox = QDoubleSpinBox()
        self.offset_z_spinbox.setMinimum(-1000.0)
        self.offset_z_spinbox.setMaximum(1000.0)
        self.offset_z_spinbox.setDecimals(2)
        layout.addWidget(self.offset_z_spinbox, 1, 5)
        
        # Add Slot Offsets Button
        self.add_offsets_btn = QPushButton("Add Slot Offsets")
        self.add_offsets_btn.clicked.connect(self.on_add_slot_offsets)
        layout.addWidget(self.add_offsets_btn, 2, 0, 1, 2)
        
        # Pipette Configuration
        layout.addWidget(QLabel("Pipette Type:"), 3, 0)
        self.pipette_type_combo = QComboBox()
        self.pipette_type_combo.addItems(["p300_single", "p300_multi", "p1000_single", "p20_single"])
        layout.addWidget(self.pipette_type_combo, 3, 1)
        
        layout.addWidget(QLabel("Mount:"), 3, 2)
        self.mount_combo = QComboBox()
        self.mount_combo.addItems(["left", "right"])
        layout.addWidget(self.mount_combo, 3, 3)
        
        # Axis Selection for Retraction
        layout.addWidget(QLabel("Retract Axis:"), 4, 0)
        self.retract_axis_combo = QComboBox()
        self.retract_axis_combo.addItems(["X", "Y", "Z", "A", "B", "C"])
        layout.addWidget(self.retract_axis_combo, 4, 1)
        
        group.setLayout(layout)
        return group
    
    def create_additional_functions_group(self):
        """Create additional functions group."""
        group = QGroupBox("Additional Functions")
        layout = QVBoxLayout()
        
        # Run configuration
        run_config_layout = QHBoxLayout()
        run_config_layout.addWidget(QLabel("Run Name:"))
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setText("Default Run")
        run_config_layout.addWidget(self.run_name_edit)
        
        layout.addLayout(run_config_layout)
        
        # Instructions
        instructions = QLabel("""
        Instructions:
        
        1. Initialize Robot: Connect to and initialize the robot system
        2. Add Slot Offsets: Configure positional offsets for precise alignment
        3. Toggle Lights: Turn robot lighting on/off
        4. Home Robot: Move robot to home position
        5. Get Run Info: Retrieve current run status and information
        6. Retract Axis: Safely retract specified axis
        7. Create Run: Initialize a new experimental run
        8. Load Pipette: Mount and configure pipette
        
        Note: Some functions require the robot to be initialized first.
        """)
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(instructions)
        
        group.setLayout(layout)
        return group
    
    def setup_status_timer(self):
        """Setup timer for status updates."""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_robot_status)
        self.status_timer.start(2000)  # Update every 2 seconds
    
    def update_robot_status(self):
        """Update robot status display."""
        try:
            status = self.controller.get_robot_status()
            
            status_text = f"Robot Initialized: {'Yes' if status['initialized'] else 'No'}\\n"
            status_text += f"Lights: {'On' if status['lights_on'] else 'Off'}\\n"
            
            if status['run_info']:
                status_text += f"\\nRun Info:\\n"
                for key, value in status['run_info'].items():
                    status_text += f"  {key}: {value}\\n"
            
            self.status_text.setPlainText(status_text)
            
        except Exception as e:
            self.status_text.setPlainText(f"Error updating status: {str(e)}")
    
    def show_progress(self, message: str = "Processing..."):
        """Show progress bar with message."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_text.append(f"\\n{message}")
    
    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.setVisible(False)
    
    def show_result(self, success: bool, message: str):
        """Show operation result."""
        self.hide_progress()
        if success:
            self.status_text.append(f"✓ {message}")
        else:
            self.status_text.append(f"✗ {message}")
    
    # Event handlers
    def on_initialize_robot(self):
        """Handle initialize robot button click."""
        self.show_progress("Initializing robot...")
        success = self.controller.initialize_robot()
        self.show_result(success, "Robot initialized" if success else "Failed to initialize robot")
    
    def on_home_robot(self):
        """Handle home robot button click."""
        self.show_progress("Homing robot...")
        success = self.controller.home_robot()
        self.show_result(success, "Robot homed" if success else "Failed to home robot")
    
    def on_toggle_lights(self):
        """Handle toggle lights button click."""
        success = self.controller.toggle_lights()
        self.show_result(success, "Lights toggled" if success else "Failed to toggle lights")
    
    def on_retract_axis(self):
        """Handle retract axis button click."""
        axis = self.retract_axis_combo.currentText()
        self.show_progress(f"Retracting {axis} axis...")
        success = self.controller.retract_axis(axis)
        self.show_result(success, f"{axis} axis retracted" if success else f"Failed to retract {axis} axis")
    
    def on_get_run_info(self):
        """Handle get run info button click."""
        self.show_progress("Getting run info...")
        run_info = self.controller.get_run_info()
        if run_info:
            self.show_result(True, "Run info retrieved")
        else:
            self.show_result(False, "Failed to get run info")
    
    def on_create_run(self):
        """Handle create run button click."""
        run_name = self.run_name_edit.text().strip()
        if not run_name:
            QMessageBox.warning(self, "Warning", "Please enter a run name.")
            return
        
        run_config = {
            "name": run_name,
            "timestamp": "2025-01-01 12:00:00",
            "status": "created"
        }
        
        self.show_progress("Creating run...")
        success = self.controller.create_run(run_config)
        self.show_result(success, f"Run '{run_name}' created" if success else "Failed to create run")
    
    def on_load_pipette(self):
        """Handle load pipette button click."""
        pipette_type = self.pipette_type_combo.currentText()
        mount = self.mount_combo.currentText()
        
        self.show_progress(f"Loading {pipette_type} on {mount} mount...")
        success = self.controller.load_pipette(pipette_type, mount)
        self.show_result(success, f"Pipette loaded" if success else "Failed to load pipette")
    
    def on_add_slot_offsets(self):
        """Handle add slot offsets button click."""
        x = self.offset_x_spinbox.value()
        y = self.offset_y_spinbox.value()
        z = self.offset_z_spinbox.value()
        
        self.show_progress("Adding slot offsets...")
        success = self.controller.add_slot_offsets(x, y, z)
        self.show_result(success, f"Slot offsets added: X={x}, Y={y}, Z={z}" if success else "Failed to add slot offsets")
    
    def on_placeholder_1(self):
        """Handle placeholder 1 button click."""
        self.show_progress("Executing placeholder function 1...")
        success = self.controller.placeholder_function_1()
        self.show_result(success, "Placeholder function 1 executed" if success else "Placeholder function 1 failed")
    
    def on_placeholder_2(self):
        """Handle placeholder 2 button click."""
        self.show_progress("Executing placeholder function 2...")
        success = self.controller.placeholder_function_2()
        self.show_result(success, "Placeholder function 2 executed" if success else "Placeholder function 2 failed")
    
    def on_placeholder_3(self):
        """Handle placeholder 3 button click."""
        self.show_progress("Executing placeholder function 3...")
        success = self.controller.placeholder_function_3()
        self.show_result(success, "Placeholder function 3 executed" if success else "Placeholder function 3 failed")
