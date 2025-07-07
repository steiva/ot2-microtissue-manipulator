"""
Camera view for the microtissue manipulator GUI.
"""

import cv2
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                           QListWidgetItem, QPushButton, QLabel, QSlider, 
                           QSpinBox, QGroupBox, QMessageBox, QDialog, QDialogButtonBox,
                           QSplitter, QFrame, QCheckBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from View.zoomable_video_widget import VideoDisplayWidget


class CameraTestWindow(QDialog):
    """Separate window for testing individual cameras."""
    
    def __init__(self, camera_name: str, camera_index: int, controller, parent=None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.camera_index = camera_index
        self.controller = controller
        self.is_capturing = False
        
        self.setWindowTitle(f"Camera Test - {camera_name}")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        self.setup_timer()
        
        # Start camera capture
        self.start_capture()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Camera info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"Camera: {self.camera_name}"))
        info_layout.addWidget(QLabel(f"Index: {self.camera_index}"))
        info_layout.addStretch()
        
        # Camera controls
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QHBoxLayout()
        
        # Focus control
        focus_layout = QVBoxLayout()
        focus_layout.addWidget(QLabel("Focus:"))
        self.focus_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_slider.setMinimum(0)
        self.focus_slider.setMaximum(255)
        self.focus_slider.setValue(128)
        self.focus_slider.valueChanged.connect(self.on_focus_changed)
        focus_layout.addWidget(self.focus_slider)
        
        self.focus_value_label = QLabel("128")
        focus_layout.addWidget(self.focus_value_label)
        
        # Control buttons
        button_layout = QVBoxLayout()
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        button_layout.addWidget(self.reset_view_btn)
        
        self.capture_btn = QPushButton("Stop Capture")
        self.capture_btn.clicked.connect(self.toggle_capture)
        button_layout.addWidget(self.capture_btn)
        
        controls_layout.addLayout(focus_layout)
        controls_layout.addLayout(button_layout)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        
        # Video display
        self.video_display = VideoDisplayWidget()
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        
        # Add to main layout
        layout.addLayout(info_layout)
        layout.addWidget(controls_group)
        layout.addWidget(self.video_display)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def setup_timer(self):
        """Setup timer for video updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
    
    def start_capture(self):
        """Start camera capture."""
        success = self.controller.start_camera_capture(
            self.camera_name, 
            self.camera_index,
            focus=self.focus_slider.value()
        )
        
        if success:
            self.is_capturing = True
            self.capture_btn.setText("Stop Capture")
        else:
            QMessageBox.warning(self, "Error", f"Failed to start capture for {self.camera_name}")
    
    def stop_capture(self):
        """Stop camera capture."""
        self.controller.stop_camera_capture(self.camera_name)
        self.is_capturing = False
        self.capture_btn.setText("Start Capture")
        self.video_display.clear_frame()
    
    def toggle_capture(self):
        """Toggle camera capture."""
        if self.is_capturing:
            self.stop_capture()
        else:
            self.start_capture()
    
    def update_frame(self):
        """Update the video frame."""
        if self.is_capturing:
            ret, frame = self.controller.get_camera_frame(self.camera_name)
            if ret and frame is not None:
                self.video_display.set_frame(frame)
    
    def on_focus_changed(self, value):
        """Handle focus slider changes."""
        self.focus_value_label.setText(str(value))
        if self.is_capturing:
            self.controller.set_camera_focus(self.camera_name, value)
    
    def reset_view(self):
        """Reset the video view."""
        self.video_display.reset_view()
    
    def closeEvent(self, event):
        """Handle close event."""
        self.timer.stop()
        if self.is_capturing:
            self.stop_capture()
        event.accept()


class CameraView(QWidget):
    """Main camera view widget."""
    
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.test_windows = {}
        self.active_embedded_cameras = {}
        
        self.setup_ui()
        self.setup_timer()
        self.refresh_cameras()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Camera list and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Embedded camera display
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 800])
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(main_splitter)
        self.setLayout(layout)
        
        # Connect camera list selection
        self.camera_list.itemSelectionChanged.connect(self.on_camera_selection_changed)
    
    def create_left_panel(self):
        """Create the left control panel."""
        left_widget = QWidget()
        layout = QVBoxLayout()
        
        # Camera list group
        camera_group = QGroupBox("Available Cameras")
        camera_layout = QVBoxLayout()
        
        # Camera list
        self.camera_list = QListWidget()
        self.camera_list.itemDoubleClicked.connect(self.on_camera_double_clicked)
        camera_layout.addWidget(self.camera_list)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(self.refresh_btn)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Camera controls group
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QVBoxLayout()
        
        # Selected camera info
        self.selected_camera_label = QLabel("No camera selected")
        self.selected_camera_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        controls_layout.addWidget(self.selected_camera_label)
        
        # Display option
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Embedded View", "Separate Window"])
        display_layout.addWidget(self.display_combo)
        controls_layout.addLayout(display_layout)
        
        # Start/Stop camera button
        self.start_stop_btn = QPushButton("Start Camera")
        self.start_stop_btn.clicked.connect(self.toggle_selected_camera)
        self.start_stop_btn.setEnabled(False)
        controls_layout.addWidget(self.start_stop_btn)
        
        # Focus control
        focus_layout = QVBoxLayout()
        focus_layout.addWidget(QLabel("Focus:"))
        self.focus_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_slider.setMinimum(0)
        self.focus_slider.setMaximum(255)
        self.focus_slider.setValue(128)
        self.focus_slider.valueChanged.connect(self.on_focus_changed)
        self.focus_slider.setEnabled(False)
        focus_layout.addWidget(self.focus_slider)
        
        self.focus_value_label = QLabel("128")
        focus_layout.addWidget(self.focus_value_label)
        controls_layout.addLayout(focus_layout)
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_embedded_view)
        self.reset_view_btn.setEnabled(False)
        controls_layout.addWidget(self.reset_view_btn)
        
        # Stop all cameras button
        self.stop_all_btn = QPushButton("Stop All Cameras")
        self.stop_all_btn.clicked.connect(self.stop_all_cameras)
        controls_layout.addWidget(self.stop_all_btn)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions = QLabel("""
        Camera Controls:
        
        • Select a camera from the list
        • Choose display mode (Embedded/Window)
        • Click "Start Camera" to begin capture
        • Use focus slider to adjust camera focus
        • Double-click camera list for separate window
        
        Embedded View Controls:
        • Mouse wheel: Zoom in/out
        • Click and drag: Pan around
        • Double-click: Reset view
        • Reset View button: Center and reset zoom
        
        Tips:
        • Multiple cameras can run simultaneously
        • Focus adjustment works in real-time
        • Embedded view is zoomable and pannable
        """)
        
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignTop)
        instructions_layout.addWidget(instructions)
        
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        left_widget.setLayout(layout)
        left_widget.setMaximumWidth(400)
        return left_widget
    
    def create_right_panel(self):
        """Create the right panel with embedded camera display."""
        right_widget = QWidget()
        layout = QVBoxLayout()
        
        # Embedded camera display group
        display_group = QGroupBox("Embedded Camera Display")
        display_layout = QVBoxLayout()
        
        # Active camera info
        self.active_camera_label = QLabel("No active camera")
        self.active_camera_label.setStyleSheet("font-weight: bold; color: #009900;")
        display_layout.addWidget(self.active_camera_label)
        
        # Video display widget
        self.embedded_video_display = VideoDisplayWidget()
        self.embedded_video_display.setMinimumSize(640, 480)
        display_layout.addWidget(self.embedded_video_display)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        right_widget.setLayout(layout)
        return right_widget
    
    def setup_timer(self):
        """Setup timer for video updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_embedded_display)
        self.timer.start(33)  # ~30 FPS
    
    def refresh_cameras(self):
        """Refresh the list of available cameras."""
        self.camera_list.clear()
        
        cameras = self.controller.get_available_cameras()
        
        if not cameras:
            item = QListWidgetItem("No cameras detected")
            item.setData(Qt.ItemDataRole.UserRole, None)
            self.camera_list.addItem(item)
            return
        
        for camera_name, camera_index in cameras:
            item = QListWidgetItem(f"{camera_name} (Index: {camera_index})")
            item.setData(Qt.ItemDataRole.UserRole, (camera_name, camera_index))
            self.camera_list.addItem(item)
    
    def on_camera_selection_changed(self):
        """Handle camera selection changes."""
        current_item = self.camera_list.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole) is not None:
            camera_data = current_item.data(Qt.ItemDataRole.UserRole)
            camera_name, camera_index = camera_data
            
            self.selected_camera_label.setText(f"Selected: {camera_name}")
            self.start_stop_btn.setEnabled(True)
            
            # Update button text based on current state
            if camera_name in self.active_embedded_cameras:
                self.start_stop_btn.setText("Stop Camera")
                self.focus_slider.setEnabled(True)
                self.reset_view_btn.setEnabled(True)
            else:
                self.start_stop_btn.setText("Start Camera")
                self.focus_slider.setEnabled(False)
                self.reset_view_btn.setEnabled(False)
        else:
            self.selected_camera_label.setText("No camera selected")
            self.start_stop_btn.setEnabled(False)
            self.focus_slider.setEnabled(False)
            self.reset_view_btn.setEnabled(False)
    
    def toggle_selected_camera(self):
        """Toggle the selected camera on/off."""
        current_item = self.camera_list.currentItem()
        if not current_item:
            return
        
        camera_data = current_item.data(Qt.ItemDataRole.UserRole)
        if not camera_data:
            return
        
        camera_name, camera_index = camera_data
        display_mode = self.display_combo.currentText()
        
        if camera_name in self.active_embedded_cameras:
            # Stop camera
            self.stop_embedded_camera(camera_name)
        else:
            # Start camera
            if display_mode == "Embedded View":
                self.start_embedded_camera(camera_name, camera_index)
            else:
                self.test_camera(camera_name, camera_index)
    
    def start_embedded_camera(self, camera_name: str, camera_index: int):
        """Start a camera in embedded view."""
        try:
            # Stop any currently active embedded camera
            if self.active_embedded_cameras:
                for active_name in list(self.active_embedded_cameras.keys()):
                    self.stop_embedded_camera(active_name)
            
            # Start new camera
            success = self.controller.start_camera_capture(
                camera_name, 
                camera_index,
                focus=self.focus_slider.value()
            )
            
            if success:
                self.active_embedded_cameras[camera_name] = camera_index
                self.active_camera_label.setText(f"Active: {camera_name}")
                self.start_stop_btn.setText("Stop Camera")
                self.focus_slider.setEnabled(True)
                self.reset_view_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", f"Failed to start camera: {camera_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting camera: {str(e)}")
    
    def stop_embedded_camera(self, camera_name: str):
        """Stop an embedded camera."""
        try:
            self.controller.stop_camera_capture(camera_name)
            
            if camera_name in self.active_embedded_cameras:
                del self.active_embedded_cameras[camera_name]
            
            self.active_camera_label.setText("No active camera")
            self.embedded_video_display.clear_frame()
            self.start_stop_btn.setText("Start Camera")
            self.focus_slider.setEnabled(False)
            self.reset_view_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error stopping camera: {str(e)}")
    
    def stop_all_cameras(self):
        """Stop all active cameras."""
        # Stop embedded cameras
        for camera_name in list(self.active_embedded_cameras.keys()):
            self.stop_embedded_camera(camera_name)
        
        # Close all test windows
        for window in list(self.test_windows.values()):
            window.close()
        
        QMessageBox.information(self, "Info", "All cameras stopped")
    
    def update_embedded_display(self):
        """Update the embedded camera display."""
        if self.active_embedded_cameras:
            # Get the first (and should be only) active camera
            camera_name = list(self.active_embedded_cameras.keys())[0]
            ret, frame = self.controller.get_camera_frame(camera_name)
            
            if ret and frame is not None:
                self.embedded_video_display.set_frame(frame)
    
    def on_focus_changed(self, value):
        """Handle focus slider changes."""
        self.focus_value_label.setText(str(value))
        
        # Apply focus to active embedded camera
        if self.active_embedded_cameras:
            camera_name = list(self.active_embedded_cameras.keys())[0]
            self.controller.set_camera_focus(camera_name, value)
    
    def reset_embedded_view(self):
        """Reset the embedded video view."""
        self.embedded_video_display.reset_view()
    
    def on_camera_double_clicked(self, item):
        """Handle double-click on camera item."""
        camera_data = item.data(Qt.ItemDataRole.UserRole)
        if camera_data is not None:
            self.test_camera(camera_data[0], camera_data[1])
    
    def test_camera(self, camera_name: str, camera_index: int):
        """Open a test window for the specified camera."""
        # Close existing test window for this camera if it exists
        if camera_name in self.test_windows:
            self.test_windows[camera_name].close()
        
        # Create new test window
        test_window = CameraTestWindow(camera_name, camera_index, self.controller, self)
        test_window.show()
        
        # Store reference
        self.test_windows[camera_name] = test_window
        
        # Connect close event to clean up reference
        test_window.finished.connect(lambda: self.on_test_window_closed(camera_name))
    
    def on_test_window_closed(self, camera_name: str):
        """Handle test window closure."""
        if camera_name in self.test_windows:
            del self.test_windows[camera_name]
    
    def closeEvent(self, event):
        """Handle close event."""
        # Stop all embedded cameras
        for camera_name in list(self.active_embedded_cameras.keys()):
            self.stop_embedded_camera(camera_name)
        
        # Close all test windows
        for window in list(self.test_windows.values()):
            window.close()
        
        event.accept()
