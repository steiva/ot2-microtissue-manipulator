"""
Main GUI application for the microtissue manipulator.
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QMessageBox, QStatusBar, QMenuBar, QMenu)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

from Controller.main_controller import MainController
from View.camera_view import CameraView
from View.settings_view import SettingsView
from View.labware_view import LabwareView


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.controller = MainController()
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Connect controller to views
        self.controller.set_views(self, self.settings_view, self.labware_view, self.camera_view)
    
    def setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("Microtissue Manipulator Control")
        self.setMinimumSize(1200, 800)
        
        # Create central widget with tab widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.create_tabs()
        
        layout.addWidget(self.tab_widget)
    
    def create_tabs(self):
        """Create the main tabs."""
        # Settings tab
        self.settings_view = SettingsView(self.controller)
        self.tab_widget.addTab(self.settings_view, "Settings")
        
        # Labware Declaration tab
        self.labware_view = LabwareView(self.controller)
        self.tab_widget.addTab(self.labware_view, "Labware Declaration")
        
        # Camera View tab
        self.camera_view = CameraView(self.controller)
        self.tab_widget.addTab(self.camera_view, "Camera View")
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New action
        new_action = QAction("New Configuration", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_configuration)
        file_menu.addAction(new_action)
        
        # Open action
        open_action = QAction("Open Configuration", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_configuration)
        file_menu.addAction(open_action)
        
        # Save action
        save_action = QAction("Save Configuration", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Refresh cameras action
        refresh_action = QAction("Refresh Cameras", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_cameras)
        view_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # User Guide action
        guide_action = QAction("User Guide", self)
        guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(guide_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Show ready message
        self.status_bar.showMessage("Ready")
        
        # Setup status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
    
    def update_status(self):
        """Update the status bar with current information."""
        try:
            # Get system status
            robot_status = self.controller.get_robot_status()
            occupied_slots = len(self.controller.get_occupied_slots())
            
            status_text = f"Robot: {'Connected' if robot_status['initialized'] else 'Disconnected'} | "
            status_text += f"Occupied Slots: {occupied_slots} | "
            status_text += f"Lights: {'On' if robot_status['lights_on'] else 'Off'}"
            
            self.status_bar.showMessage(status_text)
        except Exception as e:
            self.status_bar.showMessage(f"Status update error: {str(e)}")
    
    def new_configuration(self):
        """Create a new configuration."""
        reply = QMessageBox.question(
            self, "New Configuration",
            "This will clear all current settings. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear deck
            self.controller.clear_deck()
            
            # Reset settings to defaults
            self.status_bar.showMessage("New configuration created")
    
    def open_configuration(self):
        """Open a configuration file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration", "", "JSON Files (*.json)"
        )
        
        if filename:
            success = self.controller.import_deck_layout(filename)
            if success:
                self.status_bar.showMessage(f"Configuration loaded from {filename}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load configuration file.")
    
    def save_configuration(self):
        """Save the current configuration."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        
        if filename:
            success = self.controller.export_deck_layout(filename)
            if success:
                self.status_bar.showMessage(f"Configuration saved to {filename}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save configuration file.")
    
    def refresh_cameras(self):
        """Refresh the camera list."""
        self.camera_view.refresh_cameras()
        self.status_bar.showMessage("Cameras refreshed")
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, "About Microtissue Manipulator",
            """
            <h3>Microtissue Manipulator Control</h3>
            <p>Version 1.0</p>
            <p>A PyQt6-based GUI application for controlling microtissue manipulation robots.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Robot control and configuration</li>
                <li>Labware deck management</li>
                <li>Multi-camera support with zoom and focus control</li>
                <li>Real-time video feeds</li>
            </ul>
            <p>Built with PyQt6 and OpenCV</p>
            """
        )
    
    def show_user_guide(self):
        """Show the user guide."""
        QMessageBox.information(
            self, "User Guide",
            """
            <h3>User Guide</h3>
            <p><b>Settings Tab:</b></p>
            <ul>
                <li>Initialize the robot connection</li>
                <li>Configure slot offsets and pipette settings</li>
                <li>Control robot lighting and positioning</li>
                <li>Monitor robot status in real-time</li>
            </ul>
            
            <p><b>Labware Declaration Tab:</b></p>
            <ul>
                <li>Click on deck slots to select them</li>
                <li>Choose labware from the available list</li>
                <li>Assign labware to selected slots</li>
                <li>Validate deck layout for completeness</li>
                <li>Export/import deck configurations</li>
            </ul>
            
            <p><b>Camera View Tab:</b></p>
            <ul>
                <li>View all detected cameras</li>
                <li>Double-click to test a camera</li>
                <li>Use mouse wheel to zoom in/out</li>
                <li>Click and drag to pan around</li>
                <li>Adjust focus with the slider</li>
                <li>Double-click video to reset view</li>
            </ul>
            """
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self, "Quit Application",
            "Are you sure you want to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Cleanup resources
            self.controller.cleanup()
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Microtissue Manipulator")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Lab Automation")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
