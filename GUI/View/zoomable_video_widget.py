"""
Zoomable video widget for camera displays.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent


class ZoomableVideoWidget(QLabel):
    """A widget that displays video with zoom and pan capabilities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        
        # Pan variables
        self.pan_start_point = QPoint()
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        
        # Original frame storage
        self.original_frame = None
        self.scaled_pixmap = None
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Default text
        self.setText("No video feed")
    
    def set_frame(self, frame):
        """Set the current frame to display."""
        if frame is not None:
            self.original_frame = frame.copy()
            self.update_display()
    
    def update_display(self):
        """Update the display with current zoom and pan settings."""
        if self.original_frame is None:
            return
        
        # Convert OpenCV frame to QPixmap
        height, width, channel = self.original_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.original_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Create pixmap and apply zoom
        original_pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap based on zoom factor
        if self.zoom_factor != 1.0:
            scaled_size = original_pixmap.size() * self.zoom_factor
            self.scaled_pixmap = original_pixmap.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            self.scaled_pixmap = original_pixmap
        
        # Apply pan offset
        if self.pan_offset != QPoint(0, 0) or self.zoom_factor != 1.0:
            # Create a new pixmap with the widget size
            result_pixmap = QPixmap(self.size())
            result_pixmap.fill(Qt.GlobalColor.black)
            
            # Calculate position to draw the scaled pixmap
            draw_x = (self.width() - self.scaled_pixmap.width()) // 2 + self.pan_offset.x()
            draw_y = (self.height() - self.scaled_pixmap.height()) // 2 + self.pan_offset.y()
            
            # Draw the scaled pixmap onto the result pixmap
            painter = QPainter(result_pixmap)
            painter.drawPixmap(draw_x, draw_y, self.scaled_pixmap)
            painter.end()
            
            self.setPixmap(result_pixmap)
        else:
            # No zoom or pan, just scale to fit widget
            fitted_pixmap = self.scaled_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(fitted_pixmap)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming."""
        if self.original_frame is None:
            return
        
        # Get the mouse position relative to the widget
        mouse_pos = event.position().toPoint()
        
        # Calculate zoom change
        zoom_in = event.angleDelta().y() > 0
        old_zoom = self.zoom_factor
        
        if zoom_in:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
        else:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
        
        # Calculate new pan offset to zoom towards mouse position
        if self.zoom_factor != old_zoom:
            # Calculate the position in the original image
            widget_center = QPoint(self.width() // 2, self.height() // 2)
            mouse_offset = mouse_pos - widget_center
            
            # Adjust pan offset to zoom towards mouse position
            zoom_ratio = self.zoom_factor / old_zoom
            self.pan_offset = QPoint(
                int(self.pan_offset.x() * zoom_ratio - mouse_offset.x() * (zoom_ratio - 1)),
                int(self.pan_offset.y() * zoom_ratio - mouse_offset.y() * (zoom_ratio - 1))
            )
        
        self.update_display()
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for panning."""
        if event.button() == Qt.MouseButton.LeftButton and self.original_frame is not None:
            self.is_panning = True
            self.pan_start_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for panning."""
        if self.is_panning and self.original_frame is not None:
            # Calculate pan delta
            current_point = event.position().toPoint()
            delta = current_point - self.pan_start_point
            
            # Update pan offset
            self.pan_offset += delta
            self.pan_start_point = current_point
            
            self.update_display()
        event.accept()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click to reset zoom and pan."""
        if self.original_frame is not None:
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.update_display()
        event.accept()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        if self.original_frame is not None:
            self.update_display()
    
    def reset_view(self):
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        if self.original_frame is not None:
            self.update_display()
    
    def clear_frame(self):
        """Clear the current frame."""
        self.original_frame = None
        self.scaled_pixmap = None
        self.setText("No video feed")
        self.clear()


class VideoDisplayWidget(QWidget):
    """Widget that contains a zoomable video display with controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Create scroll area for video
        self.scroll_area = QScrollArea()
        self.video_widget = ZoomableVideoWidget()
        self.scroll_area.setWidget(self.video_widget)
        self.scroll_area.setWidgetResizable(True)
        
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
    
    def set_frame(self, frame):
        """Set the current frame to display."""
        self.video_widget.set_frame(frame)
    
    def reset_view(self):
        """Reset zoom and pan."""
        self.video_widget.reset_view()
    
    def clear_frame(self):
        """Clear the current frame."""
        self.video_widget.clear_frame()
