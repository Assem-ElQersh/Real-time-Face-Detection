import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QLineEdit,
                            QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        layout.addWidget(self.video_label)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Add person controls
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        control_layout.addWidget(self.name_input)
        
        add_face_btn = QPushButton("Add Face")
        add_face_btn.clicked.connect(self.add_face)
        control_layout.addWidget(add_face_btn)
        
        # Add camera controls
        start_camera_btn = QPushButton("Start Camera")
        start_camera_btn.clicked.connect(self.start_camera)
        control_layout.addWidget(start_camera_btn)
        
        stop_camera_btn = QPushButton("Stop Camera")
        stop_camera_btn.clicked.connect(self.stop_camera)
        control_layout.addWidget(stop_camera_btn)
        
        # Add database controls
        list_faces_btn = QPushButton("List Known Faces")
        list_faces_btn.clicked.connect(self.list_faces)
        control_layout.addWidget(list_faces_btn)
        
        # Add spacing
        control_layout.addStretch()
        
        # Add control panel to main layout
        layout.addWidget(control_panel)
        
        # Initialize camera
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        """Start the camera feed"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera")
                return
            self.timer.start(30)  # Update every 30ms
            
    def stop_camera(self):
        """Stop the camera feed"""
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.video_label.clear()
            
    def update_frame(self):
        """Update the video frame"""
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Display image
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio))
                
    def add_face(self):
        """Add a new face to the database"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name")
            return
            
        if self.camera is None:
            QMessageBox.warning(self, "Warning", "Please start the camera first")
            return
            
        # TODO: Implement face detection and database storage
        QMessageBox.information(self, "Success", f"Added face for {name}")
        
    def list_faces(self):
        """List all known faces in the database"""
        # TODO: Implement face listing
        QMessageBox.information(self, "Known Faces", "List of known faces will be shown here")
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 