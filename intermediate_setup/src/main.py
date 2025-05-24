import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from detection.face_detector import FaceDetector
from alignment.face_aligner import FaceAligner
from recognition.feature_extractor import FeatureExtractor
from data.face_database import FaceDatabase

class FaceRecognitionApp:
    def __init__(self):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_aligner = FaceAligner()  # Will use default path
        self.feature_extractor = FeatureExtractor()
        self.face_database = FaceDatabase()
        
        # Initialize GUI
        self.app = QApplication(sys.argv)
        self.main_window = MainWindow()
        
        # Connect GUI signals
        self.main_window.add_face_signal.connect(self.add_face)
        self.main_window.list_faces_signal.connect(self.list_faces)
        
    def add_face(self, name, frame):
        """Add a new face to the database"""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        if not faces:
            return False, "No face detected"
            
        # Get the first face
        face = faces[0]
        
        # Align face
        aligned_face = self.face_aligner.align_face(frame, face)
        
        # Extract features
        features = self.feature_extractor.extract_features(aligned_face)
        if features is None:
            return False, "Failed to extract features"
            
        # Add to database
        person_id = self.face_database.add_person(name)
        self.face_database.add_face(person_id, features, None)  # No image path for now
        
        return True, f"Added face for {name}"
        
    def list_faces(self):
        """List all known faces in the database"""
        return self.face_database.get_all_persons()
        
    def recognize_face(self, frame):
        """Recognize faces in a frame"""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        if not faces:
            return []
            
        results = []
        for face in faces:
            # Align face
            aligned_face = self.face_aligner.align_face(frame, face)
            
            # Extract features
            features = self.feature_extractor.extract_features(aligned_face)
            if features is None:
                continue
                
            # Search database
            match = self.face_database.search_face(features)
            if match:
                person_id, name, similarity = match
                results.append({
                    'face': face,
                    'name': name,
                    'similarity': similarity
                })
                
        return results
        
    def run(self):
        """Run the application"""
        self.main_window.show()
        return self.app.exec_()

def main():
    app = FaceRecognitionApp()
    sys.exit(app.run())

if __name__ == "__main__":
    main() 