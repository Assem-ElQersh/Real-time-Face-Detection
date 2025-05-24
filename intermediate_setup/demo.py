import cv2
import numpy as np
from src.detection.face_detector import FaceDetector
from src.alignment.face_aligner import FaceAligner
from src.recognition.feature_extractor import FeatureExtractor
from src.data.face_database import FaceDatabase
from src.utils.model_downloader import ModelDownloader
from src.utils.opencv_setup import OpenCVSetup

class FaceRecognitionDemo:
    def __init__(self):
        # Set up OpenCV files
        opencv_setup = OpenCVSetup()
        if not opencv_setup.setup_opencv_files():
            raise Exception("Failed to set up OpenCV files")
            
        # Download model if needed
        downloader = ModelDownloader()
        if not downloader.download_model():
            raise Exception("Failed to download required model")
            
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_aligner = FaceAligner()  # Will use default path
        self.feature_extractor = FeatureExtractor()
        self.face_database = FaceDatabase()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance optimization variables
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.last_detection = None  # Store last detection results
        
    def add_face(self, name):
        """Add a new face to the database"""
        print(f"Adding face for {name}. Press 'c' to capture when ready...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Draw rectangle around detected face
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Add Face', frame)
            
            # Wait for 'c' key to capture
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if not faces:
                    print("No face detected. Please ensure your face is clearly visible.")
                    continue
                    
                # Get the first face
                face = faces[0]
                
                # Align face
                aligned_face = self.face_aligner.align_face(frame, face)
                if aligned_face is None:
                    print("Failed to align face. Please try again.")
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(aligned_face)
                if features is None:
                    print("Failed to extract features. Please ensure good lighting and a clear view of your face.")
                    continue
                    
                # Add to database
                person_id = self.face_database.add_person(name)
                self.face_database.add_face(person_id, features, None)
                print(f"Successfully added face for {name}")
                break
                
            elif key == ord('q'):
                return False
                
        cv2.destroyWindow('Add Face')
        return True
        
    def draw_detection(self, frame, detection):
        """Draw detection results on the frame"""
        if detection is None:
            return frame
            
        for face_info in detection:
            x, y, w, h, name, similarity, color = face_info
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare text
            text = f"{name} ({similarity:.2f})"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                        (x, y - text_height - 10),
                        (x + text_width, y),
                        color, -1)  # -1 fills the rectangle
            
            # Draw text
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        detection_results = []
        
        # Process each face
        for face in faces:
            # Get face rectangle
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            
            # Align face
            aligned_face = self.face_aligner.align_face(frame, face)
            
            # Extract features
            features = self.feature_extractor.extract_features(aligned_face)
            if features is None:
                continue
                
            # Search database
            match = self.face_database.search_face(features)
            
            # Prepare detection info
            if match:
                person_id, name, similarity = match
                color = (0, 255, 0)  # Green for recognized face
            else:
                name = "Unknown"
                similarity = 0.0
                color = (0, 0, 255)  # Red for unknown face
                
            detection_results.append((x, y, w, h, name, similarity, color))
        
        # Update last detection
        self.last_detection = detection_results
        return frame
        
    def run(self):
        """Run the face recognition demo"""
        print("Face Recognition Demo")
        print("Commands:")
        print("  'a' - Add a new face")
        print("  'q' - Quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process every nth frame for better performance
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames == 0:
                self.process_frame(frame)
            
            # Always draw the last detection results
            frame = self.draw_detection(frame, self.last_detection)
            
            # Show frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                name = input("Enter person's name: ")
                if not self.add_face(name):
                    break
            elif key == ord('q'):
                break
                
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        demo = FaceRecognitionDemo()
        demo.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 