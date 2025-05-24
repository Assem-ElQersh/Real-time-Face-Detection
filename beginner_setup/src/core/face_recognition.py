import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import sys
import dlib
import face_recognition

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Fix the path to the shape predictor model
        model_path = os.path.join(project_root, 'models', 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please download the model file from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract it and place it in the 'models' directory")
            sys.exit(1)
            
        self.predictor = dlib.shape_predictor(model_path)
        
        # Ensure data directory exists
        self.data_dir = os.path.join(project_root, 'beginner_setup', 'src', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Ensure known_faces directory exists
        self.known_faces_dir = os.path.join(project_root, 'beginner_setup', 'known_faces')
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        self.db_path = os.path.join(self.data_dir, 'known_faces.pkl')
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['faces']
                self.known_names = data['names']
            print(f"Loaded {len(self.known_names)} known faces")

    def compare_faces(self, face1, face2):
        """Compare two face images and return confidence score"""
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
            face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            face1_encoding = face_recognition.face_encodings(face1_rgb)[0]
            face2_encoding = face_recognition.face_encodings(face2_rgb)[0]
            
            # Calculate face distance (lower is more similar)
            face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
            
            # Convert distance to confidence score (0-100%)
            confidence = (1 - face_distance) * 100
            
            return confidence > 60, confidence  # Return both match result and confidence score
        except Exception as e:
            print(f"Error in face comparison: {str(e)}")
            return False, 0.0

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Try to recognize the face
            name = "Unknown"
            confidence = 0.0
            for known_face, known_name in zip(self.known_faces, self.known_names):
                is_match, conf = self.compare_faces(face_roi, known_face)
                if is_match and conf > confidence:
                    name = known_name
                    confidence = conf
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw name label with confidence
            label = f"{name} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (x + 6, y - 6), font, 0.6, (255, 255, 255), 1)
        
        return frame

    def run(self):
        """Run the face recognition system"""
        # Try different camera indices
        for camera_index in [0, 1]:
            video_capture = cv2.VideoCapture(camera_index)
            if video_capture.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
            else:
                print(f"Failed to open camera {camera_index}")
        
        if not video_capture.isOpened():
            print("Error: Could not open any video capture device")
            print("Please check if:")
            print("1. Your camera is properly connected")
            print("2. No other application is using the camera")
            print("3. You have necessary permissions to access the camera")
            return
        
        print("Face recognition system started. Press 'q' to quit, 's' to save frame, 'a' to add face")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Video', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                print(f"Saved frame as capture_{timestamp}.jpg")
            elif key == ord('a'):
                # Add new face
                name = input("Enter name for the new face: ")
                if self.add_face(frame, name):
                    print(f"Added new face: {name}")
        
        # Clean up
        video_capture.release()
        cv2.destroyAllWindows()

    def add_face(self, frame, name):
        """Add a new face to the database"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 1:
            # Get the face rectangle
            (x, y, w, h) = faces[0]
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Create directory for this person
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            person_dir = os.path.join(self.known_faces_dir, safe_name)
            
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            
            # Save the face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"face_{timestamp}.jpg"
            face_path = os.path.join(person_dir, face_filename)
            cv2.imwrite(face_path, face_roi)
            
            # Add to known faces
            self.known_faces.append(face_roi)
            self.known_names.append(name)
            
            # Save to database
            data = {
                'faces': self.known_faces,
                'names': self.known_names
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        else:
            print("No face or multiple faces detected. Please try again with a single face.")
            return False

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run() 