import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import sys
import dlib

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models', 'shape_predictor_68_face_landmarks.dat'))
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database"""
        db_path = os.path.join('src', 'data', 'known_faces.pkl')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['faces']
                self.known_names = data['names']
            print(f"Loaded {len(self.known_names)} known faces")

    def compare_faces(self, face1, face2):
        """Compare two face images using template matching"""
        # Resize faces to same size
        face1 = cv2.resize(face1, (100, 100))
        face2 = cv2.resize(face2, (100, 100))
        
        # Convert to grayscale
        face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
        face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        
        # Calculate similarity using template matching
        result = cv2.matchTemplate(face1_gray, face2_gray, cv2.TM_CCOEFF_NORMED)
        similarity = np.max(result)
        
        return similarity > 0.6  # Threshold for face matching

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
            for known_face, known_name in zip(self.known_faces, self.known_names):
                if self.compare_faces(face_roi, known_face):
                    name = known_name
                    break
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw name label
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y - 6), font, 0.6, (255, 255, 255), 1)
        
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
            person_dir = os.path.join('known_faces', safe_name)
            
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
            db_path = os.path.join('src', 'data', 'known_faces.pkl')
            with open(db_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        else:
            print("No face or multiple faces detected. Please try again with a single face.")
            return False

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run() 