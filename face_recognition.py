import cv2
import numpy as np
import os
import pickle
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database"""
        if os.path.exists('known_faces.pkl'):
            with open('known_faces.pkl', 'rb') as f:
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
        # Initialize video capture
        video_capture = cv2.VideoCapture(1)
        
        if not video_capture.isOpened():
            print("Error: Could not open video capture device")
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
            with open('known_faces.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            return True
        else:
            print("No face or multiple faces detected. Please try again with a single face.")
            return False

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run() 