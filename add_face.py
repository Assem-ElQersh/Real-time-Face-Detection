import cv2
import numpy as np
import argparse
import pickle
import os
from datetime import datetime

def load_face_cascade():
    """Load the face detection cascade classifier"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def validate_face_image(image, face_rect):
    """Validate face image quality"""
    x, y, w, h = face_rect
    
    # Calculate face size
    face_height = h
    face_width = w
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate face size relative to image
    face_size_ratio = (face_height * face_width) / (img_height * img_width)
    
    # Check if face is too small (less than 10% of image)
    if face_size_ratio < 0.1:
        return False, "Face is too small in the image. Please get closer to the camera."
    
    # Check if face is too large (more than 50% of image)
    if face_size_ratio > 0.5:
        return False, "Face is too large in the image. Please step back from the camera."
    
    # Check image brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 40:
        return False, "Image is too dark. Please ensure better lighting."
    if brightness > 220:
        return False, "Image is too bright. Please reduce lighting."
    
    # Check image blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, "Image is too blurry. Please hold the camera steady."
    
    return True, "Image quality is good"

def create_face_directory(name):
    """Create a directory for a person's faces"""
    # Create a sanitized directory name
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    # Create directory path
    person_dir = os.path.join('known_faces', safe_name)
    
    # Create directory if it doesn't exist
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    return person_dir

def add_face_from_image(image_path, name):
    """Add a face from an image file to the database"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = load_face_cascade()
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 1:
        # Get the face rectangle
        (x, y, w, h) = faces[0]
        
        # Validate face image quality
        is_valid, message = validate_face_image(image, (x, y, w, h))
        if not is_valid:
            print(f"Warning: {message}")
            print("Recommended image guidelines:")
            print("1. Face should be clearly visible and centered")
            print("2. Good lighting (not too dark or bright)")
            print("3. Clear, non-blurry image")
            print("4. Face should take up about 20-30% of the image")
            print("5. Neutral expression or slight smile")
            print("6. No sunglasses or face coverings")
            proceed = input("Do you want to proceed anyway? (y/n): ")
            if proceed.lower() != 'y':
                return False
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Create directory for this person
        person_dir = create_face_directory(name)
        
        # Save the face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"face_{timestamp}.jpg"
        face_path = os.path.join(person_dir, face_filename)
        cv2.imwrite(face_path, face_roi)
        
        # Load existing database if it exists
        known_faces = []
        known_names = []
        if os.path.exists('known_faces.pkl'):
            with open('known_faces.pkl', 'rb') as f:
                data = pickle.load(f)
                known_faces = data['faces']
                known_names = data['names']
        
        # Add new face
        known_faces.append(face_roi)
        known_names.append(name)
        
        # Save updated database
        data = {
            'faces': known_faces,
            'names': known_names
        }
        with open('known_faces.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Successfully added face: {name}")
        print(f"Face image saved to: {face_path}")
        return True
    else:
        print(f"Error: Found {len(faces)} faces in the image. Please provide an image with exactly one face.")
        return False

def list_known_faces():
    """List all known faces in the database"""
    if not os.path.exists('known_faces'):
        print("No known faces directory found.")
        return
    
    print("\nKnown Faces Database:")
    print("-" * 50)
    
    # List all person directories
    for person_dir in os.listdir('known_faces'):
        person_path = os.path.join('known_faces', person_dir)
        if os.path.isdir(person_path):
            # Count face images
            face_count = len([f for f in os.listdir(person_path) if f.startswith('face_')])
            print(f"Person: {person_dir.replace('_', ' ')}")
            print(f"Number of face images: {face_count}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Manage face recognition database')
    parser.add_argument('--name', help='Name of the person')
    parser.add_argument('--image', help='Path to the image file')
    parser.add_argument('--list', action='store_true', help='List all known faces')
    
    args = parser.parse_args()
    
    if args.list:
        list_known_faces()
    elif args.name and args.image:
        add_face_from_image(args.image, args.name)
    else:
        print("Please provide either --list to view known faces or both --name and --image to add a new face.")
        print("\nExample usage:")
        print("  Add a face: python add_face.py --name \"John Doe\" --image path/to/face.jpg")
        print("  List faces: python add_face.py --list")
        print("\nImage Guidelines:")
        print("1. Face should be clearly visible and centered")
        print("2. Good lighting (not too dark or bright)")
        print("3. Clear, non-blurry image")
        print("4. Face should take up about 20-30% of the image")
        print("5. Neutral expression or slight smile")
        print("6. No sunglasses or face coverings")

if __name__ == "__main__":
    main() 