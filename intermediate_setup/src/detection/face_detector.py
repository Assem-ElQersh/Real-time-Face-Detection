import dlib
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """Initialize the face detector using dlib's HOG detector"""
        self.detector = dlib.get_frontal_face_detector()
        
    def detect_faces(self, image):
        """
        Detect faces in an image using dlib's HOG detector
        
        Args:
            image: numpy array of the image in BGR format
            
        Returns:
            list of dlib rectangles containing face locations
        """
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces with increased upsampling for better detection
        faces = self.detector(rgb_image, 2)  # 2 means upsampling twice
        
        # Convert dlib rectangles to list
        return list(faces)
    
    def get_face_rectangles(self, image):
        """
        Get face rectangles in OpenCV format
        
        Args:
            image: numpy array of the image in BGR format
            
        Returns:
            list of (x, y, w, h) tuples for each face
        """
        faces = self.detect_faces(image)
        rectangles = []
        
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            rectangles.append((x, y, w, h))
            
        return rectangles
    
    def extract_faces(self, image):
        """
        Extract face regions from the image
        
        Args:
            image: numpy array of the image in BGR format
            
        Returns:
            list of face images and their locations
        """
        faces = self.detect_faces(image)
        face_images = []
        locations = []
        
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            face_img = image[y:y+h, x:x+w]
            face_images.append(face_img)
            locations.append((x, y, w, h))
            
        return face_images, locations 