import dlib
import cv2
import numpy as np
import os

class FaceAligner:
    def __init__(self, predictor_path=None):
        """
        Initialize the face aligner with dlib's facial landmark predictor
        
        Args:
            predictor_path: path to the dlib facial landmark predictor model
        """
        if predictor_path is None:
            predictor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models', 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(predictor_path)
        self.desired_size = (150, 150)  # Standard size for aligned faces
        
    def get_landmarks(self, image, face):
        """
        Get facial landmarks for a face
        
        Args:
            image: numpy array of the image in BGR format
            face: dlib rectangle containing face location
            
        Returns:
            numpy array of 68 facial landmarks
        """
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get landmarks
        shape = self.predictor(rgb_image, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        return landmarks
    
    def align_face(self, image, face):
        """
        Align a face using facial landmarks
        
        Args:
            image: numpy array of the image in BGR format
            face: dlib rectangle containing face location
            
        Returns:
            aligned face image
        """
        # Get facial landmarks
        landmarks = self.get_landmarks(image, face)
        
        # Get left and right eye centers
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate desired right eye position
        desired_left_eye = (0.35, 0.35)
        desired_right_eye = (1 - desired_left_eye[0], desired_left_eye[1])
        
        # Calculate scale
        desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * self.desired_size[0]
        current_dist = np.sqrt((dX ** 2) + (dY ** 2))
        scale = desired_dist / current_dist
        
        # Calculate center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update translation
        tX = self.desired_size[0] * 0.5
        tY = self.desired_size[1] * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply transformation
        aligned_face = cv2.warpAffine(image, M, self.desired_size,
                                     flags=cv2.INTER_CUBIC)
        
        return aligned_face
    
    def align_faces(self, image, faces):
        """
        Align multiple faces in an image
        
        Args:
            image: numpy array of the image in BGR format
            faces: list of dlib rectangles containing face locations
            
        Returns:
            list of aligned face images
        """
        aligned_faces = []
        for face in faces:
            aligned_face = self.align_face(image, face)
            aligned_faces.append(aligned_face)
            
        return aligned_faces 