from deepface import DeepFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2

class FeatureExtractor:
    def __init__(self):
        # Get the Haar Cascade path from environment variable
        self.cascade_path = os.path.join(os.environ.get('OPENCV_DATA_PATH', ''), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found at {self.cascade_path}")
        
        # Set the OpenCV data path for DeepFace
        os.environ['OPENCV_DATA_PATH'] = os.path.dirname(self.cascade_path)
        
    def extract_features(self, face_image):
        """
        Extract facial features from an image
        
        Args:
            face_image: numpy array of the face image in BGR format
            
        Returns:
            numpy array of facial features
        """
        try:
            # Ensure the image is in the correct format
            if isinstance(face_image, np.ndarray):
                # Convert to RGB if needed
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Extract features using DeepFace
            result = DeepFace.represent(
                face_image,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )
            
            if isinstance(result, list):
                result = result[0]
                
            return np.array(result['embedding'])
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def compare_faces(self, face1_features, face2_features):
        """
        Compare two faces using cosine similarity
        
        Args:
            face1_features: features of first face
            face2_features: features of second face
            
        Returns:
            similarity score between 0 and 1
        """
        if face1_features is None or face2_features is None:
            return 0.0
            
        # Reshape for cosine similarity calculation
        face1_features = face1_features.reshape(1, -1)
        face2_features = face2_features.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(face1_features, face2_features)[0][0]
        
        return float(similarity)
    
    def is_match(self, face1_features, face2_features, threshold=0.6):
        """
        Determine if two faces match based on similarity threshold
        
        Args:
            face1_features: features of first face
            face2_features: features of second face
            threshold: similarity threshold (default: 0.6)
            
        Returns:
            boolean indicating if faces match
        """
        similarity = self.compare_faces(face1_features, face2_features)
        return similarity >= threshold 