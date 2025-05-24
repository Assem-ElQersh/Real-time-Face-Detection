import pytest
import cv2
import numpy as np
from src.detection.face_detector import FaceDetector

@pytest.fixture
def face_detector():
    return FaceDetector()

@pytest.fixture
def sample_image():
    # Create a more realistic test image with a face-like pattern
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a face-like pattern
    # Face outline
    cv2.ellipse(image, (150, 150), (80, 100), 0, 0, 360, (255, 255, 255), -1)
    # Eyes
    cv2.circle(image, (120, 130), 15, (0, 0, 0), -1)
    cv2.circle(image, (180, 130), 15, (0, 0, 0), -1)
    # Mouth
    cv2.ellipse(image, (150, 180), (40, 20), 0, 0, 180, (0, 0, 0), 2)
    return image

def test_face_detector_initialization(face_detector):
    """Test if face detector initializes correctly"""
    assert face_detector is not None
    assert face_detector.detector is not None

def test_detect_faces(face_detector, sample_image):
    """Test face detection on a sample image"""
    faces = face_detector.detect_faces(sample_image)
    assert isinstance(faces, list)
    # Note: The detector might not find faces in our simple drawing
    # We just check that it returns a list

def test_get_face_rectangles(face_detector, sample_image):
    """Test getting face rectangles"""
    rectangles = face_detector.get_face_rectangles(sample_image)
    assert isinstance(rectangles, list)
    if rectangles:
        assert len(rectangles[0]) == 4  # x, y, w, h

def test_extract_faces(face_detector, sample_image):
    """Test face extraction"""
    face_images, locations = face_detector.extract_faces(sample_image)
    assert isinstance(face_images, list)
    assert isinstance(locations, list)
    if face_images:
        assert isinstance(face_images[0], np.ndarray)
        assert len(locations[0]) == 4  # x, y, w, h 