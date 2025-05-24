import pytest
import cv2
import numpy as np
import os
from src.alignment.face_aligner import FaceAligner

MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"

@pytest.fixture
def face_aligner():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return FaceAligner(MODEL_PATH)

@pytest.fixture
def sample_image():
    # Create a simple test image
    image = np.zeros((150, 150, 3), dtype=np.uint8)
    return image

@pytest.fixture
def sample_face():
    # Create a dlib rectangle for testing
    from dlib import rectangle
    return rectangle(30, 30, 70, 70)

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_face_aligner_initialization(face_aligner):
    """Test if face aligner initializes correctly"""
    assert face_aligner is not None
    assert face_aligner.predictor is not None
    assert face_aligner.desired_size == (150, 150)

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_get_landmarks(face_aligner, sample_image, sample_face):
    """Test facial landmark detection"""
    landmarks = face_aligner.get_landmarks(sample_image, sample_face)
    assert isinstance(landmarks, np.ndarray)
    assert landmarks.shape[1] == 2  # x, y coordinates

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_align_face(face_aligner, sample_image, sample_face):
    """Test face alignment"""
    aligned_face = face_aligner.align_face(sample_image, sample_face)
    assert isinstance(aligned_face, np.ndarray)
    assert aligned_face.shape[:2] == face_aligner.desired_size

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_align_faces(face_aligner, sample_image):
    """Test alignment of multiple faces"""
    # Create multiple face rectangles
    from dlib import rectangle
    faces = [
        rectangle(30, 30, 70, 70),
        rectangle(80, 80, 120, 120)
    ]
    
    aligned_faces = face_aligner.align_faces(sample_image, faces)
    assert isinstance(aligned_faces, list)
    assert len(aligned_faces) == len(faces)
    for face in aligned_faces:
        assert face.shape[:2] == face_aligner.desired_size 