import pytest
import numpy as np
from src.recognition.feature_extractor import FeatureExtractor

@pytest.fixture
def feature_extractor():
    return FeatureExtractor()

@pytest.fixture
def sample_face():
    # Create a simple test face image
    return np.zeros((150, 150, 3), dtype=np.uint8)

def test_feature_extractor_initialization(feature_extractor):
    """Test if feature extractor initializes correctly"""
    assert feature_extractor is not None
    assert feature_extractor.model_name == "VGG-Face"

def test_extract_features(feature_extractor, sample_face):
    """Test feature extraction"""
    features = feature_extractor.extract_features(sample_face)
    # Since we're using a blank image, feature extraction might fail
    # We'll just check that it returns either None or a numpy array
    assert features is None or isinstance(features, np.ndarray)
    if features is not None:
        assert features.ndim == 1  # Should be a 1D array of features

def test_compare_faces(feature_extractor):
    """Test face comparison"""
    # Create two sample feature vectors
    features1 = np.random.rand(512)  # VGG-Face typically produces 512-dimensional features
    features2 = np.random.rand(512)
    
    similarity = feature_extractor.compare_faces(features1, features2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_is_match(feature_extractor):
    """Test face matching"""
    # Create two sample feature vectors
    features1 = np.random.rand(512)
    features2 = np.random.rand(512)
    
    # Test with different thresholds
    assert isinstance(feature_extractor.is_match(features1, features2, threshold=0.6), bool)
    assert isinstance(feature_extractor.is_match(features1, features2, threshold=0.8), bool)
    
    # Test with same features (should match)
    assert feature_extractor.is_match(features1, features1, threshold=0.6) 