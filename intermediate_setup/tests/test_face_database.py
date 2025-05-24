import pytest
import numpy as np
import os
from src.data.face_database import FaceDatabase

@pytest.fixture
def face_database():
    # Use a test database file
    db_path = "test_face_database.db"
    db = FaceDatabase(db_path)
    yield db
    # Cleanup after tests
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def sample_features():
    return np.random.rand(512)

def test_database_initialization(face_database):
    """Test if database initializes correctly"""
    assert face_database is not None
    assert os.path.exists(face_database.db_path)

def test_add_person(face_database):
    """Test adding a person to the database"""
    person_id = face_database.add_person("Test Person")
    assert isinstance(person_id, int)
    assert person_id > 0

def test_add_face(face_database, sample_features):
    """Test adding a face to the database"""
    # First add a person
    person_id = face_database.add_person("Test Person")
    
    # Then add a face
    face_id = face_database.add_face(person_id, sample_features, "test_image.jpg")
    assert isinstance(face_id, int)
    assert face_id > 0

def test_get_person_faces(face_database, sample_features):
    """Test retrieving faces for a person"""
    # Add a person and face
    person_id = face_database.add_person("Test Person")
    face_database.add_face(person_id, sample_features, "test_image.jpg")
    
    # Get faces
    faces = face_database.get_person_faces(person_id)
    assert isinstance(faces, list)
    assert len(faces) > 0
    assert len(faces[0]) == 3  # face_id, features, image_path

def test_get_all_persons(face_database):
    """Test retrieving all persons"""
    # Add some test persons
    face_database.add_person("Person 1")
    face_database.add_person("Person 2")
    
    persons = face_database.get_all_persons()
    assert isinstance(persons, list)
    assert len(persons) >= 2

def test_search_face(face_database, sample_features):
    """Test face search functionality"""
    # Add a person and face
    person_id = face_database.add_person("Test Person")
    face_database.add_face(person_id, sample_features, "test_image.jpg")
    
    # Search for the face
    match = face_database.search_face(sample_features)
    assert match is not None
    assert len(match) == 3  # person_id, name, similarity
    assert match[0] == person_id
    assert match[1] == "Test Person"
    assert 0 <= match[2] <= 1 