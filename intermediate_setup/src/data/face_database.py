import sqlite3
import numpy as np
import os
import pickle
from datetime import datetime

class FaceDatabase:
    def __init__(self, db_path="face_database.db"):
        """
        Initialize the face database
        
        Args:
            db_path: path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create faces table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            features BLOB,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_person(self, name):
        """
        Add a new person to the database
        
        Args:
            name: name of the person
            
        Returns:
            person_id: ID of the newly created person
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO persons (name) VALUES (?)', (name,))
        person_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return person_id
        
    def add_face(self, person_id, features, image_path):
        """
        Add a face to the database
        
        Args:
            person_id: ID of the person
            features: facial features as numpy array
            image_path: path to the face image
            
        Returns:
            face_id: ID of the newly created face
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert features to bytes
        features_bytes = pickle.dumps(features)
        
        cursor.execute('''
        INSERT INTO faces (person_id, features, image_path)
        VALUES (?, ?, ?)
        ''', (person_id, features_bytes, image_path))
        
        face_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return face_id
        
    def get_person_faces(self, person_id):
        """
        Get all faces for a person
        
        Args:
            person_id: ID of the person
            
        Returns:
            list of (face_id, features, image_path) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, features, image_path
        FROM faces
        WHERE person_id = ?
        ''', (person_id,))
        
        faces = []
        for row in cursor.fetchall():
            face_id, features_bytes, image_path = row
            features = pickle.loads(features_bytes)
            faces.append((face_id, features, image_path))
            
        conn.close()
        
        return faces
        
    def get_all_persons(self):
        """
        Get all persons in the database
        
        Returns:
            list of (person_id, name) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name FROM persons')
        persons = cursor.fetchall()
        
        conn.close()
        
        return persons
        
    def search_face(self, features, threshold=0.6):
        """
        Search for a matching face in the database
        
        Args:
            features: facial features to search for
            threshold: similarity threshold
            
        Returns:
            (person_id, name, similarity) tuple if match found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, features, person_id FROM faces')
        best_match = None
        best_similarity = 0
        
        for row in cursor.fetchall():
            face_id, features_bytes, person_id = row
            stored_features = pickle.loads(features_bytes)
            
            # Calculate similarity
            similarity = np.dot(features, stored_features) / (
                np.linalg.norm(features) * np.linalg.norm(stored_features)
            )
            
            # Ensure similarity is between 0 and 1
            similarity = float(min(max(similarity, 0.0), 1.0))
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                cursor.execute('SELECT name FROM persons WHERE id = ?', (person_id,))
                name = cursor.fetchone()[0]
                best_match = (person_id, name, similarity)
        
        conn.close()
        
        return best_match 