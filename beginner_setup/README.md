# Beginner Setup - Face Detection and Recognition

A simple implementation of face detection and recognition using OpenCV and face_recognition library.

## Features
- Real-time face detection using OpenCV
- Basic face recognition
- Simple command-line interface
- Easy to understand and modify code

## Project Structure
```
beginner_setup/
├── src/
│   ├── core/             # Core functionality
│   │   ├── face_detector.py
│   │   ├── face_recognizer.py
│   │   └── face_database.py
│   ├── data/            # Data storage
│   │   └── known_faces.pkl
│   └── utils/           # Utility functions
├── known_faces/         # Face image storage
└── requirements.txt
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. The required model file (`shape_predictor_68_face_landmarks.dat`) will be automatically downloaded to the shared `models/` directory at the project root.

## Usage

1. Run the face recognition system:
   ```bash
   python src/core/face_recognition.py
   ```

2. Available commands:
   - Press 'a' to add a new face
   - Press 's' to save the current frame
   - Press 'q' to quit

## How It Works

1. **Face Detection**:
   - Uses OpenCV's Haar Cascade classifier
   - Detects faces in real-time video stream

2. **Face Recognition**:
   - Compares detected faces with known faces
   - Uses simple template matching for recognition

3. **Data Storage**:
   - Stores face data in a pickle file
   - Saves face images in the known_faces directory

## Common Issues

1. **Camera Access**:
   - Ensure your camera is properly connected
   - Check if other applications are using the camera
   - Verify camera permissions

2. **Model Files**:
   - The system will automatically download required models
   - Models are stored in the shared `models/` directory at the project root

3. **Performance**:
   - Reduce video resolution if experiencing lag
   - Ensure good lighting conditions
   - Keep faces clearly visible to the camera

## Next Steps

1. Try the intermediate setup for more advanced features
2. Experiment with different face detection parameters
3. Add your own face recognition algorithms

## Contributing
Feel free to submit issues and enhancement requests!
