# Intermediate Setup - Advanced Face Detection and Recognition

An advanced implementation of face detection and recognition using dlib, DeepFace, and SQLite database.

## Features
- Real-time face detection using dlib's HOG detector
- Face alignment using 68-point facial landmarks
- Feature extraction using DeepFace
- SQLite database for persistent storage
- Real-time webcam interface with GUI
- Performance optimizations

## Project Structure
```
intermediate_setup/
├── src/
│   ├── detection/        # Face detection
│   │   └── face_detector.py
│   ├── alignment/        # Face alignment
│   │   └── face_aligner.py
│   ├── recognition/      # Feature extraction
│   │   └── feature_extractor.py
│   ├── data/            # Database management
│   │   └── face_database.py
│   └── utils/           # Utility functions
│       ├── model_downloader.py
│       └── opencv_setup.py
├── face_database.db     # SQLite database
├── demo.py             # Real-time demo
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

1. Run the real-time demo:
   ```bash
   python demo.py
   ```

2. Available commands:
   - Press 'a' to add a new face
   - Press 'q' to quit

## How It Works

1. **Face Detection**:
   - Uses dlib's HOG detector for accurate face detection
   - Handles multiple faces in the frame
   - Optimized for real-time performance

2. **Face Alignment**:
   - Uses 68-point facial landmarks
   - Aligns faces to a standard position
   - Improves recognition accuracy

3. **Feature Extraction**:
   - Uses DeepFace for robust feature extraction
   - Generates face embeddings
   - Handles different face angles

4. **Database Management**:
   - SQLite database for persistent storage
   - Efficient face search and matching
   - Supports multiple faces per person

## Performance Optimizations

1. **Frame Processing**:
   - Reduced webcam resolution (640x480)
   - Process every 3rd frame
   - Persistent display of detection results

2. **Database**:
   - Indexed searches
   - Efficient face matching
   - Batch operations

3. **Memory Management**:
   - Efficient model loading
   - Proper resource cleanup
   - Optimized data structures

## Common Issues

1. **Model Files**:
   - The system will automatically download required models
   - Models are stored in the shared `models/` directory at the project root
   - Check internet connection for automatic downloads

2. **Performance**:
   - Ensure good lighting conditions
   - Keep faces clearly visible
   - Adjust frame processing rate if needed

3. **Database**:
   - Database is automatically created
   - Backup database before major changes
   - Check disk space for database growth

## Next Steps

1. Implement GPU acceleration
2. Add support for video file processing
3. Create a graphical user interface
4. Add face tracking capabilities

## Contributing
Feel free to submit issues and enhancement requests! 