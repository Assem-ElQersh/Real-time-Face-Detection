# Real-Time Face Detection and Recognition

This project implements real-time face detection and recognition using OpenCV. It can detect faces in video streams and recognize known faces using template matching.

## Features

- Real-time face detection using OpenCV's Haar Cascade classifier
- Face recognition using template matching
- Support for webcam input
- Known face database management
- Real-time display of recognition results
- Save frames from video stream
- Add new faces in real-time

## Setup

1. Install Python 3.7 or higher
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required model file:
   - Create a `models` directory in the project root:
     ```bash
     mkdir models
     ```
   - Download the face landmark predictor model:
     - Visit [dlib's official website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
     - Extract the downloaded file
     - Place `shape_predictor_68_face_landmarks.dat` in the `models` directory

## Usage

1. Add known faces to the database:
   ```bash
   python add_face.py --name "Person Name" --image path/to/face/image.jpg
   ```
   Or add faces in real-time while running the system (press 'a')

2. Run the face recognition system:
   ```bash
   python face_recognition.py
   ```

## Controls

While running the face recognition system:
- Press 'q' to quit
- Press 's' to save the current frame
- Press 'a' to add a new face to the database

## Image Guidelines

For best recognition results, follow these guidelines when adding faces:
1. Face should be clearly visible and centered
2. Good lighting (not too dark or bright)
3. Clear, non-blurry image
4. Face should take up about 20-30% of the image
5. Neutral expression or slight smile
6. No sunglasses or face coverings

## Directory Structure

- `known_faces/`: Directory containing all known face images
  - Each person has their own subdirectory (e.g., `known_faces/John_Doe/`)
  - Face images are saved with timestamps (e.g., `face_20240321_123456.jpg`)
- `known_faces.pkl`: Database file containing face data and names
- `models/`: Directory containing face detection models
  - `shape_predictor_68_face_landmarks.dat`: Face landmark predictor model (download separately)

## Requirements

- Webcam
- Python 3.7+
- OpenCV
- NumPy

## Troubleshooting

1. If the webcam doesn't open, try changing the camera index in `face_recognition.py`:
   ```python
   video_capture = cv2.VideoCapture(1)  # Try different numbers (0, 1, 2)
   ```

2. If face recognition accuracy is low:
   - Add more faces of the same person from different angles
   - Ensure good lighting conditions
   - Follow the image guidelines above

3. If the system is slow:
   - Close other applications
   - Reduce the video resolution
   - Use a more powerful computer 