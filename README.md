# Real-time Face Detection Project

This project provides two different implementations of real-time face detection and recognition:
1. A beginner-friendly setup with simpler code and basic features
2. An intermediate setup with advanced features and optimizations

## Project Structure

```
Real-time-Face-Detection/
├── models/                     # Shared model files directory
│   └── shape_predictor_68_face_landmarks.dat
├── beginner_setup/            # Beginner-friendly implementation
│   ├── src/                   # Source code
│   │   ├── core/             # Core functionality
│   │   ├── data/             # Data storage
│   │   └── utils/            # Utility functions
│   ├── known_faces/          # Face image storage
│   ├── requirements.txt      # Dependencies
│   └── README.md            # Setup instructions
├── intermediate_setup/        # Advanced implementation
│   ├── src/                  # Source code
│   │   ├── detection/        # Face detection
│   │   ├── alignment/        # Face alignment
│   │   ├── recognition/      # Feature extraction
│   │   ├── data/            # Database management
│   │   └── utils/           # Utility functions
│   ├── tests/               # Test files
│   ├── docs/                # Documentation
│   ├── data/                # Data storage
│   ├── models/              # Model files
│   ├── demo.py             # Demo application
│   ├── download_model.py   # Model downloader
│   ├── face_database.db    # SQLite database
│   ├── requirements.txt    # Dependencies
│   └── README.md          # Setup instructions
├── DOCUMENTATION.md          # Detailed project documentation
├── TECHNIQUES_COMPARISON.md  # Technical comparison
├── project_proposal.tex     # Project proposal
└── LICENSE                  # Project license
```

## Quick Start

1. Choose your preferred setup:
   - For beginners: Follow instructions in `beginner_setup/README.md`
   - For intermediate users: Follow instructions in `intermediate_setup/README.md`

2. The face detection model will be automatically downloaded to the shared `models/` directory when you first run either setup.

## Features

### Beginner Setup
- Simple and easy to understand code
- Basic face detection and recognition
- Good for learning and understanding the concepts
- Suitable for beginners in computer vision

### Intermediate Setup
- Advanced face detection and recognition
- Performance optimizations
- Additional features like:
  - Face tracking
  - Multiple face detection
  - Real-time performance improvements
- Suitable for more experienced developers

## Requirements

- Python 3.7+
- OpenCV
- dlib
- face_recognition
- numpy

Detailed requirements and installation instructions are provided in each setup's README file.

## Documentation

- `DOCUMENTATION.md`: Contains detailed documentation about the project
- Each setup has its own README with specific instructions
- Code is well-commented for better understanding

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License

This project is open source and available under the MIT License. 