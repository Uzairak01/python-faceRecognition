# Python Face Recognition Attendance

A simple Python-based face recognition attendance system using `face_recognition`, `OpenCV`, and `numpy`.

## Features

- Detect faces from webcam video stream
- Recognize known faces using preloaded images in `faces/`
- Mark attendance once per person per session
- Save attendance records to a dated CSV file

## Requirements

- Python 3.8+
- Webcam or camera device

## Dependencies

Install the required packages before running the app:

```bash
pip install cmake
pip install face_recognition
pip install opencv-python
pip install numpy
```

## Setup

1. Place known face images in the `faces/` folder.
2. Update `main.py` to load the correct files and labels if needed.

## Usage

Run the application from the project root:

```bash
python main.py
```

Press `q` to stop the webcam and exit.

## Output

- A CSV attendance file is created using the current date, e.g. `2026-05-23.csv`
- Each recognized person is written once with their name and timestamp

## Notes

- The current implementation loads two sample images: `faces/me1.jpg` and `faces/me.jpg`.
- Adjust the known face image paths and names in `main.py` to add or change recognized users.
- Make sure the webcam is accessible and not used by another application.
