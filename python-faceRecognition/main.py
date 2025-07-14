#pip install cmake
#pip install face_recognition
#pip install opencv-python
#pip install numpy

# | Import             | Role                                  |
# | ------------------ | ------------------------------------- |
# | `face_recognition` | Detect and recognize faces            |
# | `cv2`              | Access webcam, show video, draw boxes |
# | `numpy`            | Handle images and math data           |
# | `csv`              | Save names & timestamps to file       |
# | `datetime`         | Get current date & time for logs      |

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#load images of known faces
uzair1_image = face_recognition.load_image_file("faces/me1.jpg")
uzair1_encoding = face_recognition.face_encodings(uzair1_image)[0]

uzair_image = face_recognition.load_image_file("faces/me.jpg")
uzair_encoding = face_recognition.face_encodings(uzair_image)[0]

known_face_encoding = [uzair1_encoding, uzair_encoding]
known_face_name = ["Uzair1", "Uzair"]

#list of expected students
students = known_face_name.copy()

face_locations = []
face_encodings = []

#get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #recognize face
    
