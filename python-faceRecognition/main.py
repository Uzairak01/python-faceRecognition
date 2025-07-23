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

# Start webcam
video_capture = cv2.VideoCapture(0)

# Load known face images
uzair1_image = face_recognition.load_image_file("faces/me1.jpg")
uzair1_encoding = face_recognition.face_encodings(uzair1_image)[0]

uzair_image = face_recognition.load_image_file("faces/me.jpg")
uzair_encoding = face_recognition.face_encodings(uzair_image)[0]

# List of known face encodings and their names
known_face_encoding = [uzair1_encoding, uzair_encoding]
known_face_name = ["Uzair1", "Uzair"]

# List of students expected today
students = known_face_name.copy()

# Initialize variables
face_locations = []
face_encodings = []

# Get the current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create a CSV file to save attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    # Read video frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = ""  # Default name

        if matches[best_match_index]:
            name = known_face_name[best_match_index]

            # If the student is present and not yet marked
            if name in students:
                students.remove(name)  # Prevent re-logging
                now_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, now_time])  # Write to CSV

        # Draw rectangle and name on original frame
        top, right, bottom, left = face_location
        # Scale back up since the frame we processed was scaled to 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Attendance", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()
