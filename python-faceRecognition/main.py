import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Start webcam
video_capture = cv2.VideoCapture(0)

# Load known face images
# Ensure these files exist in a folder named 'faces'
try:
    uzair1_image = face_recognition.load_image_file("faces/me1.jpg")
    uzair1_encoding = face_recognition.face_encodings(uzair1_image)[0]

    uzair_image = face_recognition.load_image_file("faces/me.jpg")
    uzair_encoding = face_recognition.face_encodings(uzair_image)[0]
except IndexError:
    print("Error: Could not find faces in the provided images.")
    exit()
except FileNotFoundError:
    print("Error: Image files not found in 'faces/' directory.")
    exit()

# List of known face encodings and their names
known_face_encodings = [uzair1_encoding, uzair_encoding]
known_face_names = ["Uzair1", "Uzair"]

# List of students expected today
students = known_face_names.copy()

# Get the current date for the filename
current_date = datetime.now().strftime("%Y-%m-%d")

# Open the file in 'append' mode ('a') so you don't lose previous data
with open(f"{current_date}.csv", "a", newline="") as f:
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
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            name = "Unknown" # Default if no match
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    # If the student is present and not yet marked
                    if name in students:
                        students.remove(name)
                        now_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, now_time])
                        f.flush()  # Ensures data is written to the file immediately

            # Scale back up since the frame was processed at 1/4 size
            top, right, bottom, left = [i * 4 for i in face_location]

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("Attendance System", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()