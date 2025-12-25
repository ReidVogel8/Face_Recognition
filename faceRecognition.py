import cv2 as cv
import numpy as np
import pickle
import tkinter as tk
from trainer import recognizer

# Load labels
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

# Load face cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#keep camera running until broken with q
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    #use recognizer.predict to figure out ids and confidence using trained images
    for (x, y, w, h) in faces:
        #takes just the box around the face to get confidence and id
        box = gray[y:y+h, x:x+w]
        ids, confidence = recognizer.predict(box)

        #70 percent confidence it will print out the name else will show unknown
        if confidence < 55:
            name = labels[ids]
            display = name
        else:
            display = "Unknown"

        #put name and box on screen
        cv.putText(frame, display, (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Camera", frame)

    #quit video capture with q
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
