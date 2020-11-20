import cv2
import time
import os
from flask_pymongo import PyMongo
from app import mongo

cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)  # 0 for web camera live stream

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'save_face/')  #folder path
if not os.path.isdir(target):
    os.mkdir(target)

def camera_stream():
     # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_face = frame[y:y+h, x:x+w]
        file_name = f"{time.time()}.jpg"
        cv2.imwrite(f"{target}{file_name}",crop_face)
        mongo.db.images.insert_one({"image":file_name})


    # Display the resulting frame in browser
    return cv2.imencode('.jpg', frame)[1].tobytes()