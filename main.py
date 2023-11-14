import cv2
import numpy as np

cam = cv2.VideoCapture(0)

people_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while(cam.isOpened()):
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = people_cascade.detectMultiScale(gray)

    for (x, y, w, h) in detect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.destroyWindow()