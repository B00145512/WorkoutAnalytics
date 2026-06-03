import cv2 
import numpy as np
import mediapipe as mp


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()