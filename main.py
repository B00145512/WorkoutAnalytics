import cv2 
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector

"""
0 - nose
1 - left eye (inner)
2 - left eye
3 - left eye (outer)
4 - right eye (inner)
5 - right eye
6 - right eye (outer)
7 - left ear
8 - right ear
9 - mouth (left)
10 - mouth (right)
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index
"""
idList = [0, 7, 8, 11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
detector = PoseDetector()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = detector.findPose(image)
    lmList, bboxInfo = detector.findPosition(image, draw=False)

    for lm in lmList:
        if lm[0] in idList:
            x, y = lm[1], lm[2]
            cv2.circle(image, (x, y), 5, (0, 255, 0), cv2.FILLED)

    cv2.imshow('Real time window', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
