import cv2
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector
from tkinter import *
from Exercise import curl

# Works best on Python 3.12
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
# Start by showing a window to select exercise
select_window = Tk()
select_window.title("Workout Analyser")
select_window.geometry("600x300")
icon = PhotoImage(file="WA.png")
select_window.iconphoto(True, icon)
select_window.configure(bg="#ffeaea")

title = Label(select_window, text="Welcome, select what you'd like to train!",
              font=("Blaka", 25, "bold"), bg="#ffeaea",image=icon, compound="bottom")
title.pack()
curl_button = Button(select_window, text="Bicep Curl")
curl_button.config(command=curl, font=("Ariel", 15, "bold"), activebackground="#ffeaea")
curl_button.pack()

select_window.mainloop()

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
            cv2.circle(image, (x, y), 3, (0, 255, 0), cv2.FILLED)

    cv2.imshow('Real time window', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

