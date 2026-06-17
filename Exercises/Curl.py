import cv2
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector

from Utils import find_angle
from Utils.show_points_and_lines import draw_points, connect_landmarks
from Utils.find_angle import find_angle
from Utils.spine import draw_spine

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
detector = PoseDetector()
cap = cv2.VideoCapture(0)

def curl():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            #Recolour to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            #Make detection
            results = pose.process(image)
            #Recolour back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_landmarks(image, results)

            cv2.imshow('Real time window', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def draw_landmarks(image, results):
    if results.pose_landmarks:
        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        left_elbow = lm[13]
        right_elbow = lm[14]
        left_wrist = lm[15]
        right_wrist = lm[16]
        left_hip = lm[23]
        right_hip = lm[24]

        pts = {'L_shoulder': left_shoulder,
                'R_shoulder': right_shoulder,
                'L_elbow': left_elbow,
                'R_elbow': right_elbow,
                'L_wrist': left_wrist,
                'R_wrist': right_wrist,
                'L_hip': left_hip,
                'R_hip': right_hip}

        print("Left Bicep: ",find_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y]))
        print("Right Bicep: ",find_angle([right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y], [right_wrist.x, right_wrist.y]))
        for name, l in pts.items():
            draw_points(image, l, name, w,h)

        connect_landmarks(image, left_shoulder, left_elbow, w, h)
        connect_landmarks(image, right_shoulder, right_elbow, w, h)
        connect_landmarks(image, left_wrist, left_elbow, w, h)
        connect_landmarks(image, right_wrist, right_elbow, w, h)
        connect_landmarks(image, left_shoulder, right_shoulder, w ,h)
        draw_spine(image, results)
