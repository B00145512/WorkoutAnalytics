import os.path

import cv2
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector
import csv
import datetime

from Utils import find_angle
from Utils.show_points_and_lines import draw_points, connect_landmarks
from Utils.find_angle import find_angle
from Utils.spine import draw_spine

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
detector = PoseDetector()
cap = cv2.VideoCapture(0)
cur_time = datetime.datetime.now()
filename = os.path.join("exercise_hist","Curl",f"curl_{cur_time:%Y-%m-%d_%H-%M}.csv")

def curl():
    with open(filename, "w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "l_shouler_x", "l_shoulder_y","l_elbow_x", "l_elbow_y", "l_wrist_x", "l_wrist_y", "l_bicep_angle",
                        "r_shouler_x", "r_shoulder_y","r_elbow_x", "r_elbow_y", "r_wrist_x", "r_wrist_y", "r_bicep_angle"])
        frame_count = 1
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
                landmarks = draw_landmarks(image, results)

                cv2.imshow('Real time window', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                if landmarks:
                    writer.writerow([frame_count] + landmarks)

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

        left_angle = find_angle([left_shoulder.x, left_shoulder.y],
                                        [left_elbow.x, left_elbow.y],
                                        [left_wrist.x, left_wrist.y])
        right_angle = find_angle([right_shoulder.x, right_shoulder.y],
                                         [right_elbow.x, right_elbow.y],
                                         [right_wrist.x, right_wrist.y])


        for name, l in pts.items():
            draw_points(image, l, name, w,h)

        connect_landmarks(image, left_shoulder, left_elbow, w, h)
        connect_landmarks(image, right_shoulder, right_elbow, w, h)
        connect_landmarks(image, left_wrist, left_elbow, w, h)
        connect_landmarks(image, right_wrist, right_elbow, w, h)
        connect_landmarks(image, left_shoulder, right_shoulder, w ,h)
        draw_spine(image, results)

        return[
            left_shoulder.x, left_shoulder.y,
            left_elbow.x, left_elbow.y,
            left_wrist.x, left_wrist.y,
            left_angle,
            right_shoulder.x, right_shoulder.y,
            right_elbow.x, right_elbow.y,
            right_wrist.x, right_wrist.y,
            right_angle
        ]
    return None
