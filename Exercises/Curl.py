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
import time

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
        writer.writerow(["Frame", "Timestamp",
            "Left Shoulder X","Left Shoulder Y","Left Elbow X","Left Elbow Y","Left Wrist X","Left Wrist Y",
            "Right Shoulder X", "Right Shoulder Y","Right Elbow X", "Right Elbow Y","Right Wrist X","Right Wrist Y",
            "Left Elbow Angle","Right Elbow Angle","Left Shoulder Angle","Right Shoulder Angle","Torso Angle",
            "Left Angular Velocity","Right Angular Velocity",
            "Left Wrist Velocity","Right Wrist Velocity",
            "Left Elbow Velocity","Right Elbow Velocity"])
        # Initialise values
        start_time = time.time()
        frame_count = 1
        prev_left_angle = None
        prev_right_angle = None
        prev_left_wrist = None
        prev_right_wrist = None
        prev_time = None
        prev_left_elbow = None
        prev_right_elbow = None

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
                #landmarks = draw_landmarks(image, results)
                timestamp = time.time() - start_time

                features = draw_landmarks(image,
                                          results,
                                          timestamp,
                                          prev_time,
                                          prev_left_angle,
                                          prev_right_angle,
                                          prev_left_wrist,
                                          prev_right_wrist,
                                          prev_left_elbow,
                                          prev_right_elbow)

                cv2.imshow('Real time window', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                #if landmarks:
                #    writer.writerow([frame_count] + landmarks)
                if features:
                    writer.writerow(
                        [frame_count, timestamp] +
                        list(features.values())
                    )

                    prev_time = timestamp

                    prev_left_angle = features["left_elbow_angle"]
                    prev_right_angle = features["right_elbow_angle"]

                    prev_left_wrist = (
                        features["left_wrist_x"],
                        features["left_wrist_y"]
                    )

                    prev_right_wrist = (
                        features["right_wrist_x"],
                        features["right_wrist_y"]
                    )

                    prev_left_elbow = (
                        features["left_elbow_x"],
                        features["left_elbow_y"]
                    )

                    prev_right_elbow = (
                        features["right_elbow_x"],
                        features["right_elbow_y"]
                    )

                    frame_count += 1

            cap.release()
            cv2.destroyAllWindows()


def draw_landmarks(image, results, timestamp, prev_time, prev_left_angle, prev_right_angle, prev_left_wrist, prev_right_wrist, prev_left_elbow, prev_right_elbow):

    if not results.pose_landmarks:
        return None

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
    left_knee = lm[25]
    
    # Draw skeleton
    pts = {
        "L_shoulder": left_shoulder,
        "R_shoulder": right_shoulder,
        "L_elbow": left_elbow,
        "R_elbow": right_elbow,
        "L_wrist": left_wrist,
        "R_wrist": right_wrist,
        "L_hip": left_hip,
        "R_hip": right_hip
    }

    for name, p in pts.items():
        draw_points(image, p, name, w, h)

    connect_landmarks(image, left_shoulder, left_elbow, w, h)
    connect_landmarks(image, left_elbow, left_wrist, w, h)
    connect_landmarks(image, right_shoulder, right_elbow, w, h)
    connect_landmarks(image, right_elbow, right_wrist, w, h)
    connect_landmarks(image, left_shoulder, right_shoulder, w, h)

    draw_spine(image, results)


    left_angle = find_angle([left_shoulder.x, left_shoulder.y],
                            [left_elbow.x, left_elbow.y],
                            [left_wrist.x, left_wrist.y])
    right_angle = find_angle([right_shoulder.x, right_shoulder.y],
                            [right_elbow.x, right_elbow.y],
                            [right_wrist.x, right_wrist.y])
    left_shoulder_angle = find_angle([left_hip.x, left_hip.y],
                                     [left_shoulder.x, left_shoulder.y],
                                     [left_elbow.x, left_elbow.y])
    right_shoulder_angle = find_angle([right_hip.x, right_hip.y],
                                      [right_shoulder.x, right_shoulder.y],
                                      [right_elbow.x, right_elbow.y])
    torso_angle = find_angle([left_shoulder.x, left_shoulder.y],
                             [left_hip.x, left_hip.y],
                             [left_knee.x, left_knee.y])

    # Find Velocity of current joints based on previous frame
    dt = 0
    left_ang_vel = 0
    right_ang_vel = 0
    left_elbow_vel = 0
    right_elbow_vel = 0
    left_wrist_vel = 0
    right_wrist_vel = 0

    if prev_time is not None:
        dt = timestamp - prev_time

    if dt > 0:
        if prev_left_angle is not None:
            left_ang_vel = (left_angle - prev_left_angle) / dt

        if prev_right_angle is not None:
            right_ang_vel = (right_angle - prev_right_angle) / dt

    if dt > 0:
        if prev_left_wrist is not None:
            dx = left_wrist.x - prev_left_wrist[0]
            dy = left_wrist.y - prev_left_wrist[1]
            left_wrist_vel = np.sqrt(dx**2 + dy**2) / dt

        if prev_right_wrist is not None:
            dx = right_wrist.x - prev_right_wrist[0]
            dy = right_wrist.y - prev_right_wrist[1]
            right_wrist_vel = np.sqrt(dx**2 + dy**2) / dt

    if dt > 0:
        if prev_left_elbow is not None:
            dx = left_elbow.x - prev_left_elbow[0]
            dy = left_elbow.y - prev_left_elbow[1]
            left_elbow_vel = np.sqrt(dx**2 + dy**2) / dt

        if prev_right_elbow is not None:
            dx = right_elbow.x - prev_right_elbow[0]
            dy = right_elbow.y - prev_right_elbow[1]
            right_elbow_vel = np.sqrt(dx**2 + dy**2) / dt

    return {

        "left_shoulder_x": left_shoulder.x, "left_shoulder_y": left_shoulder.y,
        "left_elbow_x": left_elbow.x, "left_elbow_y": left_elbow.y,
        "left_wrist_x": left_wrist.x, "left_wrist_y": left_wrist.y,
        "right_shoulder_x": right_shoulder.x, "right_shoulder_y": right_shoulder.y,
        "right_elbow_x": right_elbow.x, "right_elbow_y": right_elbow.y,
        "right_wrist_x": right_wrist.x, "right_wrist_y": right_wrist.y,
        "left_elbow_angle": left_angle, "right_elbow_angle": right_angle,
        "left_shoulder_angle": left_shoulder_angle, "right_shoulder_angle": right_shoulder_angle,
        "torso_angle": torso_angle,
        "left_ang_velocity": left_ang_vel, "right_ang_velocity": right_ang_vel,
        "left_wrist_velocity": left_wrist_vel, "right_wrist_velocity": right_wrist_vel,
        "left_elbow_velocity": left_elbow_vel,"right_elbow_velocity": right_elbow_vel
    }