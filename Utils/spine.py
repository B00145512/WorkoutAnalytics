import cv2
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector
from pyparsing import results

import Utils.find_angle
import draw_landmarks
from Utils.find_angle import find_angle


def draw_spine(image, results):
    if results.pose_landmarks:
        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        left_hip = lm[23]
        right_hip = lm[24]

        pts = {'left_shoulder': left_shoulder,
               'right_shoulder': right_shoulder,
               'left_hip': left_hip,
               'right_hip': right_hip}
    for name, l in pts.items():
        draw_landmarks(image, l)