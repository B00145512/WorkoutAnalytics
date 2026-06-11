import cv2
import cvzone
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone.PlotModule as LivePlot
from cvzone.PoseModule import PoseDetector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
idList = [0, 7, 8, 11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
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

def tricep_extension():
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

def squat():
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
        #print("Shoulder: ",left_shoulder)
        #print("Wrist: ",left_wrist)
        #print("Elbow: ",left_elbow)
        pts = {'L_shoulder': left_shoulder,
                'R_shoulder': right_shoulder,
                'L_elbow': left_elbow,
                'R_elbow': right_elbow,
                'L_wrist': left_wrist,
                'R_wrist': right_wrist}
        print(find_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y]))
        for name, l in pts.items():
            x_px, y_px = int(l.x * w), int(l.y * h)
            cv2.circle(image, (x_px, y_px), 6, (0, 255, 0), -1)
            cv2.putText(image, name, (x_px + 6, y_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
def find_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle