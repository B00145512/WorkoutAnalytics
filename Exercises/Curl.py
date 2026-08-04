import os.path
import cv2
import numpy as np
import mediapipe as mp
import datetime
import time
import torch

from Utils.show_points_and_lines import draw_points, connect_landmarks
from Utils.find_angle import find_angle
from Utils.spine import draw_spine
from Utils.FindDistance import find_velocity, find_velocity_nodt
from Utils.save_rep import save_rep

from Machine_Learning.LSTM import predict_rep_lstm, fix_sequence_length
from Machine_Learning.ML_MULTIROW import CurlLSTM

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

FRAME_FEATURES = [
    "left_shoulder_x","left_shoulder_y",
    "left_elbow_x","left_elbow_y",
    "left_wrist_x","left_wrist_y",
    "right_shoulder_x","right_shoulder_y",
    "right_elbow_x","right_elbow_y",
    "right_wrist_x","right_wrist_y",
    "left_elbow_angle","right_elbow_angle",
    "left_shoulder_angle","right_shoulder_angle",
    "torso_angle",
    "left_ang_velocity","right_ang_velocity",
    "left_wrist_velocity","right_wrist_velocity",
    "left_elbow_velocity","right_elbow_velocity",
    "left_elbow_drift","right_elbow_drift"
]
# Load model
model = CurlLSTM(input_size=len(FRAME_FEATURES))
model.load_state_dict(torch.load("curl_lstm.pth"))
model.eval()

mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

labels = {
    0: "Perfect",
    1: "Fatigue",
    2: "Elbow Sway"
}
feedback_messages = {
    0: "Good Form",
    1: "Fatigue Detected, try to rest or lower weight",
    2: "Elbow Sway Detected, keep your elbows close to your body"
}

def curl():

    start_time          = time.time()
    frame_count         = 0
    prev_left_angle     = None
    prev_right_angle    = None
    prev_left_wrist     = None
    prev_right_wrist    = None
    prev_left_elbow     = None
    prev_right_elbow    = None
    prev_time           = None
    stage               = "down"
    rep_count           = 0
    current_rep         = []
    rep_start_time      = None
    rep_min_angle       = 180
    rep_max_angle       = 0
    curl_top            = 40
    curl_bottom         = 150
    baseline_spine_dist = None

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
            #Recolour back
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            timestamp = time.time() - start_time

            features, baseline_spine_dist = draw_landmarks(image,
                                                           results,
                                                           timestamp,
                                                           prev_time,
                                                           prev_left_angle,
                                                           prev_right_angle,
                                                           prev_left_wrist, 
                                                           prev_right_wrist,
                                                           prev_left_elbow, 
                                                           prev_right_elbow,
                                                           baseline_spine_dist)

            #If no features are found, continue without writing to CSV
            if features is None:
                cv2.imshow("Real time window", image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue

            # Update previous values
            prev_time = timestamp
            prev_left_angle = features["left_elbow_angle"]
            prev_right_angle = features["right_elbow_angle"]

            prev_left_wrist = (features["left_wrist_x"], features["left_wrist_y"])
            prev_right_wrist = (features["right_wrist_x"], features["right_wrist_y"])
            prev_left_elbow = (features["left_elbow_x"], features["left_elbow_y"])
            prev_right_elbow = (features["right_elbow_x"], features["right_elbow_y"])

            frame_count += 1

            arm_angle = features["left_elbow_angle"]

            # Start rep
            if arm_angle < curl_top and stage == "down":

                stage = "up"
                rep_start_time = time.time()

                rep_min_angle = arm_angle
                rep_max_angle = arm_angle

                current_rep = []

            # Record every frame while curling
            if stage == "up":

                current_rep.append({
                    "timestamp": timestamp,
                    **features
            })
                rep_min_angle = min(rep_min_angle, arm_angle)
                rep_max_angle = max(rep_max_angle, arm_angle)

            # Finish rep
            if arm_angle > curl_bottom and stage == "up":

                stage = "down"
                rep_count += 1

                rep_tempo = time.time() - rep_start_time
                rom = rep_max_angle - rep_min_angle

                save_rep(
                    current_rep,
                    rep_count,
                    rep_tempo,
                    rep_min_angle,
                    rep_max_angle, rom
                )

                # LSTM Prediction

                sequence = np.array([
                    [frame[f] for f in FRAME_FEATURES]
                    for frame in current_rep
                ])

                sequence = fix_sequence_length(sequence)

                #print("Sequence shape:", sequence.shape)  # should be (151,25)

                prediction = predict_rep_lstm(sequence, model, mean, std)
                pred = prediction.argmax()  # Get the index of the highest probability
                conf = prediction[pred]  # Get the confidence of the predicted class

                print("\n------------Curl Information------------")
                print(f"Rep: {rep_count}")
                print(f"Time: {rep_tempo:.2f} seconds")
                print(f"Range of Motion: {rom:.2f} degrees")
                print(f"Min Angle: {rep_min_angle:.2f} degrees", f"Max Angle: {rep_max_angle:.2f} degrees")
                for i in range(len(prediction)):
                    print(f"{labels[i]}: {prediction[i]*100:.1f}%")
                print("--------------Curl Feedback--------------\n")
                print(f"Detected: {labels[pred]} with {conf*100:.1f}%")
                print(f"Feedback: {feedback_messages[pred]}")
                print("-----------------------------------------\n")

                current_rep.clear()
                rep_min_angle = 180
                rep_max_angle = 0
                rep_start_time = None

            cv2.imshow("Real time window", image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

def draw_landmarks(image, results, timestamp,
                    prev_time, prev_left_angle, prev_right_angle, prev_left_wrist, prev_right_wrist, prev_left_elbow, prev_right_elbow, baseline_spine_dist):

    if not results.pose_landmarks:
        return None, baseline_spine_dist

    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark

    ls = lm[11]; rs = lm[12]
    le = lm[13]; re = lm[14]
    lw = lm[15]; rw = lm[16]
    lh = lm[23]; rh = lm[24]
    lk = lm[25]

    draw_points(image, ls, "L_shoulder", w, h)
    draw_points(image, rs, "R_shoulder", w, h)

    connect_landmarks(image, ls, le, w, h)
    connect_landmarks(image, le, lw, w, h)
    connect_landmarks(image, rs, re, w, h)
    connect_landmarks(image, re, rw, w, h)

    # spine ONLY for UI
    spine_data = draw_spine(image, results, baseline_spine_dist)
    if spine_data:
        baseline_spine_dist = spine_data[0]

    # angles
    left_angle  = find_angle([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
    right_angle = find_angle([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])

    left_shoulder_angle     = find_angle([lh.x, lh.y], [ls.x, ls.y], [le.x, le.y])
    right_shoulder_angle    = find_angle([rh.x, rh.y], [rs.x, rs.y], [re.x, re.y])

    torso_angle = find_angle([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])

    dt = timestamp - prev_time if prev_time else 0

    left_ang_vel  = (left_angle - prev_left_angle) / dt if dt and prev_left_angle else 0
    right_ang_vel = (right_angle - prev_right_angle) / dt if dt and prev_right_angle else 0

    # find velocity of joints
    left_wrist_vel      = find_velocity((lw.x, lw.y), prev_left_wrist, dt)
    right_wrist_vel     = find_velocity((rw.x, rw.y), prev_right_wrist, dt)
    left_elbow_vel      = find_velocity((le.x, le.y), prev_left_elbow, dt)
    right_elbow_vel     = find_velocity((re.x, re.y), prev_right_elbow, dt)

    # find elbow drift
    left_elbow_drift    = find_velocity_nodt((le.x, le.y), prev_left_elbow)
    right_elbow_drift   = find_velocity_nodt((re.x, re.y), prev_right_elbow)

    features = {
        "left_shoulder_x":      ls.x,               "left_shoulder_y":      ls.y,
        "left_elbow_x":         le.x,               "left_elbow_y":         le.y,
        "left_wrist_x":         lw.x,               "left_wrist_y":         lw.y,
        "right_shoulder_x":     rs.x,               "right_shoulder_y":     rs.y,
        "right_elbow_x":        re.x,               "right_elbow_y":        re.y,
        "right_wrist_x":        rw.x,               "right_wrist_y":        rw.y,
        "left_elbow_angle":     left_angle,         "right_elbow_angle":    right_angle,
        "left_shoulder_angle":  left_shoulder_angle,"right_shoulder_angle":right_shoulder_angle,
        "torso_angle":          torso_angle,
        "left_ang_velocity":    left_ang_vel,       "right_ang_velocity":   right_ang_vel,
        "left_wrist_velocity":  left_wrist_vel,     "right_wrist_velocity": right_wrist_vel,
        "left_elbow_velocity":  left_elbow_vel,     "right_elbow_velocity": right_elbow_vel,
        "left_elbow_drift":     left_elbow_drift,   "right_elbow_drift":    right_elbow_drift
    }
    return features, baseline_spine_dist