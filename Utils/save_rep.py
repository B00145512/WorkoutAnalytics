import csv
import os
from datetime import datetime

def save_rep(current_rep, rep_number, rep_tempo, rep_min_angle, rep_max_angle, rep_rom):
    folder_path = os.path.join("Datasets", "Unlabelled")
    os.makedirs(folder_path, exist_ok=True)

    filename = os.path.join(folder_path, f"rep_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{rep_number:04d}.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame","Timestamp", "Rep Duration",
            "Left Shoulder X","Left Shoulder Y",
            "Left Elbow X","Left Elbow Y",
            "Left Wrist X","Left Wrist Y",
            "Right Shoulder X","Right Shoulder Y",
            "Right Elbow X","Right Elbow Y",
            "Right Wrist X","Right Wrist Y",
            "Left Elbow Angle","Right Elbow Angle",
            "Left Shoulder Angle","Right Shoulder Angle",
            "Torso Angle",
            "Left Angular Velocity","Right Angular Velocity",
            "Left Wrist Velocity","Right Wrist Velocity",
            "Left Elbow Velocity","Right Elbow Velocity",
            "Left Elbow Drift","Right Elbow Drift",
            "Rep Tempo",
            "Rep Min Angle",
            "Rep Max Angle",
            "Rep ROM"
        ])
        for frame in current_rep:
            rep_duration = frame["timestamp"] - current_rep[0]["timestamp"]
            writer.writerow([
                frame["frame"],
                frame["timestamp"],
                rep_duration,
                frame["left_shoulder_x"],
                frame["left_shoulder_y"],
                frame["left_elbow_x"],
                frame["left_elbow_y"],
                frame["left_wrist_x"],
                frame["left_wrist_y"],
                frame["right_shoulder_x"],
                frame["right_shoulder_y"],
                frame["right_elbow_x"],
                frame["right_elbow_y"],
                frame["right_wrist_x"],
                frame["right_wrist_y"],
                frame["left_elbow_angle"],
                frame["right_elbow_angle"],
                frame["left_shoulder_angle"],
                frame["right_shoulder_angle"],
                frame["torso_angle"],
                frame["left_ang_velocity"],
                frame["right_ang_velocity"],
                frame["left_wrist_velocity"],
                frame["right_wrist_velocity"],
                frame["left_elbow_velocity"],
                frame["right_elbow_velocity"],
                frame["left_elbow_drift"],
                frame["right_elbow_drift"],
                rep_tempo,
                rep_min_angle,
                rep_max_angle,
                rep_rom
            ])