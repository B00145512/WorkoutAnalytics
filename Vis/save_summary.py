import csv
import os
from datetime import datetime

SUMMARY_FILE = os.path.join("Datasets", "exercise_hist", f"summary_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv")

def init_summary():
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            "Rep",
            "Prediction",
            "Confidence",
            "Perfect %",
            "Fatigue %",
            "Posture %",
            "Tempo",
            "ROM"
        ])

def append_summary(rep, pred, conf, probs, tempo, rom, labels):
    with open(SUMMARY_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            rep,
            labels[pred],
            round(conf, 3),
            round(probs[0], 3),
            round(probs[1], 3),
            round(probs[2], 3),
            round(tempo, 3),
            round(rom, 3)
        ])
