import csv
import os
from datetime import datetime

SUMMARY_FILE = os.path.join("exercise_hist", f"summary_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv")

def init_summary(SUMMARY_FILE):
    os.makedirs("exercise_hist", exist_ok=True)

    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow([
                "Rep",
                "Prediction",
                "Confidence",
                "Perfect %",
                "Fatigue %",
                "Postuire %",
                "Tempo", 
                "ROM"])

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
