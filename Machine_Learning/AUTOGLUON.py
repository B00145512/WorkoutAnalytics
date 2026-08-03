#ML
from autogluon.tabular import TabularPredictor
import pandas as pd

stage1_model = TabularPredictor.load("ag_hier_stage1")
stage2_model = TabularPredictor.load("ag_hier_stage2")

def predict_rep_autogluon(current_rep,
                rep_tempo,
                rep_min_angle,
                rep_max_angle,
                range_of_motion):

    # Create one-row dataframe from the repetition
    df = aggregate_rep_features(
        current_rep,
        rep_tempo,
        rep_min_angle,
        rep_max_angle,
        range_of_motion
    )

    # ---------- Stage 1 ----------
    stage1_prediction = stage1_model.predict(df)[0]
    stage1_prob = stage1_model.predict_proba(df).max(axis=1).iloc[0]

    if stage1_prediction == 0:

        return {
            "class":0,
            "label":"Good Form",
            "confidence":stage1_prob
        }

    # ---------- Stage 2 ----------

    stage2_prediction = stage2_model.predict(df)[0]
    stage2_prob = stage2_model.predict_proba(df).max(axis=1).iloc[0]

    labels = {
        1:"Class 1",
        2:"Class 2",
        3:"Class 3"
    }

    return {
        "class":stage2_prediction,
        "label":labels[stage2_prediction],
        "confidence":stage2_prob
    }

def aggregate_rep_features(current_rep, rep_tempo, rep_min_angle, rep_max_angle, range_of_motion):
    df = pd.DataFrame(current_rep)

    # maps training column names -> live dict keys captured in draw_landmarks()
    FEATURE_KEY_MAP = {
        "Left Elbow Angle":       "left_elbow_angle",
        "Right Elbow Angle":      "right_elbow_angle",
        "Left Shoulder Angle":    "left_shoulder_angle",
        "Right Shoulder Angle":   "right_shoulder_angle",
        "Torso Angle":            "torso_angle",
        "Left Angular Velocity":  "left_ang_velocity",
        "Right Angular Velocity": "right_ang_velocity",
        "Left Wrist Velocity":    "left_wrist_velocity",
        "Right Wrist Velocity":   "right_wrist_velocity",
        "Left Elbow Velocity":    "left_elbow_velocity",
        "Right Elbow Velocity":   "right_elbow_velocity",
        "Left Elbow Drift":       "left_elbow_drift",
        "Right Elbow Drift":      "right_elbow_drift",
    }

    row = {}
    for training_name, live_key in FEATURE_KEY_MAP.items():
        row[f"{training_name}_mean"] = df[live_key].mean()
        row[f"{training_name}_std"]  = df[live_key].std()
        row[f"{training_name}_min"]  = df[live_key].min()
        row[f"{training_name}_max"]  = df[live_key].max()

    row["Rep Tempo"]     = rep_tempo
    row["Rep Min Angle"] = rep_min_angle
    row["Rep Max Angle"] = rep_max_angle
    row["Rep ROM"]       = range_of_motion
    row["Rep Duration"]  = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

    return pd.DataFrame([row])