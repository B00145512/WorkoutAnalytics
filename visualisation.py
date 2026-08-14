import altair as alt
import pandas as pd
import os
import glob
import streamlit as st
import numpy as np

SUMMARY_DIR = os.path.join("Datasets", "exercise_hist")
REP_FOLDER = os.path.join("Datasets", "exercise_hist", "reports")

summary_files = glob.glob(os.path.join(SUMMARY_DIR, "*.csv"))
st.set_page_config(page_title="Exercise Summary", layout="wide")
st.title("Exercise Summary")

if not summary_files:
    st.warning("No summary files found in the directory.")
    st.stop()

latest_summary = max(summary_files, key=os.path.getctime)
df = pd.read_csv(latest_summary)
st.caption(f"Loaded summary file: {os.path.basename(latest_summary)}")

# Overview of the workout

st.header("Workout Overview")

total_reps = len(df)

# Change "Prediction" if your summary CSV uses a different column name
correct_reps = (df["Prediction"] == "Perfect").sum()
fatigue_reps = (df["Prediction"] == "Fatigue").sum()
elbow_sway_reps = (df["Prediction"] == "Elbow Sway").sum()

if total_reps > 0:
    average_correctness = df["Perfect %"].mean() * 100
else:
    average_correctness = 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Total Reps", value =total_reps)

with col2:
    st.metric(label="Correct Reps", value=correct_reps)

with col3:
    st.metric(label="Fatigue Detected", value=fatigue_reps)

with col4:
    st.metric(label="Elbow Sway", value=elbow_sway_reps)

with col5:
    st.metric(label="Average Form Correctness",value=f"{average_correctness:.1f}%")

rep_display = df[[
    "Rep", "Prediction", "Confidence",
    "Perfect %", "Fatigue %", "Posture %",
    "Tempo", "ROM"
]].copy()

# Convert probabilities to percentages
percent_cols = ["Confidence", "Perfect %", "Fatigue %", "Posture %"]
rep_display[percent_cols] *= 100

# Rename columns
rep_display.rename(columns={
    "Prediction": "Result",
    "Perfect %": "Correct %",
    "Posture %": "Elbow Sway %"
}, inplace=True)

st.dataframe(
    rep_display,
    width="stretch",
    hide_index=True
)