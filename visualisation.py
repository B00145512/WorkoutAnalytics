import altair as alt
import pandas as pd
import os
import glob
import streamlit as st
import numpy as np

SUMMARY_DIR = os.path.join("Datasets", "exercise_hist")
REP_FOLDER = os.path.join("Datasets", "Unlabelled")

# Load summary files and rep files
summary_files = glob.glob(os.path.join(SUMMARY_DIR, "*.csv"))
latest_summary = max(summary_files, key=os.path.getctime)

df = pd.read_csv(latest_summary)
total_reps = len(df)
# Combine all reps into single df

rep_files = sorted(glob.glob(os.path.join(REP_FOLDER, "*.csv")),key=os.path.getctime)
rep_files = rep_files[-total_reps:]

rep_data = []
for rep_file, file in enumerate(rep_files, start=1):
    rep_df = pd.read_csv(file)
    rep_df["Rep"] = rep_file
    rep_data.append(rep_df)

if rep_data:
    all_reps_df = pd.concat(rep_data, ignore_index=True)
else:
    all_reps_df = pd.DataFrame()

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

# Display the rep data in a table
st.dataframe(
    rep_display,
    width="stretch",
    hide_index=True
)
# Visualisation of the workout
st.header("Movement Analysis")
col1, col2 = st.columns(2)

# Writst movement path + height
wrist_df = pd.concat([
    all_reps_df[["Rep", "Timestamp", "Left Wrist X", "Left Wrist Y"]]
        .rename(columns={"Left Wrist X": "X", "Left Wrist Y": "Y"})
        .assign(Wrist="Left"),

    all_reps_df[["Rep", "Timestamp", "Right Wrist X", "Right Wrist Y"]]
        .rename(columns={"Right Wrist X": "X", "Right Wrist Y": "Y"})
        .assign(Wrist="Right")
])

# Elbow movement path + height
elbow_df = pd.concat([
    all_reps_df[["Rep", "Timestamp", "Left Elbow X", "Left Elbow Y"]]
        .rename(columns={"Left Elbow X": "X", "Left Elbow Y": "Y"})
        .assign(Elbow="Left"),

    all_reps_df[["Rep", "Timestamp", "Right Elbow X", "Right Elbow Y"]]
        .rename(columns={"Right Elbow X": "X", "Right Elbow Y": "Y"})
        .assign(Elbow="Right")
])
# Start each rep's time from 0
wrist_df["Time"] = wrist_df.groupby(["Rep", "Wrist"])["Timestamp"].transform(lambda x: x - x.min())
elbow_df["Time"] = elbow_df.groupby(["Rep", "Elbow"])["Timestamp"].transform(lambda x: x)


# Column 1, wrist loaction scatter plot
with col1:
    st.subheader("Wrist Movement Path")

    chart = alt.Chart(wrist_df).mark_circle(
        opacity=0.45
    ).encode(
        x=alt.X("X:Q", title="Horizontal Position"),
        y=alt.Y("Y:Q", title="Vertical Position", scale=alt.Scale(reverse=True)),
        color=alt.Color("Rep:N"),
        shape=alt.Shape("Wrist:N"),
        tooltip=["Rep", "Wrist", "X", "Y"])

    st.altair_chart(chart, width="stretch")

    st.subheader("Elbow Movement Path")

    chart = alt.Chart(elbow_df).mark_circle(
        opacity=0.45
    ).encode(
        x=alt.X("X:Q", title="Horizontal Position"),
        y=alt.Y("Y:Q", title="Vertical Position", scale=alt.Scale(reverse=True)),
        color=alt.Color("Rep:N"),
        shape=alt.Shape("Elbow:N"),
        tooltip=["Rep", "Elbow", "X", "Y"])

    st.altair_chart(chart, width="stretch")

# column 2 wrist height over time
with col2:
    st.subheader("Wrist Height Over Time")

    chart = alt.Chart(wrist_df).mark_line(
        opacity=0.5
    ).encode(
        x=alt.X("Time:Q", title="Time (seconds)"),
        y=alt.Y("Y:Q", title="Wrist Height", scale=alt.Scale(reverse=True)),
        color=alt.Color("Rep:N"),
        strokeDash=alt.StrokeDash("Wrist:N"),
        tooltip=["Rep", "Wrist", "Time", "Y"])

    st.altair_chart(chart, width="stretch")

with col2:
    st.subheader("Elbow Sway Over Time")

    chart = alt.Chart(elbow_df).mark_circle(
        opacity=0.5
    ).encode(
        x=alt.X("Time:Q", title="Time (seconds)"),
        y=alt.Y("Y:Q", title="Elbow Sway", scale=alt.Scale(reverse=True)),
        color=alt.Color("Rep:N"),
        strokeDash=alt.StrokeDash("Elbow:N"),
        tooltip=["Rep", "Elbow", "Time", "Y"])

    st.altair_chart(chart, width="stretch")