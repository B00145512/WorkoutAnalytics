import glob # Finds CSV files
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

DATASET_ROOT = "Datasets"
LABEL_COL = "Class"
REP_ID_COL = "Rep ID"
CLASS_FOLDER_TO_LABEL = {
    "0": 0,   # perfect posture
    "1": 1,   # fatigued
    "2": 2   # elbow sway
    #"3": 3,   # bad back posture
}
MAX_FRAMES = 150
# per-frame columns
FRAME_FEATURES = [
    "Left Elbow Angle", "Right Elbow Angle",
    "Left Shoulder Angle", "Right Shoulder Angle", "Torso Angle",
    "Left Angular Velocity", "Right Angular Velocity",
    "Left Wrist Velocity", "Right Wrist Velocity",
    "Left Elbow Velocity", "Right Elbow Velocity",
    "Left Elbow Drift", "Right Elbow Drift",
]
REP_LEVEL_FEATURES_CONSTANT = ["Rep Tempo", "Rep Min Angle", "Rep Max Angle", "Rep ROM"]

# loads CSV into a single row of features
def load_rep_file(filepath: str, label: int) -> dict:
    df = pd.read_csv(filepath)

    row = {"Rep ID": os.path.splitext(os.path.basename(filepath))[0], LABEL_COL: label}

    for col in FRAME_FEATURES:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_max"] = df[col].max()

    for col in REP_LEVEL_FEATURES_CONSTANT:
        row[col] = df[col].iloc[0]

    row["Rep Duration"] = df["Rep Duration"].max()  # running total -> final value = total duration

    return row

# one row per rep across the whole dataset
def build_rep_level_dataset(dataset_root: str = DATASET_ROOT) -> pd.DataFrame:
    rows = []
    for folder_name, label in CLASS_FOLDER_TO_LABEL.items():
        pattern = os.path.join(dataset_root, folder_name, "*.csv")
        filepaths = glob.glob(pattern)
        if not filepaths:
            print(f"Warning: no CSV files found in {pattern}")
        for filepath in filepaths:
            rows.append(load_rep_file(filepath, label))

    rep_df = pd.DataFrame(rows)
    return rep_df

# Hierarchical 2-stage model, predict perfect vs not, then which problem if not perfect
def train_hierarchical(rep_df: pd.DataFrame, save_prefix="ag_hier"):
    df = rep_df.copy()
    df["stage1_label"] = (df[LABEL_COL] != 0).astype(int)  # 0=good, 1=bad

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[LABEL_COL], random_state=42
    )

    # checks if its 0 or (1,2,3)
    stage1_predictor = TabularPredictor(
        label="stage1_label",
        path=f"{save_prefix}_stage1",
        eval_metric="balanced_accuracy",
    ).fit(
        TabularDataset(train_df.drop(columns=[REP_ID_COL, LABEL_COL])),
        presets="best_quality"
    )

    # Checks which flaw it has 
    bad_train = train_df[train_df["stage1_label"] == 1].drop(
        columns=[REP_ID_COL, "stage1_label"]
    )
    stage2_predictor = TabularPredictor(
        label=LABEL_COL,
        path=f"{save_prefix}_stage2",
        eval_metric="balanced_accuracy",
    ).fit(
        TabularDataset(bad_train),
        presets="best_quality"
    )

    # Combined evaluation on held-out test set
    test_features = test_df.drop(columns=[REP_ID_COL, LABEL_COL, "stage1_label"])
    stage1_preds = stage1_predictor.predict(test_features)

    final_preds = []
    for i, pred in zip(test_features.index, stage1_preds):
        if pred == 0:
            final_preds.append(0)  # predicted "perfect"
        else:
            row = test_features.loc[[i]]
            stage2_pred = stage2_predictor.predict(row).iloc[0]
            final_preds.append(stage2_pred)

    from sklearn.metrics import balanced_accuracy_score, classification_report
    acc = balanced_accuracy_score(test_df[LABEL_COL], final_preds)
    print("Hierarchical combined balanced accuracy:", acc)
    print(classification_report(test_df[LABEL_COL], final_preds))

    return stage1_predictor, stage2_predictor

#run
if __name__ == "__main__":
    rep_df = build_rep_level_dataset(DATASET_ROOT)
    print(f"Built {len(rep_df)} rep-level rows from {DATASET_ROOT}")
    print(rep_df[LABEL_COL].value_counts())
    print("\n=== Training hierarchical 2-stage model ===")
    train_hierarchical(rep_df)