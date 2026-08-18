# LSTM Training
import glob
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# CONFIG
# ---------------------------
DATASET_ROOT = "Datasets"

CLASS_FOLDER_TO_LABEL = {
    "0": 0,   # correct
    "1": 1,   # fatigue
    "2": 2    # elbow sway
}

MAX_FRAMES = 151
NUM_CLASSES = 3
HIDDEN_SIZE = 64
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 20
# 25 input features
# 151 sequence length
# 1 LSTM layer
# ADAM optimizer
# CrossEntropyLoss
# 80/20 split

FRAME_FEATURES = [
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
    "Left Elbow Drift","Right Elbow Drift"
]

# LOAD + PAD SEQUENCES

def load_sequence(filepath, label):
    df = pd.read_csv(filepath)

    df = df[FRAME_FEATURES]

    # truncate
    if len(df) > MAX_FRAMES:
        df = df.iloc[:MAX_FRAMES]

    # pad
    elif len(df) < MAX_FRAMES:
        last_row = df.iloc[-1]
        pad_count = MAX_FRAMES - len(df)
        padding = pd.DataFrame([last_row] * pad_count)
        df = pd.concat([df, padding], ignore_index=True)

    return df.values, label


def build_sequence_dataset(root):
    X_all = []
    y_all = []

    for folder, label in CLASS_FOLDER_TO_LABEL.items():
        files = glob.glob(os.path.join(root, folder, "*.csv"))

        for f in files:
            X, y = load_sequence(f, label)
            X_all.append(X)
            y_all.append(y)

    return np.array(X_all), np.array(y_all)


# Model
class CurlLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super().__init__()

        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# Training

if __name__ == "__main__":

    print("Loading dataset...")
    X, y = build_sequence_dataset(DATASET_ROOT)

    print("Dataset shape:", X.shape)  # (N, 151, features)


    # normalization

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model setup

    model = CurlLSTM(input_size=len(FRAME_FEATURES))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    print("Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1)

        accuracy = (preds == y_test_tensor).float().mean()

    print("Test Accuracy:", accuracy.item())

    # SAVE MODEl
    torch.save(model.state_dict(), "curl_lstm.pth")
    np.save("norm_mean.npy", mean)
    np.save("norm_std.npy", std)

    print("Model + normalization saved.")