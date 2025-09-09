"""
train_lstm.py
Trains a simple LSTM for binary failure prediction using generated CSVs.
Saves model to models/lstm_model.h5 and scaler to models/scaler.pkl

Run:
python models/train_lstm.py --data ./data/synthetic_slow_degradation.csv --out models
"""
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Use TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_and_preprocess(csv_path, seq_len=100, features=None):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if features is None:
        features = ["temperature", "vibration", "pressure", "rpm"]
    X = df[features].values
    y = df["fault"].values
    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # build sliding windows
    sequences = []
    labels = []
    for i in range(len(Xs) - seq_len):
        sequences.append(Xs[i:i+seq_len])
        # label: if any fault in future window (simple) — take label at end of window
        labels.append(y[i+seq_len])
    X_arr = np.array(sequences)
    y_arr = np.array(labels)
    return X_arr, y_arr, scaler

def build_model(seq_len, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "Precision", "Recall"])
    return model

def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    seq_len = 100
    X, y, scaler = load_and_preprocess(csv_path, seq_len=seq_len)
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = build_model(seq_len, X.shape[2])
    # Callbacks
    chk = ModelCheckpoint(os.path.join(outdir, "lstm_model.h5"), save_best_only=True, monitor="val_loss")
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, callbacks=[chk, es])
    # Save scaler
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    # Save history as CSV
    pd.DataFrame(history.history).to_csv(os.path.join(outdir, "training_history.csv"), index=False)
    print("Training complete. Model and scaler saved to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="./models")
    args = parser.parse_args()
    main(args.data, args.out)
