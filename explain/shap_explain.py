"""
shap_explain.py
Loads saved model + scaler and computes SHAP values for one sample window.
Generates a bar chart of mean absolute SHAP values across features.

Note: Requires the `shap` package. For Keras models, KernelExplainer or DeepExplainer can be used.
DeepExplainer needs tensorflow and may be faster; KernelExplainer is more general but slower.

Run:
python explain/shap_explain.py --data ./data/synthetic_sudden_fault.csv --model ./models/lstm_model.h5 --scaler ./models/scaler.pkl --out ./explain
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

def load_sample_window(csv_path, seq_len=100, start_index=4900):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    features = ["temperature", "vibration", "pressure", "rpm"]
    X = df[features].values
    window = X[start_index:start_index+seq_len]
    return window, features

def main(csv_path, model_path, scaler_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    seq_len = 100
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    window, features = load_sample_window(csv_path, seq_len=seq_len, start_index=4900)
    # scale
    window_scaled = scaler.transform(window)
    # shap expects 2D feature vectors for tabular; we can average over timesteps or explain using KernelExplainer on flattened features.
    # Here we'll flatten (seq_len * n_features) to a 1D array and use KernelExplainer with a small background.
    flattened = window_scaled.flatten()
    # background: take a few random windows from the CSV
    df = pd.read_csv(csv_path)
    X_all = df[features].values
    # build some random background windows
    bg = []
    for i in [100, 500, 1000]:
        bg.append(scaler.transform(X_all[i:i+seq_len]).flatten())
    bg = np.array(bg)
    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape((-1, seq_len, len(features)))) , bg)
    shap_values = explainer.shap_values(flattened, nsamples=100)
    # aggregate shap values per feature by summing absolute values across timesteps
    shap_arr = np.abs(shap_values).reshape(seq_len, len(features))
    mean_per_feature = np.mean(shap_arr, axis=0)
    # plot
    plt.figure()
    plt.bar(features, mean_per_feature)
    plt.xlabel("Feature")
    plt.ylabel("Mean |SHAP value|")
    plt.title("SHAP importance averaged across timesteps (sample window)")
    plt.tight_layout()
    out_path = os.path.join(outdir, "shap_bar.png")
    plt.savefig(out_path)
    print("Saved SHAP bar to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--out", default="./explain")
    args = parser.parse_args()
    main(args.data, args.model, args.scaler, args.out)
