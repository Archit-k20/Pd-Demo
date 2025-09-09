"""
app.py - Streamlit dashboard
Shows:
- A timeseries viewer (choose dataset and window)
- Model prediction (calls local FastAPI or runs local model predict function)
- SHAP bar plot for the selected window (if shap output exists)

Run:
streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from PIL import Image

DATA_DIR = "./data"
MODEL_SERVICE_URL = "http://localhost:8000/predict"  # assumes predict_service is running

st.title("Digital Twin PdM Dashboard (Demo)")

dataset = st.selectbox("Dataset", ["synthetic_normal.csv", "synthetic_slow_degradation.csv", "synthetic_sudden_fault.csv"])
csv_path = os.path.join(DATA_DIR, dataset)
df = pd.read_csv(csv_path, parse_dates=["timestamp"])

start = st.number_input("Start index", min_value=0, max_value=len(df)-200, value=4800)
window = 200
sub = df.iloc[start:start+window]
st.subheader("Sensor time series (window)")
st.line_chart(sub[["temperature", "vibration", "pressure", "rpm"]].set_index(pd.to_datetime(sub["timestamp"])))

# Prepare model input: take seq_len last points
seq_len = 100
sample_window = df[["temperature","vibration","pressure","rpm"]].iloc[start+window-seq_len:start+window].values.tolist()

st.subheader("Model prediction")
if st.button("Get prediction"):
    payload = {"data": sample_window}
    try:
        resp = requests.post(MODEL_SERVICE_URL, json=payload, timeout=5)
        prob = resp.json().get("failure_probability", None)
        st.metric("Failure probability", f"{prob:.3f}")
    except Exception as e:
        st.error("Could not reach model service. Ensure it is running (uvicorn).")
        st.write("Local fallback: random demo probability.")
        st.metric("Failure probability (demo)", f"{np.random.rand():.3f}")

st.subheader("SHAP explanation (cached)")
shap_img_path = "./explain/shap_bar.png"
if os.path.exists(shap_img_path):
    st.image(Image.open(shap_img_path), caption="SHAP importance (sample window)")
else:
    st.info("Run shap_explain.py to generate a SHAP bar plot saved to ./explain/shap_bar.png")
