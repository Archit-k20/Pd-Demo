"""
predict_service.py
Simple FastAPI server to load the saved model and serve predictions.

Run:
uvicorn models.predict_service:app --reload --port 8000
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/lstm_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")
SEQ_LEN = 100

class Payload(BaseModel):
    # expects a list of feature rows length == SEQ_LEN
    data: list

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")

@app.post("/predict")
def predict(payload: Payload):
    arr = np.array(payload.data)
    # arr shape expected: (seq_len, n_features)
    arr_scaled = scaler.transform(arr)
    input_batch = np.expand_dims(arr_scaled, axis=0)  # (1, seq_len, n_features)
    prob = float(model.predict(input_batch)[0,0])
    return {"failure_probability": prob}
