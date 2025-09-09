# ML-Powered Digital Twin - Demo (Review-2 artifacts)

## Goal
Create a runnable demo with synthetic sensor data, baseline LSTM, SHAP explanation, and a dashboard.

## Steps to reproduce (local)
1. Create a Python virtual env and install requirements:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


2. Generate synthetic data:
python data/generate_synthetic.py --outdir ./data


3. Train baseline LSTM:
python models/train_lstm.py --data ./data/synthetic_slow_degradation.csv --out ./models

This saves `models/lstm_model.h5`, `models/scaler.pkl`, and `models/training_history.csv`.

4. Generate SHAP explanation (requires model + scaler saved):
python explain/shap_explain.py --data ./data/synthetic_sudden_fault.csv --model ./models/lstm_model.h5 --
scaler ./models/scaler.pkl --out ./explain

This creates `explain/shap_bar.png`.

5. Run model service:
uvicorn models.predict_service:app --reload --port 8000


6. Run Streamlit dashboard:
streamlit run dashboard/app.py



