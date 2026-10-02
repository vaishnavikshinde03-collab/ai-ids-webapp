import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_intrusion_model.keras")
CNN_PATH = os.path.join(MODELS_DIR, "cnn_ids.keras")
AUTOENCODER_PATH = os.path.join(MODELS_DIR, "autoencoder_ids.keras")

app = FastAPI(title="AI-IDS Machine Learning Microservice")

scaler = None
models = {}

@app.on_event("startup")
def load_artifacts():
    global scaler, models
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] MinMax Scaler loaded successfully.")
        else:
            print("[WARNING] MinMax Scaler file not found.")

        if os.path.exists(LSTM_PATH):
            models['lstm'] = tf.keras.models.load_model(LSTM_PATH)
            print("[INFO] LSTM model loaded.")

        if os.path.exists(CNN_PATH):
            models['cnn'] = tf.keras.models.load_model(CNN_PATH)
            print("[INFO] CNN model loaded.")

        if os.path.exists(AUTOENCODER_PATH):
            models['autoencoder'] = tf.keras.models.load_model(AUTOENCODER_PATH)
            print("[INFO] Autoencoder model loaded.")

    except Exception as e:
        print(f"[ERROR] Failed loading model artifacts: {str(e)}")

class TrafficFlowRequest(BaseModel):
    model_choice: Optional[str] = "lstm"  # Options: 'lstm', 'cnn', 'autoencoder'
    features: List[float]                 # Numerical flow features matching training dimension

class BatchTrafficRequest(BaseModel):
    model_choice: Optional[str] = "lstm"
    flows: List[List[float]]

@app.get("/")
def read_root():
    return {
        "service": "AI-IDS ML Engine",
        "status": "online",
        "loaded_models": list(models.keys())
    }

@app.post("/predict")
def predict_flow(payload: TrafficFlowRequest):
    if not scaler:
        raise HTTPException(status_code=500, detail="Scaler not loaded.")
    
    selected_model = payload.model_choice.lower()
    if selected_model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{selected_model}' is not available.")

    # Reshape and scale input features
    features_array = np.array(payload.features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    
    # LSTM needs 3D input: (batch, timesteps, features)
    if selected_model == "lstm":
        model_input = scaled_features.reshape(1, 1, -1)
    else:
        model_input = scaled_features

    model = models[selected_model]

    if selected_model == "autoencoder":
        # Anomaly detection via Reconstruction Loss (MSE)
        reconstruction = model.predict(model_input, verbose=0)
        mse = float(np.mean(np.power(model_input - reconstruction, 2)))
        
        # Empirical threshold for anomaly detection (e.g., 0.05)
        threshold = 0.05
        is_attack = int(mse > threshold)
        confidence = float(min(mse / (threshold * 2), 1.0))

        return {
            "model_used": "autoencoder",
            "prediction": is_attack,
            "label": "Attack" if is_attack == 1 else "Benign",
            "reconstruction_mse": mse,
            "confidence": confidence
        }
    else:
        # Supervised classification (LSTM / CNN)
        prediction_prob = float(model.predict(model_input, verbose=0)[0][0])
        is_attack = int(prediction_prob > 0.5)

        return {
            "model_used": selected_model,
            "prediction": is_attack,
            "label": "Attack" if is_attack == 1 else "Benign",
            "confidence_score": prediction_prob if is_attack == 1 else (1 - prediction_prob)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)