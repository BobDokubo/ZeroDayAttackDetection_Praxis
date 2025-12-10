# app.py (Flask Backend)

import os
import random
import time
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json # For debugging/logging if needed

# --- Configuration ---
app = Flask(__name__)
MODEL_PATH = 'zero_day_encoder_model.pth'
# Assuming the scaler was fit on data with the same number of features as input_dim
# For a real system, you'd save/load the scaler as well.
# For this demo, we'll re-initialize a dummy scaler and use the exact number of features
# determined by the loaded model's input layer.
GLOBAL_SCALER = None # Will be initialized after model loads
MODEL_INPUT_DIM = None # Will be set by the loaded model
MODEL_LATENT_DIM = 32 # Must match the latent_dim used during training
ANOMALY_THRESHOLD = 5.0 # Adjustable threshold for flagging attacks (Euclidean distance)

# --- PyTorch Model Architecture (Must match training script) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim) # Latent dimension for embeddings

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- Load the Trained Model and Initialize Scaler/Centroid ---
# This will be run once when the Flask app starts
def load_model_and_params():
    global GLOBAL_SCALER, MODEL_INPUT_DIM, GLOBAL_CENTROID

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the Jupyter Notebook first.")
        # Fallback to dummy model if not found
        # This will allow the app to run but not perform real anomaly detection
        MODEL_INPUT_DIM = 7 + 39 # Dummy, assuming original 7 financial + 39 CICIDS features
        GLOBAL_SCALER = StandardScaler()
        # Initialize a dummy encoder for app startup without a model file
        dummy_encoder = Encoder(MODEL_INPUT_DIM, MODEL_LATENT_DIM)
        GLOBAL_CENTROID = np.random.rand(MODEL_LATENT_DIM) * 0.1 # Small random centroid
        return dummy_encoder

    try:
        # Create a dummy instance to load the state_dict into
        # We need to infer the input_dim from the saved state_dict or hardcode it
        # A more robust way is to save model architecture as well, or pass input_dim during saving.
        # For now, let's assume input_dim = 7 (financial) + 39 (CICIDS selected) = 46.
        # This input_dim must precisely match what the trained model expects.
        temp_input_dim = 7 + 39 # Base assumption: 7 financial + 39 CICIDS features
        temp_encoder = Encoder(temp_input_dim, MODEL_LATENT_DIM)

        # Load the state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu')) # Map to CPU as Flask runs on CPU

        # Update input_dim based on the loaded state_dict if possible
        # Check the first linear layer's weight shape
        if 'fc1.weight' in state_dict:
            MODEL_INPUT_DIM = state_dict['fc1.weight'].shape[1]
            temp_encoder = Encoder(MODEL_INPUT_DIM, MODEL_LATENT_DIM) # Recreate with correct input_dim
        else:
            print(f"Warning: Could not infer input_dim from model state_dict. Using assumed: {temp_input_dim}")
            MODEL_INPUT_DIM = temp_input_dim

        temp_encoder.load_state_dict(state_dict)
        temp_encoder.eval() # Set to evaluation mode

        # Load the scaler and centroid. In a real system, you'd save these from your training notebook.
        # For this demo, we'll create a dummy scaler and centroid that correspond to the model's input_dim.
        GLOBAL_SCALER = StandardScaler()
        # In a production setup, the scaler's parameters (mean, std) and the centroid
        # would be saved during training and loaded here. For simplicity, we'll
        # just initialize a generic scaler and a placeholder centroid.
        # The centroid should ideally be learned from the *benign* training data.
        # For demonstration, we'll generate a random one and rely on the model's embeddings.
        GLOBAL_CENTROID = np.random.rand(MODEL_LATENT_DIM) # Placeholder centroid.

        print(f"Model loaded successfully. Input Dimension: {MODEL_INPUT_DIM}")
        return temp_encoder

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy model for application startup.")
        MODEL_INPUT_DIM = 7 + 39 # Fallback
        GLOBAL_SCALER = StandardScaler()
        dummy_encoder = Encoder(MODEL_INPUT_DIM, MODEL_LATENT_DIM)
        GLOBAL_CENTROID = np.random.rand(MODEL_LATENT_DIM) * 0.1
        return dummy_encoder

ENCODER_MODEL = load_model_and_params()

# --- Helper Function for Anomaly Detection ---
def detect_anomaly(raw_data_point):
    """
    Processes a single raw data point and returns its anomaly score and classification.
    """
    global GLOBAL_SCALER, ENCODER_MODEL, GLOBAL_CENTROID, MODEL_INPUT_DIM

    # Ensure the input data has the correct number of features
    if len(raw_data_point) != MODEL_INPUT_DIM:
        print(f"Input data dimension mismatch: Expected {MODEL_INPUT_DIM}, got {len(raw_data_point)}")
        # Pad or truncate if dimensions don't match (for robust demo)
        if len(raw_data_point) < MODEL_INPUT_DIM:
            raw_data_point = np.pad(raw_data_point, (0, MODEL_INPUT_DIM - len(raw_data_point)), 'constant')
        else:
            raw_data_point = raw_data_point[:MODEL_INPUT_DIM]


    # Reshape for scaler (needs 2D array: n_samples, n_features)
    data_point_2d = np.array(raw_data_point).reshape(1, -1)

    # Use a dummy fit_transform if scaler hasn't seen data, otherwise transform
    # In a real app, the scaler would be loaded, or fit on a small sample of representative data at startup.
    # For robust demo: if scaler has no 'mean_' attr (not fitted), fit it on some dummy data first.
    if not hasattr(GLOBAL_SCALER, 'mean_') or GLOBAL_SCALER.mean_ is None or GLOBAL_SCALER.mean_.shape[0] != MODEL_INPUT_DIM:
        print("Scaler not fitted or dimension mismatch, fitting dummy scaler...")
        # Create dummy data for scaler to fit, matching input_dim
        dummy_fit_data = np.random.rand(100, MODEL_INPUT_DIM)
        GLOBAL_SCALER.fit(dummy_fit_data)

    scaled_data_point = GLOBAL_SCALER.transform(data_point_2d)

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(scaled_data_point, dtype=torch.float32)

    with torch.no_grad():
        embedding = ENCODER_MODEL(data_tensor).cpu().numpy().flatten()

    # Calculate anomaly score (Euclidean distance to centroid)
    anomaly_score = np.linalg.norm(embedding - GLOBAL_CENTROID)

    # Classify based on threshold
    is_anomaly = anomaly_score > ANOMALY_THRESHOLD
    attack_status = "Attack Detected!" if is_anomaly else "Normal Behavior"
    reaction_message = ""
    if is_anomaly:
        reaction_message = "Immediate transaction review triggered. Connection flagged."
        # Simulate prevention by, e.g., setting a flag, initiating a block, etc.
        # In a real system, this would trigger actual security measures.
        print(f"ALERT: Zero-Day Attack Detected! Score: {anomaly_score:.2f}")

    return {
        'score': float(anomaly_score),
        # Explicitly convert numpy.bool_ to Python bool for jsonify compatibility
        'is_anomaly': bool(is_anomaly),
        'status': attack_status,
        'reaction': reaction_message,
        'attack_type': random.choice(['Phishing', 'Malware', 'DDoS', 'Insider Threat', 'Zero-Day Exploitation']) if is_anomaly else 'Benign'
    }

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main dashboard HTML page."""
    # Pass MODEL_INPUT_DIM to the frontend for simulation logic
    return render_template('index.html', MODEL_INPUT_DIM=MODEL_INPUT_DIM)

@app.route('/api/analyze_log', methods=['POST'])
def analyze_log():
    """
    API endpoint to receive simulated log data, run anomaly detection,
    and return results.
    """
    try:
        data = request.get_json()
        raw_log_data = data.get('log_features')
        if not raw_log_data:
            return jsonify({'error': 'No log_features provided'}), 400

        # Convert list to numpy array
        raw_log_data = np.array(raw_log_data, dtype=np.float32)

        result = detect_anomaly(raw_log_data)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /api/analyze_log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def get_metrics():
    """
    Simulates real-time metrics for the dashboard.
    In a real system, this would fetch from a database or monitoring system.
    """
    total_transactions = random.randint(100000, 1000000)
    threats_detected = random.randint(50, 500)
    blocked_attempts = random.randint(30, threats_detected)
    active_users = random.randint(1000, 50000)
    return jsonify({
        'totalTransactions': total_transactions,
        'threatsDetected': threats_detected,
        'blockedAttempts': blocked_attempts,
        'activeUsers': active_users,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    app.run(debug=True) # debug=True enables auto-reloading and better error messages