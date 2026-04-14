import joblib
import os
import pandas as pd
import numpy as np
import sys

# Add parent directory to path to import ml module
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from ml.model import load_model, predict_risk
from ml.preprocessing import preprocess_single_input

def test_consensus():
    model, scaler, le, feat_names, meta = load_model('backend/models')
    if not model:
        print("Model not found")
        return

    # User's exact manual entry values
    # Temperature: 23.5, Rainfall: 1012, Humidity: 42, Wind: 15, EQ: 0, Drought: 0.22, Cyclone: 0.27.
    input_data = {
        "Latitude": 20.0,
        "Longitude": 78.0,
        "Temperature": 23.5,
        "Rainfall": 1012.0,
        "Humidity": 42.0,
        "Wind_Speed": 15.0,
        "Earthquake_Frequency": 0,
        "Drought_Index": 0.22,
        "Cyclone_Risk": 0.27
    }

    X_scaled, actual_feat_names = preprocess_single_input(input_data, scaler)
    accuracies = meta.get("individual_accuracies")
    
    result = predict_risk(model, le, X_scaled, actual_feat_names, accuracies)
    
    print(f"\nOverall Risk: {result['risk_level']}")
    print(f"Confidence Level: {result['confidence_level']}")
    print(f"Individual Predictions: {result['model_predictions']}")
    print("\nExplanation:\n", result['explanation'])

if __name__ == "__main__":
    test_consensus()
