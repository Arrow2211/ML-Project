import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from ml.model import predict_risk, load_model
from ml.preprocessing import preprocess_single_input

def test_probabilities():
    # Load model and components
    try:
        model, scaler, cluster_mapping, feature_names = load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test Case 1: Low Hazard (Minimal values)
    low_hazard = {
        "Temperature": 20, "Rainfall": 10, "Humidity": 50, "Wind_Speed": 5,
        "Earthquake_Frequency": 0, "Drought_Index": 0.1, "Cyclone_Risk": 0.05,
        "Latitude": 19.0, "Longitude": 72.0
    }
    
    # Test Case 2: High Hazard (Maximum values)
    high_hazard = {
        "Temperature": 45, "Rainfall": 3500, "Humidity": 90, "Wind_Speed": 120,
        "Earthquake_Frequency": 100, "Drought_Index": 1.0, "Cyclone_Risk": 1.0,
        "Latitude": 10.0, "Longitude": 90.0
    }

    for name, features in [("Low Hazard", low_hazard), ("High Hazard", high_hazard)]:
        print(f"\n--- Testing {name} ---")
        X_scaled, feat_names = preprocess_single_input(features, scaler)
        result = predict_risk(model, cluster_mapping, X_scaled, feat_names)
        
        print(f"Result: {result['risk_level']}")
        print("Probabilities:")
        for level, prob in result['probabilities'].items():
            print(f"  {level}: {prob}%")

if __name__ == "__main__":
    test_probabilities()
