"""
Data preprocessing module for the multi-hazard risk prediction system.
Handles missing values, feature engineering, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


FEATURE_COLUMNS = [
    "Latitude", "Longitude", "Temperature", "Rainfall", "Humidity",
    "Wind_Speed", "Earthquake_Frequency", "Drought_Index", "Cyclone_Risk",
    "Rainfall_Wind_Interaction", "Composite_Hazard_Index"
]


def handle_missing_values(df):
    """Fill missing numeric values with column median."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df):
    """Create derived features for better prediction."""
    df = df.copy()
    
    # Interaction: Rainfall × Wind Speed (proxy for storm severity)
    # Filling NANs to avoid issues
    df["Rainfall"] = df["Rainfall"].fillna(0)
    df["Wind_Speed"] = df["Wind_Speed"].fillna(0)
    df["Rainfall_Wind_Interaction"] = (df["Rainfall"] * df["Wind_Speed"]) / 100.0
    
    # Composite hazard index: weighted combination of key risk indicators
    # We use realistic normalization constants for the Indian context.
    max_eq = 10.0   # 10 earthquakes in a period is already extreme for most of India
    max_rain = 3500.0 # Align with extreme monsoon peaks (e.g. Konkan coast)
    max_wind = 150.0
    
    # Weights optimized for socio-economic impact in India: 
    # Rainfall and Cyclones affect more people/cities more frequently than major earthquakes.
    df["Composite_Hazard_Index"] = (
        0.35 * df.get("Rainfall", 0).fillna(0) / max_rain +
        0.25 * df.get("Cyclone_Risk", 0).fillna(0) +
        0.20 * (df.get("Earthquake_Frequency", 0).fillna(0) / max_eq) +
        0.10 * df.get("Drought_Index", 0).fillna(0) +
        0.10 * (df.get("Wind_Speed", 0).fillna(0) / max_wind)
    )
    
    return df


def preprocess(df, scaler=None, label_encoder=None, is_training=True):
    """
    Full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        scaler: Pre-fitted StandardScaler (for prediction mode)
        label_encoder: Pre-fitted LabelEncoder (for prediction mode)
        is_training: Whether we're in training mode
    
    Returns:
        If training: (X_scaled, y_encoded, scaler, label_encoder, feature_names)
        If prediction: (X_scaled, feature_names)
    """
    df = handle_missing_values(df)
    df = engineer_features(df)
    
    # Select feature columns
    available_features = [col for col in df.columns if col in FEATURE_COLUMNS]
    X = df[available_features].values
    
    if is_training:
        # Fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode target if available (optional for unsupervised)
        y_encoded = None
        if "Risk_Level" in df.columns:
            label_encoder = label_encoder or LabelEncoder()
            # Ensure label encoder is always fit to the standard levels for consistency
            label_encoder.fit(["Low", "Medium", "High"])
            y_encoded = label_encoder.transform(df["Risk_Level"])
        
        return X_scaled, y_encoded, scaler, label_encoder, available_features
    else:
        # Use pre-fitted scaler
        X_scaled = scaler.transform(X)
        return X_scaled, available_features


def preprocess_single_input(input_data, scaler):
    """
    Preprocess a single prediction input.
    
    Args:
        input_data: dict with feature values
        scaler: Pre-fitted StandardScaler
    
    Returns:
        (X_scaled, feature_names)
    """
    df = pd.DataFrame([input_data])
    df = engineer_features(df)
    
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[available_features].values
    X_scaled = scaler.transform(X)
    
    return X_scaled, available_features
