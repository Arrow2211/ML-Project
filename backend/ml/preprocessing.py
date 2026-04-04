"""
Data preprocessing module for the multi-hazard risk prediction system.
Handles missing values, feature engineering, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


FEATURE_COLUMNS = [
    "Latitude", "Longitude", "Temperature", "Rainfall", "Humidity",
    "Wind_Speed", "Earthquake_Frequency", "Drought_Index", "Cyclone_Risk"
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
    df["Rainfall_Wind_Interaction"] = (df["Rainfall"] * df["Wind_Speed"]) / 100.0
    
    # Composite hazard index: weighted combination of key risk indicators
    df["Composite_Hazard_Index"] = (
        0.3 * df["Cyclone_Risk"] +
        0.25 * (df["Earthquake_Frequency"] / df["Earthquake_Frequency"].max()) +
        0.2 * df["Drought_Index"] +
        0.15 * (df["Rainfall"] / df["Rainfall"].max()) +
        0.1 * (df["Wind_Speed"] / df["Wind_Speed"].max())
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
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[available_features].values
    
    if is_training:
        # Fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode target
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["Risk_Level"])
        
        return X_scaled, y, scaler, label_encoder, available_features
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
