"""
Random Forest model for multi-hazard risk prediction with explainability.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_model(X, y, feature_names, test_size=0.2, seed=42):
    """
    Train a Random Forest Classifier.
    
    Returns:
        dict with model, metrics, feature importances, and train/test data info
    """
    # Check if we have enough samples for stratification
    min_class_counts = np.unique(y, return_counts=True)[1].min()
    stratify_data = y if min_class_counts >= 2 else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify_data
        )
    except Exception as e:
        print(f"  ⚠ Stratified split failed: {e}. Falling back to simple split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Feature importance
    importances = model.feature_importances_
    importance_dict = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    }
    
    return {
        "model": model,
        "accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importance": importance_dict,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def predict_risk(model, label_encoder, X_scaled, feature_names):
    """
    Predict risk level with per-prediction feature contribution.
    
    Uses the tree structure to determine feature contribution for each prediction.
    
    Returns:
        dict with prediction, probabilities, and feature contributions
    """
    prediction_encoded = model.predict(X_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    probabilities = model.predict_proba(X_scaled)[0]
    prob_dict = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    # Per-prediction feature contribution using tree-based approach
    # We use the feature importances weighted by the absolute feature values
    importances = model.feature_importances_
    feature_values = X_scaled[0]
    
    # Weighted contributions: importance × |feature_value| (standardized)
    raw_contributions = importances * np.abs(feature_values)
    total = raw_contributions.sum()
    if total > 0:
        contributions = raw_contributions / total
    else:
        contributions = importances
    
    contribution_dict = {
        name: round(float(contrib) * 100, 2)
        for name, contrib in sorted(zip(feature_names, contributions), key=lambda x: -x[1])
    }
    
    return {
        "risk_level": prediction_label,
        "probabilities": prob_dict,
        "feature_contributions": contribution_dict,
        "explanation": _generate_explanation(prediction_label, contribution_dict)
    }


def _generate_explanation(risk_level, contributions):
    """Generate a human-readable explanation for the prediction."""
    top_features = list(contributions.items())[:3]
    
    feature_descriptions = {
        "Rainfall": "rainfall levels",
        "Wind_Speed": "wind speed",
        "Earthquake_Frequency": "earthquake frequency",
        "Cyclone_Risk": "cyclone risk",
        "Drought_Index": "drought conditions",
        "Temperature": "temperature",
        "Humidity": "humidity levels",
        "Latitude": "geographic latitude",
        "Longitude": "geographic longitude",
        "Rainfall_Wind_Interaction": "combined rainfall-wind intensity",
        "Composite_Hazard_Index": "composite hazard index",
    }
    
    parts = []
    for feat, pct in top_features:
        desc = feature_descriptions.get(feat, feat)
        parts.append(f"{desc} ({pct}%)")
    
    if risk_level == "High":
        return f"The risk is HIGH primarily due to {parts[0]}, followed by {parts[1]} and {parts[2]}. Immediate precautionary measures are recommended."
    elif risk_level == "Medium":
        return f"The risk is MEDIUM, mainly influenced by {parts[0]}, with {parts[1]} and {parts[2]} also contributing. Monitoring is advised."
    else:
        return f"The risk is LOW. The main factors considered were {parts[0]}, {parts[1]}, and {parts[2]}, all within safe thresholds."


def get_feature_importance(model, feature_names):
    """Get global feature importance from the trained model."""
    importances = model.feature_importances_
    return {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    }


def save_model(model, scaler, label_encoder, feature_names, path=None):
    """Save model artifacts to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or MODEL_DIR
    
    joblib.dump(model, os.path.join(path, "model.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.joblib"))


def load_model(path=None):
    """Load model artifacts from disk."""
    path = path or MODEL_DIR
    
    model = joblib.load(os.path.join(path, "model.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(path, "feature_names.joblib"))
    
    return model, scaler, label_encoder, feature_names
