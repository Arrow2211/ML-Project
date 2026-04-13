"""
Multi-model ensemble (Random Forest, Gradient Boosting, and SVM) for multi-hazard risk prediction.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_model(X, y, feature_names, test_size=0.2, seed=42):
    """
    Train an Ensemble of 3 models: Random Forest, Gradient Boosting, and SVM.
    
    Returns:
        dict with ensemble model, individual metrics, and ensemble metrics.
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
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, class_weight="balanced")
    
    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed)
    
    # 3. SVM (with probability=True for soft voting)
    sv = SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced")
    
    # Create Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('sv', sv)],
        voting='soft'
    )
    
    # Train Ensemble (this fits all underlying models)
    print("  Training Ensemble Model (RF + GBDT + SVM)...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate Ensemble
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Individual model accuracies (for UI awareness)
    individual_metrics = {}
    for name, model in ensemble.named_estimators_.items():
        m_pred = model.predict(X_test)
        m_acc = accuracy_score(y_test, m_pred)
        individual_metrics[name] = round(float(m_acc), 4)
    
    # Global Feature importance (from Random Forest as representative)
    # Note: SVM doesn't have feature_importances_ for RBF kernel
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    importance_dict = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    }
    
    return {
        "model": ensemble,
        "accuracy": round(float(accuracy), 4),
        "individual_accuracies": individual_metrics,
        "classification_report": report,
        "feature_importance": importance_dict,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def predict_risk(ensemble, label_encoder, X_scaled, feature_names):
    """
    Predict risk level using the ensemble and return individual model verdicts.
    
    Returns:
        dict with final prediction, individual model predictions, and feature info
    """
    # Final Ensemble Prediction
    prediction_encoded = ensemble.predict(X_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Probabilities from Ensemble
    probabilities = ensemble.predict_proba(X_scaled)[0]
    prob_dict = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    # Individual model predictions
    model_predictions = {}
    model_names = {"rf": "Random Forest", "gb": "Gradient Boosting", "sv": "SVM"}
    
    for name, model in ensemble.named_estimators_.items():
        m_pred_encoded = model.predict(X_scaled)
        m_label = label_encoder.inverse_transform(m_pred_encoded)[0]
        model_predictions[model_names.get(name, name)] = m_label
    
    # Feature contribution (from RF)
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    feature_values = X_scaled[0]
    raw_contributions = importances * np.abs(feature_values)
    total = raw_contributions.sum()
    contributions = raw_contributions / total if total > 0 else importances
    
    contribution_dict = {
        name: round(float(contrib) * 100, 2)
        for name, contrib in sorted(zip(feature_names, contributions), key=lambda x: -x[1])
    }
    
    return {
        "risk_level": prediction_label,
        "probabilities": prob_dict,
        "model_predictions": model_predictions,
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
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    
    parts = []
    for feat, pct in top_features:
        desc = feature_descriptions.get(feat, feat)
        parts.append(f"{desc} ({pct}%)")
    
    if risk_level == "High":
        return f"Warning: High risk primarily due to {parts[0]} and {parts[1]}. Precautionary measures are highly recommended."
    elif risk_level == "Medium":
        return f"Moderate risk detected, primarily influenced by {parts[0]}. Stay alert for weather updates."
    else:
        return f"Low hazard risk. Major factors like {parts[0]} are currently within safe limits."


def get_feature_importance(ensemble, feature_names):
    """Get global feature importance from the combined model (using RF as anchor)."""
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    return {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    }


def save_model(ensemble, scaler, label_encoder, feature_names, path=None):
    """Save model artifacts to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or MODEL_DIR
    
    joblib.dump(ensemble, os.path.join(path, "model.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.joblib"))


def load_model(path=None):
    """Load model artifacts from disk."""
    path = path or MODEL_DIR
    
    ensemble = joblib.load(os.path.join(path, "model.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(path, "feature_names.joblib"))
    
    return ensemble, scaler, label_encoder, feature_names
