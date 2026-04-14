import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_model(X, y, feature_names, test_size=0.2, seed=42):
    """
    Train an Ensemble of 3 models and calculate feature importance for each.
    
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
    
    # Train Ensemble
    print("  Training Ensemble Models...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate and get individual statistics
    individual_metrics = {}
    importances = {}
    
    # 1. RF Stats
    rf_model = ensemble.named_estimators_['rf']
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    individual_metrics["Random Forest"] = round(float(rf_acc), 4)
    importances["Random Forest"] = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x: -x[1])
    }
    
    # 2. GB Stats
    gb_model = ensemble.named_estimators_['gb']
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
    individual_metrics["Gradient Boosting"] = round(float(gb_acc), 4)
    importances["Gradient Boosting"] = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, gb_model.feature_importances_), key=lambda x: -x[1])
    }
    
    # 3. SVM Stats (Permutation Importance)
    sv_model = ensemble.named_estimators_['sv']
    sv_acc = accuracy_score(y_test, sv_model.predict(X_test))
    individual_metrics["SVM"] = round(float(sv_acc), 4)
    
    print("  Calculating Permutation Importance for SVM (this may take a moment)...")
    perm_imp = permutation_importance(sv_model, X_test, y_test, n_repeats=5, random_state=seed)
    importances["SVM"] = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, perm_imp.importances_mean), key=lambda x: -x[1])
    }
    
    # Evaluate Ensemble
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "model": ensemble,
        "accuracy": round(float(accuracy), 4),
        "individual_accuracies": individual_metrics,
        "classification_report": report,
        "feature_importance": importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def predict_risk(ensemble, label_encoder, X_scaled, feature_names, individual_accuracies=None):
    """
    Predict risk level using the ensemble and return individual model verdicts
    with improved directional feature contribution logic.
    """
    prediction_encoded = ensemble.predict(X_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    probabilities = ensemble.predict_proba(X_scaled)[0]
    prob_dict = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    model_predictions = {}
    model_names = {"rf": "Random Forest", "gb": "Gradient Boosting", "sv": "SVM"}
    
    for name, model in ensemble.named_estimators_.items():
        m_pred_encoded = model.predict(X_scaled)
        m_label = label_encoder.inverse_transform(m_pred_encoded)[0]
        model_predictions[model_names.get(name, name)] = m_label
    
    # --- Corrected Directional Feature Contribution Logic ---
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    f_vals = X_scaled[0]
    
    # We want to show what is DRIVING the specific prediction.
    if prediction_label in ["High", "Medium"]:
        # Reasons for High Risk: Features that are GREATER than their average 
        # (Except Pressure, where lower is riskier)
        directional_vals = np.array([
            -val if name == "Surface_Pressure" else val 
            for name, val in zip(feature_names, f_vals)
        ])
        # Only count positive deviations as "contributing" to the High Risk status
        raw_contributions = importances * np.maximum(0, directional_vals)
    else:
        # Reasons for Low Risk: Features that are LOWER than their average
        # (Except Pressure, where higher is safer)
        directional_vals = np.array([
            val if name == "Surface_Pressure" else -val 
            for name, val in zip(feature_names, f_vals)
        ])
        raw_contributions = importances * np.maximum(0, directional_vals)
    
    # Fallback: if no feature stands out directionally, use absolute deviation
    if raw_contributions.sum() == 0:
        raw_contributions = importances * np.abs(f_vals)
        
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
        "individual_accuracies": individual_accuracies,
        "feature_contributions": contribution_dict,
        "explanation": _generate_explanation(prediction_label, contribution_dict)
    }


def _generate_explanation(risk_level, contributions):
    """Generate a human-readable explanation for the prediction."""
    top_features = list(contributions.items())[:3]
    
    feature_descriptions = {
        "Rainfall": "unusually high rainfall",
        "Wind_Speed": "elevated wind speeds",
        "Earthquake_Frequency": "seismic activity history",
        "Cyclone_Risk": "cyclonic storm indicators",
        "Drought_Index": "drought/rainfall deficit",
        "Temperature": "temperature anomalies",
        "Humidity": "high humidity levels",
        "Latitude": "geographic positioning",
        "Longitude": "regional indicators",
    }
    
    parts = []
    for feat, pct in top_features:
        desc = feature_descriptions.get(feat, feat)
        parts.append(f"{desc} ({pct}%)")
    
    if risk_level == "High":
        return f"Warning: High risk primarily driven by {parts[0]} and {parts[1]}. Urgent precautionary measures are recommended."
    elif risk_level == "Medium":
        return f"Moderate risk detected. Key factors include {parts[0]}. Monitoring of localized weather updates is advised."
    else:
        return f"Low hazard risk. Safety indicators for {parts[0]} are currently stable and well within normal limits."


def get_feature_importance(ensemble, feature_names):
    """Placeholder logic, actual multi-model importance is handled in train_model."""
    # We return the first model's (RF) importance if called, but the API usually 
    # gets it from state['training_result']['feature_importance']
    rf_model = ensemble.named_estimators_['rf']
    return {
        "Random Forest": {
            name: round(float(imp) * 100, 2)
            for name, imp in sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x: -x[1])
        }
    }


def save_model(ensemble, scaler, label_encoder, feature_names, metadata=None, path=None):
    """Save model artifacts and optional metadata to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or MODEL_DIR
    
    joblib.dump(ensemble, os.path.join(path, "model.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.joblib"))
    
    # Save metadata (accuracies, importance records)
    if metadata:
        import json
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)


def load_model(path=None):
    """Load model artifacts and metadata from disk."""
    path = path or MODEL_DIR
    
    if not os.path.exists(os.path.join(path, "model.joblib")):
        return None, None, None, None, None
        
    ensemble = joblib.load(os.path.join(path, "model.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(path, "feature_names.joblib"))
    
    metadata = None
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        import json
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"  ⚠ Failed to load metadata: {e}")
            
    return ensemble, scaler, label_encoder, feature_names, metadata
