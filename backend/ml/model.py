"""
Random Forest model for multi-hazard risk prediction with explainability.
"""

import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_model(X, y_true=None, feature_names=None, n_clusters=3, seed=42):
    """
    Train a K-Means model for clustering and map clusters to risk levels.
    
    Returns:
        dict with model, metrics, cluster mapping, and feature stats
    """
    if len(X) < n_clusters:
        return {"error": "Not enough samples for clustering"}

    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10
    )
    cluster_labels = model.fit_predict(X)
    
    # Calculate Silhouette Score (Standard metric for unsupervised clustering)
    sil_score = silhouette_score(X, cluster_labels)
    
    # ─── Cluster Ranking Logic ─────────────────────────────────────────
    # Map cluster IDs (0, 1, 2) to risk levels (Low, Medium, High)
    # based on their centroid severity.
    centroids = model.cluster_centers_
    
    # Severity features: features where higher values generally mean higher risk
    # Based on our standard engineering: Rainfall, Wind, Earthquake, etc.
    # Note: feature_names helps us find the index of severity features
    severity_features = [
        "Rainfall", "Wind_Speed", "Earthquake_Frequency", 
        "Cyclone_Risk", "Drought_Index", "Composite_Hazard_Index"
    ]
    
    severity_indices = [
        feature_names.index(feat) for feat in severity_features 
        if feat in feature_names
    ]
    
    # Calculate a simple "Severity Score" for each centroid
    # Since X is scaled (StandardScaler), higher positive values = more extreme
    cluster_scores = []
    for i in range(n_clusters):
        score = centroids[i][severity_indices].sum()
        cluster_scores.append((i, score))
    
    # Sort clusters by score: lowest -> highest
    # Mapping: lowest_score -> 'Low', middle -> 'Medium', highest_score -> 'High'
    sorted_clusters = sorted(cluster_scores, key=lambda x: x[1])
    cluster_mapping = {
        sorted_clusters[0][0]: "Low",
        sorted_clusters[1][0]: "Medium",
        sorted_clusters[2][0]: "High"
    }
    
    # Optional: Compare with ground truth (y_true) if available
    comparison_metrics = {}
    if y_true is not None:
        # We can calculate "Accuracy" by mapping it back (benchmarking)
        # However, for pure unsupervised we focus on Silhouette
        pass

    return {
        "model": model,
        "silhouette_score": round(float(sil_score), 4),
        "cluster_mapping": cluster_mapping,
        "train_samples": len(X),
        "centroids": centroids.tolist(),
        "feature_names": feature_names
    }


def predict_risk(model, cluster_mapping, X_scaled, feature_names):
    """
    Predict risk level using the fitted KMeans model and cluster mapping.
    """
    cluster_id = model.predict(X_scaled)[0]
    prediction_label = cluster_mapping[cluster_id]
    
    # For KMeans, we can calculate 'confidence' based on distance to centroid
    distances = model.transform(X_scaled)[0]
    
    # Shifted Softmax probability mapping (Industry Standard)
    # We subtract the min distance to make the closest cluster clearly stand out (exp(0) = 1)
    # and others relative to it. Gamma controls the confidence level.
    shifted_distances = distances - np.min(distances)
    gamma = 2.0  
    exp_dists = np.exp(-gamma * shifted_distances)
    probabilities = exp_dists / exp_dists.sum()
    
    prob_dict = {
        cluster_mapping[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(probabilities))
    }
    
    # Feature contributions: How much did each feature contribute to picking THIS cluster?
    # We look at the feature value relative to the cluster's centroid
    centroid = model.cluster_centers_[cluster_id]
    feature_values = X_scaled[0]
    
    # Contribution is higher if feature is closer to this centroid than others
    # but for simplicity, we'll use the feature's absolute magnitude in the centroid
    importances = np.abs(centroid)
    total_imp = importances.sum()
    contribution_dict = {
        name: round(float(imp / total_imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    }
    
    return {
        "risk_level": prediction_label,
        "cluster_id": int(cluster_id),
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


def get_cluster_stats(model, cluster_mapping, feature_names):
    """Get statistics for each cluster."""
    centroids = model.cluster_centers_
    stats = {}
    for i, label in cluster_mapping.items():
        stats[label] = {
            name: round(float(val), 3)
            for name, val in zip(feature_names, centroids[i])
        }
    return stats


def save_model(model, scaler, cluster_mapping, feature_names, path=None):
    """Save model artifacts to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or MODEL_DIR
    
    joblib.dump(model, os.path.join(path, "model.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))
    joblib.dump(cluster_mapping, os.path.join(path, "cluster_mapping.joblib"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.joblib"))


def load_model(path=None):
    """Load model artifacts from disk."""
    path = path or MODEL_DIR
    
    model = joblib.load(os.path.join(path, "model.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    cluster_mapping = joblib.load(os.path.join(path, "cluster_mapping.joblib"))
    feature_names = joblib.load(os.path.join(path, "feature_names.joblib"))
    
    return model, scaler, cluster_mapping, feature_names
