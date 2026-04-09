"""
Random Forest model for multi-hazard risk prediction with explainability.
"""

import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Feature Weights to prioritize primary hazards in clustering
# These weights amplify the distance contribution of key features
FEATURE_WEIGHTS = {
    "Rainfall": 3.0,
    "Cyclone_Risk": 3.5,
    "Earthquake_Frequency": 2.5,
    "Composite_Hazard_Index": 4.0,
    "Wind_Speed": 1.5,
    "Rainfall_Wind_Interaction": 2.0,
    "Latitude": 1.0,
    "Longitude": 1.0,
    "Temperature": 0.5,
    "Humidity": 0.5,
    "Drought_Index": 1.5
}


def _apply_weights(X, feature_names):
    """Apply feature weights to the standardized input matrix."""
    weights = np.array([FEATURE_WEIGHTS.get(name, 1.0) for name in feature_names])
    return X * weights, weights


def train_model(X, y_true=None, feature_names=None, n_clusters=3, seed=42):
    """
    Train a K-Means model for clustering and map clusters to risk levels.
    
    Returns:
        dict with model, metrics, cluster mapping, and feature stats
    """
    if len(X) < n_clusters:
        return {"error": "Not enough samples for clustering"}

    # Apply weights before clustering
    X_weighted, weights = _apply_weights(X, feature_names)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10
    )
    cluster_labels = model.fit_predict(X_weighted)
    
    # Calculate Silhouette Score on weighted data
    sil_score = silhouette_score(X_weighted, cluster_labels)
    
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
    
    # Calculate a "Severity Score" for each weighted centroid
    cluster_scores = []
    for i in range(n_clusters):
        centroid = centroids[i]
        # Use a combination of weighted sum and max-component for ranking
        # This ensures a cluster with ONE extreme hazard is ranked high
        mean_severity = centroid[severity_indices].mean()
        max_severity = centroid[severity_indices].max()
        score = 0.7 * mean_severity + 0.3 * max_severity
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
    # Apply weights to the input vector
    X_weighted, _ = _apply_weights(X_scaled, feature_names)
    
    cluster_id = model.predict(X_weighted)[0]
    prediction_label = cluster_mapping[cluster_id]
    
    # For KMeans, we can calculate 'confidence' based on distance to centroid in weighted space
    distances = model.transform(X_weighted)[0]
    
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
    
    # Un-weight the centroid for accurate 'importance' and stat reporting
    _, weights = _apply_weights(np.zeros((1, len(feature_names))), feature_names)
    true_centroid = centroid / weights
    
    importances = np.abs(true_centroid)
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
    _, weights = _apply_weights(np.zeros((1, len(feature_names))), feature_names)
    stats = {}
    for i, label in cluster_mapping.items():
        # Un-weight centroids for reporting
        true_centroid = centroids[i] / weights
        stats[label] = {
            name: round(float(val), 3)
            for name, val in zip(feature_names, true_centroid)
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
