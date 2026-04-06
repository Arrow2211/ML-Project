"""
FastAPI backend for the Multi-Hazard Risk Prediction System.
"""

import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi import BackgroundTasks

from ml.data_fetcher import build_dataset, get_city_list, clear_cache, INDIAN_CITIES
from ml.preprocessing import preprocess, preprocess_single_input
from ml.model import train_model, predict_risk, get_cluster_stats, save_model, load_model

app = FastAPI(
    title="Multi-Hazard Risk Prediction System",
    description="Predict disaster risk levels for Indian cities using ML",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
state = {
    "model": None,
    "scaler": None,
    "cluster_mapping": None,
    "feature_names": None,
    "dataset": None,
    "training_result": None,
    "training_error": None,
}


@app.on_event("startup")
async def startup():
    """Start model training in background."""
    print("🚀 FastAPI server is starting up...")
    # Already imported at top level, no need for redundant re-imports
    # Using absolute imports at top level avoids the 'relative import' error on Render
    
    # We use a simple background execution for the initial fetch
    import threading
    def train_task():
        print("🌍 Background: Fetching real government data and training model...")
        try:
            df = build_dataset(start_date="2023-01-01", end_date="2023-12-31", use_cache=True)
            if df.empty:
                state["training_error"] = "No data available"
                print("⚠ Background: No data available. Server running without trained model.")
                return
                
            state["dataset"] = df
            X, y, scaler, cm, feat_names = preprocess(df, is_training=True)
            result = train_model(X, y, feat_names)
            
            if "error" in result:
                state["training_error"] = result["error"]
                return

            state["model"] = result["model"]
            state["scaler"] = scaler
            state["cluster_mapping"] = result["cluster_mapping"]
            state["feature_names"] = feat_names
            state["training_result"] = {
                "silhouette_score": result["silhouette_score"],
                "cluster_mapping": result["cluster_mapping"],
                "train_samples": result["train_samples"],
                "cluster_stats": get_cluster_stats(result["model"], result["cluster_mapping"], feat_names)
            }
            save_model(result["model"], scaler, result["cluster_mapping"], feat_names)
            print(f"✅ Background: Model trained — Silhouette: {result['silhouette_score']:.4f}")
        except Exception as e:
            state["training_error"] = str(e)
            print(f"❌ Background training failed: {e}")

    thread = threading.Thread(target=train_task)
    thread.start()


# ─── Schemas ──────────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    temperature: float
    rainfall: float
    humidity: float
    wind_speed: float
    earthquake_frequency: int
    drought_index: float
    cyclone_risk: float


class FetchDataRequest(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    force_refresh: bool = False


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok", 
        "model_ready": state["model"] is not None,
        "error": state["training_error"]
    }


@app.get("/api/cities")
async def list_cities():
    """Return list of available Indian cities with their historical average data."""
    cities_base = get_city_list()
    if state["dataset"] is not None:
        df = state["dataset"]
        # Calculate medians for each city
        features = ["Temperature", "Rainfall", "Humidity", "Wind_Speed", "Earthquake_Frequency", "Drought_Index", "Cyclone_Risk"]
        medians = df.groupby("City")[features].mean().to_dict('index')
        
        for c in cities_base:
            c_name = c["city"]
            if c_name in medians:
                # Add median values to the city object, rounded nicely
                for feat in features:
                    val = medians[c_name][feat]
                    c[feat] = round(val) if "Frequency" in feat else round(val, 2)
                    
    return {"cities": cities_base}


@app.post("/api/fetch-data")
async def fetch_data(req: FetchDataRequest):
    """Fetch real data from Open-Meteo (weather) and USGS (earthquakes)."""
    if req.force_refresh:
        clear_cache()
    
    df = build_dataset(start_date=req.start_date, end_date=req.end_date, use_cache=not req.force_refresh)
    state["dataset"] = df
    
    risk_dist = df["Risk_Level"].value_counts().to_dict()
    return {
        "message": f"Fetched {len(df)} data points across {df['City'].nunique()} cities",
        "sources": ["Open-Meteo (ERA5/IMD reanalysis)", "USGS Earthquake Catalog"],
        "rows": len(df),
        "columns": list(df.columns),
        "risk_distribution": risk_dist,
    }


@app.post("/api/clear-cache")
async def clear_data_cache():
    """Clear cached dataset to force re-fetch from APIs."""
    clear_cache()
    return {"message": "Cache cleared. Next fetch will pull fresh data from APIs."}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    
    required = ["Temperature", "Rainfall", "Humidity", "Wind_Speed", "Earthquake_Frequency", "Drought_Index", "Cyclone_Risk", "Risk_Level"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    
    state["dataset"] = df
    risk_dist = df["Risk_Level"].value_counts().to_dict()
    
    return {
        "message": f"Uploaded {len(df)} rows",
        "rows": len(df),
        "columns": list(df.columns),
        "risk_distribution": risk_dist,
    }


@app.post("/api/train")
async def train():
    """Train model on current dataset."""
    if state["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Generate or upload data first.")
    
    df = state["dataset"]
    X, y, scaler, cm, feat_names = preprocess(df, is_training=True)
    result = train_model(X, y, feat_names)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    state["model"] = result["model"]
    state["scaler"] = scaler
    state["cluster_mapping"] = result["cluster_mapping"]
    state["feature_names"] = feat_names
    state["training_result"] = {
        "silhouette_score": result["silhouette_score"],
        "cluster_mapping": result["cluster_mapping"],
        "train_samples": result["train_samples"],
        "cluster_stats": get_cluster_stats(result["model"], result["cluster_mapping"], feat_names)
    }
    
    save_model(result["model"], scaler, result["cluster_mapping"], feat_names)
    
    return {
        "message": "Unsupervised K-Means model trained successfully",
        "silhouette_score": result["silhouette_score"],
        "cluster_mapping": result["cluster_mapping"],
        "train_samples": result["train_samples"],
    }


@app.post("/api/predict")
async def predict(input_data: PredictionInput):
    """Predict risk level for given input."""
    if state["model"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Build feature dict
    features = {
        "Latitude": input_data.latitude or 20.0,
        "Longitude": input_data.longitude or 78.0,
        "Temperature": input_data.temperature,
        "Rainfall": input_data.rainfall,
        "Humidity": input_data.humidity,
        "Wind_Speed": input_data.wind_speed,
        "Earthquake_Frequency": input_data.earthquake_frequency,
        "Drought_Index": input_data.drought_index,
        "Cyclone_Risk": input_data.cyclone_risk,
    }
    
    # If city is specified, fill lat/lon
    if input_data.city:
        city_match = next((c for c in INDIAN_CITIES if c["city"].lower() == input_data.city.lower()), None)
        if city_match:
            features["Latitude"] = city_match["lat"]
            features["Longitude"] = city_match["lon"]
    
    X_scaled, feat_names = preprocess_single_input(features, state["scaler"])
    result = predict_risk(state["model"], state["cluster_mapping"], X_scaled, feat_names)
    
    return {
        "city": input_data.city or "Custom Location",
        **result
    }


@app.get("/api/cluster-stats")
async def cluster_stats():
    """Get statistics for each discovered cluster."""
    if state["model"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    stats = get_cluster_stats(state["model"], state["cluster_mapping"], state["feature_names"])
    return {"cluster_stats": stats}


@app.get("/api/model-info")
async def model_info():
    """Get model training results and metrics."""
    if state["training_result"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    return state["training_result"]


@app.get("/api/risk-distribution")
async def risk_distribution(city: Optional[str] = None):
    """Get risk level distribution across the dataset, optionally filtered by city."""
    if state["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    df = state["dataset"]
    if city and city.lower() != "custom location":
        df = df[df["City"].str.lower() == city.lower()]
        
    dist = df["Risk_Level"].value_counts().to_dict()
    return {"distribution": dist, "city": city}


@app.get("/api/city-risks")
async def city_risks():
    """Get risk levels for all cities (for map visualization)."""
    if state["model"] is None or state["dataset"] is None:
        raise HTTPException(status_code=400, detail="Model not ready")
    
    # Get most common risk per city from dataset
    city_risk = state["dataset"].groupby("City")["Risk_Level"].agg(
        lambda x: x.value_counts().index[0]
    ).to_dict()
    
    result = []
    for city_info in INDIAN_CITIES:
        risk = city_risk.get(city_info["city"], "Low")
        result.append({
            "city": city_info["city"],
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "risk_level": risk,
        })
    
    return {"cities": result}


if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
