# Multi-Hazard Risk Prediction System for India

AI-powered disaster risk prediction system that predicts **Low**, **Medium**, and **High** risk levels for Indian cities using environmental and geological data, with full model explainability.

## Tech Stack

| Layer     | Technology                    |
|-----------|-------------------------------|
| ML        | scikit-learn, pandas, numpy   |
| Backend   | FastAPI, uvicorn              |
| Frontend  | Next.js, Chart.js             |

## Quick Start

### 1. Backend
```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```
Server runs at `http://localhost:8000`. Model auto-trains on startup.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard at `http://localhost:3000`.

## Features

- **Risk Prediction** — Predict disaster risk for any Indian city (50+ cities)
- **Model Explainability** — Feature importance rankings and per-prediction explanations
- **Interactive Dashboard** — City selector, manual data entry, tabbed views
- **Visualizations** — Feature importance bar chart, risk distribution doughnut, India map
- **API Endpoints** — RESTful API for data upload, training, prediction, and analytics

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/cities` | GET | List available cities |
| `/api/predict` | POST | Predict risk level |
| `/api/train` | POST | Retrain the model |
| `/api/feature-importance` | GET | Global feature importance |
| `/api/risk-distribution` | GET | Risk level distribution |
| `/api/city-risks` | GET | Per-city risk for map |
| `/api/upload` | POST | Upload custom CSV |
| `/api/generate-data` | POST | Generate synthetic data |
