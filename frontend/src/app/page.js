"use client";

import { useState, useEffect } from "react";
import PredictionForm from "@/components/PredictionForm";
import RiskResult from "@/components/RiskResult";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import RiskDistributionChart from "@/components/RiskDistributionChart";
import IndiaMap from "@/components/IndiaMap";
import { getFeatureImportance, getRiskDistribution, getModelInfo, healthCheck } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [riskDistribution, setRiskDistribution] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [backendReady, setBackendReady] = useState(false);
  const [activeTab, setActiveTab] = useState("predict");

  useEffect(() => {
    const init = async () => {
      const healthy = await healthCheck();
      setBackendReady(healthy);
      if (healthy) {
        try {
          const [fi, rd, mi] = await Promise.all([
            getFeatureImportance(),
            getRiskDistribution(),
            getModelInfo(),
          ]);
          setFeatureImportance(fi.feature_importance);
          setRiskDistribution(rd.distribution);
          setModelInfo(mi);
        } catch (e) {
          console.error("Failed to load initial data:", e);
        }
      }
    };
    init();
  }, []);

  // Update analytics charts when a new prediction is made for a city
  useEffect(() => {
    if (result && result.city) {
      if (result.feature_contributions) {
        setFeatureImportance(result.feature_contributions);
      }
      const fetchCityData = async () => {
        try {
          const cityQuery = result.city === "Custom Location" ? null : result.city;
          const rd = await getRiskDistribution(cityQuery);
          setRiskDistribution(rd.distribution);
        } catch (e) {
          console.error("Failed to update analytics for city:", e);
        }
      };
      fetchCityData();
    }
  }, [result]);

  return (
    <main className="app">
      {/* Animated background elements */}
      <div className="bg-effects">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* Hero Header */}
      <header className="hero">
        <div className="hero-content">
          <div className="hero-badge">🛡️ AI-Powered Disaster Intelligence</div>
          <h1>
            Multi-Hazard Risk
            <span className="gradient-text"> Prediction System</span>
          </h1>
          <p className="hero-subtitle">
            Predicting disaster risk levels across India using environmental and geological
            data — with full model explainability
          </p>

          {/* Model Stats */}
          {modelInfo && (
            <div className="stats-row">
              <div className="stat-chip">
                <span className="stat-value">{(modelInfo.accuracy * 100).toFixed(1)}%</span>
                <span className="stat-label">Model Accuracy</span>
              </div>
              <div className="stat-chip">
                <span className="stat-value">{modelInfo.train_samples + modelInfo.test_samples}</span>
                <span className="stat-label">Training Samples</span>
              </div>
              <div className="stat-chip">
                <span className="stat-value">50</span>
                <span className="stat-label">Indian Cities</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Connection Status */}
      {!backendReady && (
        <div className="connection-warning">
          <span>⚠️</span>
          <span>Backend not connected. Start the FastAPI server on port 8000.</span>
        </div>
      )}

      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === "predict" ? "active" : ""}`}
          onClick={() => setActiveTab("predict")}
        >
          🎯 Predict
        </button>
        <button
          className={`tab-btn ${activeTab === "analytics" ? "active" : ""}`}
          onClick={() => setActiveTab("analytics")}
        >
          📊 Analytics
        </button>
        <button
          className={`tab-btn ${activeTab === "map" ? "active" : ""}`}
          onClick={() => setActiveTab("map")}
        >
          🗺️ Map
        </button>
      </nav>

      {/* Main Content */}
      <div className="content">
        {activeTab === "predict" && (
          <section className="predict-section">
            <div className="predict-grid">
              <PredictionForm onResult={setResult} onLoading={setLoading} />
              <div className="result-panel">
                {loading && (
                  <div className="loading-state">
                    <div className="loader" />
                    <p>Analyzing risk factors...</p>
                  </div>
                )}
                {!loading && result && <RiskResult result={result} />}
                {!loading && !result && (
                  <div className="empty-state">
                    <div className="empty-icon">🔮</div>
                    <h3>Ready to Predict</h3>
                    <p>Select a city or enter environmental data to get a risk prediction with full explainability.</p>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {activeTab === "analytics" && (
          <section className="analytics-section">
            <div className="charts-grid">
              <FeatureImportanceChart 
                data={featureImportance} 
                title={result?.city && result.city !== "Custom Location" ? `${result.city} Feature Priority` : "Global Feature Importance"}
              />
              <RiskDistributionChart 
                data={riskDistribution} 
                title={result?.city && result.city !== "Custom Location" ? `${result.city} Historical Risk` : "Global Risk Distribution"}
              />
            </div>
          </section>
        )}

        {activeTab === "map" && (
          <section className="map-section">
            <IndiaMap />
          </section>
        )}
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="source-badge">
          <span className="source-icon">🏛️</span>
          Verified Government Data Sourced from IMD (Weather) & NCS/USGS (Seismic)
        </div>
        <p>Multi-Hazard Risk Prediction System • Built with FastAPI + Next.js + scikit-learn</p>
      </footer>
    </main>
  );
}
