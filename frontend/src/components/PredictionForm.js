"use client";

import { useState, useEffect } from "react";
import { fetchCities, predictRisk } from "@/lib/api";

const FIELDS = [
  { name: "temperature", label: "Temperature (°C)", min: -10, max: 55, step: 0.1, default: 25 },
  { name: "rainfall", label: "Rainfall (mm)", min: 0, max: 3500, step: 1, default: 700 },
  { name: "humidity", label: "Humidity (%)", min: 0, max: 100, step: 1, default: 60 },
  { name: "wind_speed", label: "Wind Speed (km/h)", min: 0, max: 150, step: 0.1, default: 15 },
  { name: "earthquake_frequency", label: "Earthquake Frequency", min: 0, max: 100, step: 1, default: 5, isInt: true },
  { name: "drought_index", label: "Drought Index (0–1)", min: 0, max: 1, step: 0.01, default: 0.3 },
  { name: "cyclone_risk", label: "Cyclone Risk (0–1)", min: 0, max: 1, step: 0.01, default: 0.2 },
];

export default function PredictionForm({ onResult, onLoading }) {
  const [cities, setCities] = useState([]);
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState("");
  const [selectedCity, setSelectedCity] = useState("");
  const [mode, setMode] = useState("city"); // "city" or "manual"
  const [formData, setFormData] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.name, f.default]))
  );
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCities()
      .then((data) => {
        const cityList = data.cities || [];
        cityList.sort((a, b) => a.city.localeCompare(b.city));
        setCities(cityList);
        
        // Group cities by State
        const grouped = {};
        cityList.forEach((c) => {
          const s = c.state || "Unknown";
          if (!grouped[s]) grouped[s] = [];
          grouped[s].push(c);
        });

        // Sort cities alphabetically within each state
        Object.keys(grouped).forEach(state => {
          grouped[state].sort((a, b) => a.city.localeCompare(b.city));
        });
        
        // Sort states alphabetically as well
        setStates(Object.keys(grouped).sort((a, b) => a.localeCompare(b)));
      })
      .catch((err) => console.error("Error fetching cities:", err));
  }, []);

  const handleChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    onLoading(true);
    try {
      const payload = {
        ...formData,
        city: mode === "city" ? selectedCity : "Custom Location",
      };
      if (mode === "city" && selectedCity) {
        const city = cities.find((c) => c.city === selectedCity);
        if (city) {
          payload.latitude = city.lat;
          payload.longitude = city.lon;
        }
      }
      const result = await predictRisk(payload);
      onResult(result);
    } catch (err) {
      console.error(err);
      setError(err.message === "Prediction failed" 
        ? "Prediction failed. The model might still be training or the backend is busy. Please try again in 1 minute." 
        : err.message);
    }
    onLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <div className="form-header">
        <h2>🎯 Risk Prediction</h2>
        <div className="mode-toggle">
          <button
            type="button"
            className={`toggle-btn ${mode === "city" ? "active" : ""}`}
            onClick={() => setMode("city")}
          >
            Select City
          </button>
          <button
            type="button"
            className={`toggle-btn ${mode === "manual" ? "active" : ""}`}
            onClick={() => setMode("manual")}
          >
            Manual Entry
          </button>
        </div>
      </div>

      {mode === "city" && (
        <div className="city-select-group" style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
          <div className="field-group" style={{ flex: 1, marginBottom: 0 }}>
            <label htmlFor="state-select">State</label>
            <select
              id="state-select"
              value={selectedState}
              onChange={(e) => {
                setSelectedState(e.target.value);
                setSelectedCity(""); // Reset city when state changes
              }}
              required
            >
              <option value="">— Select a State —</option>
              {states.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="field-group" style={{ flex: 1, marginBottom: 0 }}>
            <label htmlFor="city-select">City / District</label>
            <select
              id="city-select"
              value={selectedCity}
              onChange={(e) => {
                const cityName = e.target.value;
                setSelectedCity(cityName);
                if (cityName) {
                  const cityData = cities.find((c) => c.city === cityName);
                  if (cityData && cityData.Temperature !== undefined) {
                    setFormData((prev) => ({
                      ...prev,
                      temperature: cityData.Temperature,
                      rainfall: cityData.Rainfall,
                      humidity: cityData.Humidity,
                      wind_speed: cityData.Wind_Speed,
                      earthquake_frequency: cityData.Earthquake_Frequency,
                      drought_index: cityData.Drought_Index,
                      cyclone_risk: cityData.Cyclone_Risk,
                    }));
                  } else {
                    // Reset to defaults if no data available yet for this city
                    setFormData(Object.fromEntries(FIELDS.map((f) => [f.name, f.default])));
                  }
                }
              }}
              required={mode === "city"}
              disabled={!selectedState}
            >
              <option value="">— Select City/District —</option>
              {cities
                .filter((c) => c.state === selectedState)
                .map((c) => (
                  <option key={c.city} value={c.city}>
                    {c.city}
                  </option>
                ))}
            </select>
          </div>
        </div>
      )}
      
      {error && (
        <div className="form-error" style={{ padding: "0.8rem", backgroundColor: "#ffefef", color: "#d32f2f", borderRadius: "8px", marginBottom: "1.5rem", border: "1px solid #ffcfcf", fontSize: "0.9rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span>⚠️</span> {error}
        </div>
      )}

      <div className="fields-grid">
        {FIELDS.map((field) => (
          <div key={field.name} className="field-group">
            <label htmlFor={`field-${field.name}`}>
              {field.label}
              <span className="field-value">{formData[field.name]}</span>
            </label>
            <input
              type="range"
              id={`field-${field.name}`}
              min={field.min}
              max={field.max}
              step={field.step}
              value={formData[field.name]}
              onChange={(e) =>
                handleChange(
                  field.name,
                  field.isInt ? parseInt(e.target.value) : parseFloat(e.target.value)
                )
              }
            />
            <div className="range-labels">
              <span>{field.min}</span>
              <span>{field.max}</span>
            </div>
          </div>
        ))}
      </div>

      <button type="submit" className="submit-btn" id="predict-button">
        <span className="btn-icon">⚡</span>
        Predict Risk Level
      </button>
    </form>
  );
}
