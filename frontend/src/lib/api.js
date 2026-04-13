const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchCities() {
  const res = await fetch(`${API_BASE}/api/cities`);
  if (!res.ok) throw new Error("Failed to fetch cities");
  return res.json();
}

export async function predictRisk(data) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function getFeatureImportance() {
  const res = await fetch(`${API_BASE}/api/feature-importance`);
  if (!res.ok) throw new Error("Failed to fetch feature importance");
  return res.json();
}

export async function getRiskDistribution(city = null) {
  const url = city 
    ? `${API_BASE}/api/risk-distribution?city=${encodeURIComponent(city)}` 
    : `${API_BASE}/api/risk-distribution`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch risk distribution");
  return res.json();
}

export async function getCityRisks() {
  const res = await fetch(`${API_BASE}/api/city-risks`);
  if (!res.ok) throw new Error("Failed to fetch city risks");
  return res.json();
}

export async function trainModel() {
  const res = await fetch(`${API_BASE}/api/train`, { method: "POST" });
  if (!res.ok) throw new Error("Training failed");
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/api/model-info`);
  if (!res.ok) throw new Error("Failed to fetch model info");
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
