"use client";

import { useEffect, useRef, useState } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

export default function FeatureImportanceChart({ data, title = "Feature Priority" }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  
  // Normalize data: ensure it's always in { ModelName: { Feature: Value } } format
  const [normalizedData, setNormalizedData] = useState({});
  const [activeModel, setActiveModel] = useState("");

  useEffect(() => {
    if (!data) return;

    // Detect if data is flat (e.g., { Rainfall: 40 }) or nested (e.g., { RF: { Rainfall: 40 } })
    const firstKey = Object.keys(data)[0];
    const isNested = firstKey && typeof data[firstKey] === "object";

    const normalized = isNested ? data : { "Ensemble Contribution": data };
    setNormalizedData(normalized);
    
    // Set active model if not set or if current active doesn't exist in new data
    const firstModel = Object.keys(normalized)[0];
    if (!activeModel || !normalized[activeModel]) {
      setActiveModel(firstModel);
    }
  }, [data]);

  const modelNames = Object.keys(normalizedData);

  useEffect(() => {
    if (!normalizedData || !activeModel || !normalizedData[activeModel] || !canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const currentData = normalizedData[activeModel];
    const labels = Object.keys(currentData);
    const values = Object.values(currentData);

    const colors = [
      "#00e5ff", "#7c4dff", "#ff6e40", "#ffab00",
      "#00e676", "#ff4081", "#448aff", "#69f0ae",
      "#ea80fc", "#ffd740", "#40c4ff",
    ];

    chartRef.current = new Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels: labels.map((l) => l.replace(/_/g, " ")),
        datasets: [
          {
            label: "Importance (%)",
            data: values,
            backgroundColor: labels.map((_, i) => colors[i % colors.length] + "cc"),
            borderColor: labels.map((_, i) => colors[i % colors.length]),
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(10,10,30,0.9)",
            titleFont: { size: 14, family: "'Inter', sans-serif" },
            bodyFont: { size: 13, family: "'Inter', sans-serif" },
            padding: 12,
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            callbacks: {
              label: (ctx) => `${ctx.parsed.x.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.06)" },
            ticks: { color: "#94a3b8", font: { family: "'Inter', sans-serif" } },
            title: {
              display: true,
              text: "Importance (%)",
              color: "#cbd5e1",
              font: { size: 13, family: "'Inter', sans-serif" },
            },
            suggestedMax: 40,
          },
          y: {
            grid: { display: false },
            ticks: { color: "#e2e8f0", font: { size: 12, family: "'Inter', sans-serif" } },
          },
        },
        animation: {
          duration: 800,
          easing: "easeOutQuart",
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [normalizedData, activeModel]);

  if (!data) return null;

  return (
    <div className="chart-card">
      <div className="chart-header-row">
        <h3>📈 {title}</h3>
        {modelNames.length > 1 && (
          <div className="model-tabs">
            {modelNames.map((name) => (
              <button
                key={name}
                className={`model-tab ${activeModel === name ? "active" : ""}`}
                onClick={() => setActiveModel(name)}
              >
                {name}
              </button>
            ))}
          </div>
        )}
      </div>
      
      <p className="chart-subtitle">
        {activeModel === "SVM" 
          ? "Permutation Importance: How much each feature affects SVM accuracy."
          : `Model-specific feature weightage for ${activeModel}.`}
      </p>
      
      <div className="chart-container" style={{ height: "380px" }}>
        <canvas ref={canvasRef} id="feature-importance-chart" />
      </div>
    </div>
  );
}
