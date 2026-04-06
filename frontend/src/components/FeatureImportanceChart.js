"use client";

import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

export default function FeatureImportanceChart({ data, title = "Global Feature Importance" }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const labels = Object.keys(data);
    const values = Object.values(data);

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
          },
          y: {
            grid: { display: false },
            ticks: { color: "#e2e8f0", font: { size: 12, family: "'Inter', sans-serif" } },
          },
        },
        animation: {
          duration: 1200,
          easing: "easeOutQuart",
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [data]);

  if (!data) return null;

  return (
    <div className="chart-card">
      <h3>📈 {title}</h3>
      <p className="chart-subtitle">
        {title.includes("Global") 
          ? "How much each feature generally affects the model's decisions." 
          : `Main factors that influenced the risk for ${title.split(" ")[0]}.`}
      </p>
      <div className="chart-container" style={{ height: "380px" }}>
        <canvas ref={canvasRef} id="feature-importance-chart" />
      </div>
    </div>
  );
}
