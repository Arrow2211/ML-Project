"use client";

import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

const RISK_COLORS = {
  Low: { bg: "rgba(0, 230, 118, 0.7)", border: "#00e676" },
  Medium: { bg: "rgba(255, 183, 27, 0.7)", border: "#ffb71b" },
  High: { bg: "rgba(255, 75, 75, 0.7)", border: "#ff4b4b" },
};

export default function RiskDistributionChart({ data, title = "Risk Distribution" }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const labels = Object.keys(data);
    const values = Object.values(data);

    chartRef.current = new Chart(canvasRef.current, {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: values,
            backgroundColor: labels.map((l) => RISK_COLORS[l]?.bg || "#888"),
            borderColor: labels.map((l) => RISK_COLORS[l]?.border || "#888"),
            borderWidth: 2,
            hoverOffset: 12,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "#e2e8f0",
              padding: 20,
              font: { size: 13, family: "'Inter', sans-serif" },
              usePointStyle: true,
              pointStyleWidth: 12,
            },
          },
          tooltip: {
            backgroundColor: "rgba(10,10,30,0.9)",
            titleFont: { size: 14, family: "'Inter', sans-serif" },
            bodyFont: { size: 13, family: "'Inter', sans-serif" },
            padding: 12,
            callbacks: {
              label: (ctx) => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                return ` ${ctx.label}: ${ctx.parsed} samples (${pct}%)`;
              },
            },
          },
        },
        animation: {
          animateRotate: true,
          duration: 1500,
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
      <h3>📊 {title}</h3>
      <p className="chart-subtitle">Distribution of risk levels across the dataset</p>
      <div className="chart-container" style={{ height: "340px" }}>
        <canvas ref={canvasRef} id="risk-distribution-chart" />
      </div>
    </div>
  );
}
