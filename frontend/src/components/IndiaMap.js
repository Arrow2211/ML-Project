"use client";

import { useEffect, useState } from "react";
import { getCityRisks } from "@/lib/api";

const RISK_COLORS = {
  High: "#ff4b4b",
  Medium: "#ffb71b",
  Low: "#00e676",
};

// Simple SVG map projection for India (Mercator approx)
// India bounds: lat 6-37, lon 68-98
function project(lat, lon) {
  const mapWidth = 500;
  const mapHeight = 580;
  const lonMin = 67, lonMax = 98;
  const latMin = 6, latMax = 37;

  const x = ((lon - lonMin) / (lonMax - lonMin)) * mapWidth;
  const y = ((latMax - lat) / (latMax - latMin)) * mapHeight;
  return { x, y };
}

export default function IndiaMap() {
  const [cityRisks, setCityRisks] = useState([]);
  const [hoveredCity, setHoveredCity] = useState(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getCityRisks()
      .then((data) => {
        setCityRisks(data.cities || []);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  if (!loaded) {
    return (
      <div className="chart-card">
        <h3>🗺️ India Risk Map</h3>
        <div className="loading-placeholder">Loading map data...</div>
      </div>
    );
  }

  return (
    <div className="chart-card india-map-card">
      <h3>🗺️ India Risk Map</h3>
      <p className="chart-subtitle">City-wise predominant risk levels across India</p>

      <div className="map-container">
        <svg viewBox="0 0 500 580" className="india-map-svg">
          {/* India outline (simplified) */}
          <path
            d="M200,20 L230,15 L260,25 L290,20 L320,30 L350,25 L380,35 L410,30 L430,45 L445,65 L460,90 L465,120 L470,150 L465,180 L460,210 L455,240 L450,260 L440,290 L435,310 L440,340 L445,370 L440,400 L430,430 L415,450 L400,465 L380,475 L360,480 L340,490 L310,500 L280,510 L260,520 L240,530 L230,540 L220,550 L210,555 L195,550 L180,540 L170,525 L160,510 L145,490 L130,470 L115,445 L105,420 L95,390 L85,360 L80,330 L75,300 L70,270 L70,240 L75,210 L80,180 L90,155 L100,130 L115,110 L130,90 L145,70 L160,55 L175,40 L190,28 Z"
            fill="rgba(255,255,255,0.03)"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth="1.5"
          />
          
          {/* City dots */}
          {cityRisks.map((city) => {
            const { x, y } = project(city.lat, city.lon);
            const color = RISK_COLORS[city.risk_level] || "#888";
            const isHovered = hoveredCity === city.city;

            return (
              <g
                key={city.city}
                onMouseEnter={() => setHoveredCity(city.city)}
                onMouseLeave={() => setHoveredCity(null)}
                style={{ cursor: "pointer" }}
              >
                {/* Glow effect */}
                <circle cx={x} cy={y} r={isHovered ? 16 : 10} fill={color} opacity={0.15} />
                <circle cx={x} cy={y} r={isHovered ? 10 : 6} fill={color} opacity={0.3} />
                <circle
                  cx={x} cy={y}
                  r={isHovered ? 6 : 4}
                  fill={color}
                  stroke="rgba(255,255,255,0.5)"
                  strokeWidth="1"
                />
                {isHovered && (
                  <g>
                    <rect
                      x={x + 10} y={y - 24}
                      width={Math.max(city.city.length * 8.5 + 16, 90)}
                      height="28"
                      rx="6"
                      fill="rgba(10,10,30,0.92)"
                      stroke={color}
                      strokeWidth="1"
                    />
                    <text
                      x={x + 18} y={y - 6}
                      fill="#fff"
                      fontSize="12"
                      fontFamily="Inter, sans-serif"
                    >
                      {city.city} — {city.risk_level}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="map-legend">
        {Object.entries(RISK_COLORS).map(([level, color]) => (
          <div key={level} className="legend-item">
            <span className="legend-dot" style={{ background: color }} />
            <span>{level}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
