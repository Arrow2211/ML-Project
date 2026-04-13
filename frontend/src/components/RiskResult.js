"use client";

export default function RiskResult({ result }) {
  if (!result) return null;

  const riskColors = {
    High: { bg: "rgba(255, 75, 75, 0.15)", border: "#ff4b4b", text: "#ff4b4b", glow: "0 0 40px rgba(255,75,75,0.3)" },
    Medium: { bg: "rgba(255, 183, 27, 0.15)", border: "#ffb71b", text: "#ffb71b", glow: "0 0 40px rgba(255,183,27,0.3)" },
    Low: { bg: "rgba(0, 230, 118, 0.15)", border: "#00e676", text: "#00e676", glow: "0 0 40px rgba(0,230,118,0.3)" },
  };

  const style = riskColors[result.risk_level] || riskColors.Low;

  const topContributions = Object.entries(result.feature_contributions || {}).slice(0, 6);

  return (
    <div className="risk-result" style={{ "--risk-color": style.text, "--risk-bg": style.bg, "--risk-glow": style.glow }}>
      <div className="result-header">
        <h2>Prediction Result</h2>
        <span className="city-name">{result.city}</span>
      </div>

      <div className="risk-badge-container">
        <div
          className="risk-badge"
          style={{ background: style.bg, borderColor: style.border, color: style.text, boxShadow: style.glow }}
        >
          <span className="risk-icon">
            {result.risk_level === "High" ? "🔴" : result.risk_level === "Medium" ? "🟡" : "🟢"}
          </span>
          <span className="risk-text">{result.risk_level} Risk</span>
        </div>
      </div>

      {/* Probabilities */}
      <div className="probabilities">
        {Object.entries(result.probabilities || {}).map(([level, prob]) => (
          <div key={level} className="prob-item">
            <span className="prob-label">{level}</span>
            <div className="prob-bar-track">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${prob}%`,
                  background: riskColors[level]?.text || "#888",
                }}
              />
            </div>
            <span className="prob-value">{prob}%</span>
          </div>
        ))}
      </div>

      {/* Feature Contributions */}
      <div className="contributions-section">
        <h3>📊 Feature Contributions</h3>
        <div className="contributions-list">
          {topContributions.map(([feature, value], i) => (
            <div key={feature} className="contribution-item" style={{ animationDelay: `${i * 0.08}s` }}>
              <div className="contribution-header">
                <span className="contribution-name">{feature.replace(/_/g, " ")}</span>
                <span className="contribution-value">{value}%</span>
              </div>
              <div className="contribution-bar-track">
                <div
                  className="contribution-bar-fill"
                  style={{
                    width: `${value}%`,
                    background: `linear-gradient(90deg, ${style.text}, ${style.text}88)`,
                    animationDelay: `${i * 0.08}s`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Explanation */}
      {result.explanation && (
        <div className="explanation-box">
          <h3>💡 Explanation</h3>
          <p>{result.explanation}</p>
        </div>
      )}

      {/* Model Consensus */}
      {result.model_predictions && (
        <div className="consensus-section">
          <div className="consensus-header">
             <h3>🤝 Model Decision Consensus</h3>
             {result.individual_accuracies && (
               <div className="best-model-hint">
                 🏆 {Object.entries(result.individual_accuracies).reduce((a, b) => a[1] > b[1] ? a : b)[0]} is Top Performer
               </div>
             )}
          </div>
          <div className="consensus-grid">
            {Object.entries(result.model_predictions).map(([model, verdict]) => {
              const vStyle = riskColors[verdict] || riskColors.Low;
              const rgb = verdict === "High" ? "255, 75, 75" : verdict === "Medium" ? "255, 183, 27" : "0, 230, 118";
              
              // Determine if this is the 'best' model
              const isBest = result.individual_accuracies && 
                             model === Object.entries(result.individual_accuracies).reduce((a, b) => a[1] > b[1] ? a : b)[0];

              return (
                <div 
                  key={model} 
                  className={`consensus-card active ${isBest ? 'best-card' : ''}`} 
                  style={{ "--consensus-color": vStyle.text, "--consensus-color-rgb": rgb }}
                >
                  {isBest && <span className="best-badge">TOP PERFORMER</span>}
                  <span className="model-name">{model}</span>
                  <span className="model-verdict">{verdict}</span>
                </div>
              );
            })}
          </div>
          <div className="consensus-footer">
            Outcome verified across ensemble profiles
          </div>
        </div>
      )}
    </div>
  );
}
