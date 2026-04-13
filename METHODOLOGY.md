# Research Methodology: Data Engineering & Model Architecture

This document details the scientific methodology used in the Multi-Hazard Risk Prediction System.

## 1. Data Sourcing & Integration
The system integrates three primary domains of environmental data:

### 1.1 Meteorological Data (Climatology)
Historical climate data is sourced from the **ECMWF ERA5 reanalysis** via the Open-Meteo API. This dataset is chosen for its multi-decade consistency and high spatial resolution (0.25° grid), effectively capturing localized weather patterns across India.
- **Features**: Max Temperature (°C), Daily Precipitation (mm), Relative Humidity (%), Wind Speed at 10m (km/h), and Surface Pressure (hPa).
- **Aggregation**: Annual averages and totals are computed for the baseline year (2023) to establish local climatological normals.

### 1.2 Seismic Intensity (Earthquake Catalog)
Seismic activity is quantified using the **USGS Earthquake Catalog**, including all recorded events with Magnitude $\geq 2.5$.
- **Spatial Search**: A reverse-radius search (200km) is conducted for each city/district headquarters.
- **Metric**: Annualized Earthquake Frequency is used as a proxy for seismic vulnerability.

## 2. Scientific Indicator Derivation

### 2.1 Drought Index (SPI Proxy)
Standardized Precipitation Index (SPI) is simulated by calculating the percentage anomaly of annual rainfall compared to the national/regional mean for that specific geographic zone.
- **Formula**: $Index_{Drought} = 1 - (Rain_{City} / Mean_{Zone})$
- **Normalization**: Clipped to [0, 1] range, where 1 indicates severe deficit.

### 2.2 Cyclone Risk (IMD Classification)
Cyclone risk is mapped using the **India Meteorological Department (IMD)** wind-speed classification thresholds:
- **Calculation**: A weighted score based on proximity to 10m max wind speeds and pressure anomalies.
- **Classification Mapping**:
    - Low: < 62 km/h (Depressions)
    - Medium: 62 - 117 km/h (Cyclonic Storms)
    - High: > 118 km/h (Severe Cyclonic Storms)

## 3. Ensemble Model Architecture
The system utilizes a **Heterogeneous Ensemble Classifier** with **Soft Voting** logic.

### 3.1 Model Components
1.  **Random Forest (Bagging)**: Excellent at handling multi-modal distributions and providing robust feature importance rankings.
2.  **Gradient Boosting (Boosting)**: Targeted at minimizing residuals in extreme case predictions (e.g., rare super-cyclones).
3.  **SVM (Distance-Based)**: Defines optimal margins between risk classes in high-dimensional feature space.

### 3.2 Feature Interaction
We implemented an interaction term: $Interaction = (Rainfall \times WindSpeed) / 100$. This term acts as a scientific proxy for **Storm Severity**, as high rainfall combined with high wind speed significantly increases the composite damage potential.

## 4. Geographic Zonal Encoding
India is divided into 7 distinct geospatial zones:
- **Coastal (West/East)**: High weighting for cyclone and humidity volatility.
- **Himalayan**: High weighting for seismic frequency and temperature inversion.
- **Inland (North/Central/South/West)**: Focus on rainfall deficit and heatwave anomalies.
