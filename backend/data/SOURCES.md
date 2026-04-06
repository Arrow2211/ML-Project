# Data Sources & Methodology

This document summarizes the origins and processing methods for the datasets used in the Multi-Hazard Risk Prediction System.

## 1. City Metadata (`cities_metadata.json`)
A consolidated list of Indian cities, including their geographic coordinates (latitude/longitude), state, and climate/seismic zone.
- **Sources**: 
  - Standard Indian City Databases (Nshntarora/Indian-Cities-JSON)
  - Geocoded districts and talukas via Nominatim (OSM).
- **Total Coverage**: 301 unique nodes across India.

## 2. Climatological Data (Weather)
Historical weather features used for cyclone and drought indexing.
- **Primary Source**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Data Backend**: ECMWF ERA5 reanalysis data, which incorporates observations from the **India Meteorological Department (IMD)**.
- **Features Captured**: Max Temperature, Total Precipitation, Mean Relative Humidity, Max Wind Speed (10m), Surface Pressure.

## 3. Seismic Data (Earthquakes)
Earthquake frequency data used for seismic risk assessment.
- **Primary Source**: [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)
- **Authority**: Authoritative global catalog also used as a primary reference by India's **National Center for Seismology (NCS)**.
- **Criteria**: Magnitude ≥ 2.5 events within a 200km radius of each city.

## 4. Derived Hazard Indices
- **Cyclone Risk**: Computed using **IMD Cyclone Classification thresholds** (km/h):
  - Depression: 31-49
  - Deep Depression: 50-61
  - Cyclonic Storm: 62-88
  - Severe Cyclone: 89-117
  - Super Cyclone: > 221
- **Drought Index**: Standardized Precipitation Index (SPI) proxy calculated from annual rainfall variability.
- **Risk Level**: Multi-criteria classification (Low/Medium/High) based on weighted contributions of seismic frequency, extreme wind speeds, and rainfall anomalies.
