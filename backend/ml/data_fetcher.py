"""
Real data fetcher for the Multi-Hazard Risk Prediction System.
Fetches actual environmental and seismic data from government-authorized sources:
  - Weather: Open-Meteo Historical API (uses ERA5 reanalysis incorporating IMD observations)
  - Earthquakes: USGS Earthquake Catalog API (used by India's NCS / seismo.gov.in)
  - Drought Index: Computed via SPI from precipitation data
  - Cyclone Risk: Computed from wind speed using IMD cyclone classification thresholds
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ─── Indian Cities ─────────────────────────────────────────────────────

INDIAN_CITIES = [
    {"city": "Mumbai", "lat": 19.076, "lon": 72.8777, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Delhi", "lat": 28.6139, "lon": 77.209, "state": "Delhi", "zone": "inland_north"},
    {"city": "Bangalore", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "zone": "inland_south"},
    {"city": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "zone": "coastal_east"},
    {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "zone": "coastal_east"},
    {"city": "Hyderabad", "lat": 17.385, "lon": 78.4867, "state": "Telangana", "zone": "inland_south"},
    {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "zone": "inland_west"},
    {"city": "Pune", "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "zone": "inland_north"},
    {"city": "Lucknow", "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "zone": "inland_north"},
    {"city": "Bhopal", "lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Patna", "lat": 25.6093, "lon": 85.1376, "state": "Bihar", "zone": "inland_north"},
    {"city": "Guwahati", "lat": 26.1445, "lon": 91.7362, "state": "Assam", "zone": "inland_northeast"},
    {"city": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "state": "Odisha", "zone": "coastal_east"},
    {"city": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "zone": "coastal_east"},
    {"city": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366, "state": "Kerala", "zone": "coastal_west"},
    {"city": "Kochi", "lat": 9.9312, "lon": 76.2673, "state": "Kerala", "zone": "coastal_west"},
    {"city": "Surat", "lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "zone": "coastal_west"},
    {"city": "Nagpur", "lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Indore", "lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Coimbatore", "lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Dehradun", "lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "zone": "himalayan"},
    {"city": "Shimla", "lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "zone": "himalayan"},
    {"city": "Gangtok", "lat": 27.3389, "lon": 88.6065, "state": "Sikkim", "zone": "himalayan"},
    {"city": "Imphal", "lat": 24.817, "lon": 93.9368, "state": "Manipur", "zone": "inland_northeast"},
    {"city": "Shillong", "lat": 25.5788, "lon": 91.8933, "state": "Meghalaya", "zone": "inland_northeast"},
    {"city": "Ranchi", "lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "zone": "inland_central"},
    {"city": "Raipur", "lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh", "zone": "inland_central"},
    {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "state": "Chandigarh", "zone": "inland_north"},
    {"city": "Jodhpur", "lat": 26.2389, "lon": 73.0243, "state": "Rajasthan", "zone": "desert"},
    {"city": "Varanasi", "lat": 25.3176, "lon": 83.0068, "state": "Uttar Pradesh", "zone": "inland_north"},
    {"city": "Mangalore", "lat": 12.9141, "lon": 74.856, "state": "Karnataka", "zone": "coastal_west"},
    {"city": "Mysore", "lat": 12.2958, "lon": 76.6394, "state": "Karnataka", "zone": "inland_south"},
    {"city": "Goa", "lat": 15.2993, "lon": 74.124, "state": "Goa", "zone": "coastal_west"},
    {"city": "Jammu", "lat": 32.7266, "lon": 74.857, "state": "Jammu & Kashmir", "zone": "himalayan"},
    {"city": "Srinagar", "lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir", "zone": "himalayan"},
    {"city": "Amritsar", "lat": 31.634, "lon": 74.8723, "state": "Punjab", "zone": "inland_north"},
    {"city": "Agra", "lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh", "zone": "inland_north"},
    {"city": "Kanpur", "lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh", "zone": "inland_north"},
    {"city": "Madurai", "lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Vijayawada", "lat": 16.5062, "lon": 80.648, "state": "Andhra Pradesh", "zone": "coastal_east"},
    {"city": "Tiruchirappalli", "lat": 10.7905, "lon": 78.7047, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Rajkot", "lat": 22.3039, "lon": 70.8022, "state": "Gujarat", "zone": "inland_west"},
    {"city": "Vadodara", "lat": 22.3072, "lon": 73.1812, "state": "Gujarat", "zone": "inland_west"},
    {"city": "Udaipur", "lat": 24.5854, "lon": 73.7125, "state": "Rajasthan", "zone": "inland_west"},
    {"city": "Dharamshala", "lat": 32.219, "lon": 76.3234, "state": "Himachal Pradesh", "zone": "himalayan"},
    {"city": "Rishikesh", "lat": 30.0869, "lon": 78.2676, "state": "Uttarakhand", "zone": "himalayan"},
    {"city": "Dibrugarh", "lat": 27.4728, "lon": 94.912, "state": "Assam", "zone": "inland_northeast"},
    {"city": "Silchar", "lat": 24.8333, "lon": 92.7789, "state": "Assam", "zone": "inland_northeast"},
    {"city": "Cuttack", "lat": 20.4625, "lon": 85.8828, "state": "Odisha", "zone": "coastal_east"},
    {"city": "Navi Mumbai", "lat": 19.0330, "lon": 73.0297, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Thane", "lat": 19.2183, "lon": 72.9781, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Nashik", "lat": 19.9975, "lon": 73.7898, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Aurangabad", "lat": 19.8762, "lon": 75.3433, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Howrah", "lat": 22.5958, "lon": 88.3110, "state": "West Bengal", "zone": "coastal_east"},
    {"city": "Darjeeling", "lat": 27.0410, "lon": 88.2663, "state": "West Bengal", "zone": "himalayan"},
    {"city": "Siliguri", "lat": 26.7271, "lon": 88.3953, "state": "West Bengal", "zone": "himalayan"},
    {"city": "Asansol", "lat": 23.6739, "lon": 86.9524, "state": "West Bengal", "zone": "inland_east"},
    {"city": "Durgapur", "lat": 23.5204, "lon": 87.3119, "state": "West Bengal", "zone": "inland_east"},
    {"city": "Salem", "lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Erode", "lat": 11.3410, "lon": 77.7172, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Tirunelveli", "lat": 8.7139, "lon": 77.7567, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Vellore", "lat": 12.9165, "lon": 79.1325, "state": "Tamil Nadu", "zone": "inland_south"},
    {"city": "Hubli", "lat": 15.3647, "lon": 75.1240, "state": "Karnataka", "zone": "inland_south"},
    {"city": "Belagavi", "lat": 15.8497, "lon": 74.4977, "state": "Karnataka", "zone": "inland_south"},
    {"city": "Davanagere", "lat": 14.4644, "lon": 75.9218, "state": "Karnataka", "zone": "inland_south"},
    {"city": "Jalandhar", "lat": 31.3260, "lon": 75.5762, "state": "Punjab", "zone": "inland_north"},
    {"city": "Ludhiana", "lat": 30.9010, "lon": 75.8573, "state": "Punjab", "zone": "inland_north"},
    {"city": "Patiala", "lat": 30.3398, "lon": 76.3869, "state": "Punjab", "zone": "inland_north"},
    {"city": "Bathinda", "lat": 30.2110, "lon": 74.9455, "state": "Punjab", "zone": "inland_north"},
    {"city": "Gwalior", "lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Jabalpur", "lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Ujjain", "lat": 23.1765, "lon": 75.7885, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Sagar", "lat": 23.8322, "lon": 78.7378, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Rewa", "lat": 24.5373, "lon": 81.3042, "state": "Madhya Pradesh", "zone": "inland_central"},
    {"city": "Kozhikode", "lat": 11.2588, "lon": 75.7804, "state": "Kerala", "zone": "coastal_west"},
    {"city": "Kollam", "lat": 8.8932, "lon": 76.6141, "state": "Kerala", "zone": "coastal_west"},
    {"city": "Thrissur", "lat": 10.5276, "lon": 76.2144, "state": "Kerala", "zone": "coastal_west"},
    {"city": "Jaisalmer", "lat": 26.9157, "lon": 70.9083, "state": "Rajasthan", "zone": "desert"},
    {"city": "Bikaner", "lat": 28.0229, "lon": 73.3119, "state": "Rajasthan", "zone": "desert"},
    {"city": "Kota", "lat": 25.2138, "lon": 75.8648, "state": "Rajasthan", "zone": "inland_west"},
    {"city": "Ajmer", "lat": 26.4499, "lon": 74.6399, "state": "Rajasthan", "zone": "inland_west"},
    {"city": "Pondicherry", "lat": 11.9416, "lon": 79.8083, "state": "Puducherry", "zone": "coastal_east"},
    {"city": "Port Blair", "lat": 11.6234, "lon": 92.7265, "state": "Andaman & Nicobar Islands", "zone": "island"},
    {"city": "Leh", "lat": 34.1526, "lon": 77.5771, "state": "Ladakh", "zone": "himalayan"},
    {"city": "Kargil", "lat": 34.5539, "lon": 76.1349, "state": "Ladakh", "zone": "himalayan"},
    {"city": "Nainital", "lat": 29.3919, "lon": 79.4542, "state": "Uttarakhand", "zone": "himalayan"},
    {"city": "Mussoorie", "lat": 30.4598, "lon": 78.0776, "state": "Uttarakhand", "zone": "himalayan"},
    {"city": "Ahmednagar", "lat": 19.1628, "lon": 74.858, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Akola", "lat": 20.7618, "lon": 77.1921, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Amravati", "lat": 21.1545, "lon": 77.6443, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Chhatrapati Sambhajinagar", "lat": 19.8773, "lon": 75.339, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Beed", "lat": 18.9918, "lon": 75.9098, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Bhandara", "lat": 21.1226, "lon": 79.7945, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Buldhana", "lat": 20.5628, "lon": 76.4087, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Chandrapur", "lat": 20.0968, "lon": 79.5045, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Dhule", "lat": 21.1305, "lon": 74.4901, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Gadchiroli", "lat": 19.7591, "lon": 80.1623, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Gondia", "lat": 21.4552, "lon": 80.1963, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Hingoli", "lat": 19.5431, "lon": 77.1739, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Jalgaon", "lat": 20.8429, "lon": 75.5261, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Jalna", "lat": 19.9188, "lon": 75.8709, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Kolhapur", "lat": 16.7028, "lon": 74.2405, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Latur", "lat": 18.3553, "lon": 76.7549, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Nanded", "lat": 19.0943, "lon": 77.4833, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Nandurbar", "lat": 21.5142, "lon": 74.5406, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Dharashiv", "lat": 18.1698, "lon": 76.118, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Palghar", "lat": 19.7572, "lon": 73.0931, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Parbhani", "lat": 19.2902, "lon": 76.6026, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Raigad", "lat": 18.4928, "lon": 73.1381, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Ratnagiri", "lat": 17.2826, "lon": 73.457, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Sangli", "lat": 17.1727, "lon": 74.5868, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Satara", "lat": 17.6361, "lon": 74.2983, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Sindhudurg", "lat": 16.1357, "lon": 73.6522, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Solapur", "lat": 17.8499, "lon": 75.2763, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Wardha", "lat": 20.8256, "lon": 78.6131, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Washim", "lat": 20.2874, "lon": 77.237, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Yavatmal", "lat": 20.0639, "lon": 78.3566, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Kalyan", "lat": 19.2397, "lon": 73.1366, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Ulhasnagar", "lat": 19.2236, "lon": 73.1672, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Bhiwandi", "lat": 19.3026, "lon": 73.0588, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Vasai", "lat": 19.3428, "lon": 72.8054, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Virar", "lat": 19.4498, "lon": 72.8121, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Mira Bhayandar", "lat": 19.3184, "lon": 72.8992, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Panvel", "lat": 18.9895, "lon": 73.1222, "state": "Maharashtra", "zone": "coastal_west"},
    {"city": "Malegaon", "lat": 20.5576, "lon": 74.5247, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Ichalkaranji", "lat": 16.6959, "lon": 74.4556, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Pimpri-Chinchwad", "lat": 18.6279, "lon": 73.801, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Baramati", "lat": 18.2199, "lon": 74.4534, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Karad", "lat": 17.2852, "lon": 74.1822, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Pandharpur", "lat": 17.7256, "lon": 75.3002, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Chiplun", "lat": 17.5253, "lon": 73.5156, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Kudal", "lat": 16.0172, "lon": 73.6781, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Kankavli", "lat": 16.3742, "lon": 73.7452, "state": "Maharashtra", "zone": "inland_west"},
    {"city": "Tuljapur", "lat": 20.9824, "lon": 77.655, "state": "Maharashtra", "zone": "inland_central"},
    {"city": "Ausa", "state": "Maharashtra", "lat": 18.251, "lon": 76.501, "zone": "inland_central"},
    {"city": "Udgir", "state": "Maharashtra", "lat": 18.3921, "lon": 77.1196, "zone": "inland_central"},
    {"city": "Khamgaon", "state": "Maharashtra", "lat": 20.6502, "lon": 76.5592, "zone": "inland_central"},
    {"city": "Malkapur", "state": "Maharashtra", "lat": 20.892, "lon": 76.204, "zone": "inland_central"},
    {"city": "Karjat", "state": "Maharashtra", "lat": 18.9128, "lon": 73.3228, "zone": "inland_west"},
    {"city": "Khopoli", "state": "Maharashtra", "lat": 18.7877, "lon": 73.3438, "zone": "inland_west"},
    {"city": "Alibaug", "state": "Maharashtra", "lat": 18.6498, "lon": 72.8765, "zone": "coastal_west"},
    {"city": "Mahabaleshwar", "state": "Maharashtra", "lat": 17.9243, "lon": 73.6576, "zone": "inland_west"},
    {"city": "Lonavala", "state": "Maharashtra", "lat": 18.7504, "lon": 73.4069, "zone": "inland_west"},
    {"city": "Igatpuri", "state": "Maharashtra", "lat": 19.6944, "lon": 73.5639, "zone": "inland_west"},
    {"city": "Bhusawal", "state": "Maharashtra", "lat": 21.0457, "lon": 75.7808, "zone": "inland_west"},
    {"city": "Amalner", "state": "Maharashtra", "lat": 21.0493, "lon": 75.0573, "zone": "inland_west"},
    {"city": "Shirpur", "state": "Maharashtra", "lat": 20.4309, "lon": 76.1858, "zone": "inland_west"},
    {"city": "Shrirampur", "state": "Maharashtra", "lat": 19.642, "lon": 74.7007, "zone": "inland_west"},
    {"city": "Kopargaon", "state": "Maharashtra", "lat": 19.837, "lon": 74.4829, "zone": "inland_west"},
    {"city": "Sangamner", "state": "Maharashtra", "lat": 19.4906, "lon": 74.2467, "zone": "inland_west"},
    {"city": "Manchar", "state": "Maharashtra", "lat": 19.0033, "lon": 73.9427, "zone": "inland_west"},
    {"city": "Junnar", "state": "Maharashtra", "lat": 19.2007, "lon": 73.9768, "zone": "inland_west"},
    {"city": "Ambajogai", "state": "Maharashtra", "lat": 18.7318, "lon": 76.3855, "zone": "inland_west"},
    {"city": "Parli", "state": "Maharashtra", "lat": 18.8486, "lon": 76.5297, "zone": "inland_west"},
    {"city": "Sawantwadi", "state": "Maharashtra", "lat": 15.908, "lon": 73.8205, "zone": "coastal_west"},
    {"city": "Khed", "state": "Maharashtra", "lat": 17.7126, "lon": 73.4107, "zone": "inland_west"},
    {"city": "Dapoli", "state": "Maharashtra", "lat": 17.758, "lon": 73.1887, "zone": "coastal_west"},
    {"city": "Guhagar", "state": "Maharashtra", "lat": 17.4838, "lon": 73.1911, "zone": "coastal_west"},
    {"city": "Shrigonda", "state": "Maharashtra", "lat": 18.6764, "lon": 74.6709, "zone": "inland_west"},
    {"city": "Matheran", "state": "Maharashtra", "lat": 18.9902, "lon": 73.27, "zone": "inland_west"},
    {"city": "Panchgani", "state": "Maharashtra", "lat": 17.924, "lon": 73.7993, "zone": "inland_west"},
    {"city": "Wai", "state": "Maharashtra", "lat": 17.9424, "lon": 73.9193, "zone": "inland_west"},
    {"city": "Dahanu", "state": "Maharashtra", "lat": 19.9887, "lon": 72.7338, "zone": "coastal_west"},
    {"city": "Jat", "state": "Maharashtra", "lat": 17.0719, "lon": 75.3331, "zone": "inland_west"},
    {"city": "Islampur", "state": "Maharashtra", "lat": 21.0396, "lon": 76.4052, "zone": "inland_west"},
    {"city": "Palus", "state": "Maharashtra", "lat": 17.045, "lon": 74.4366, "zone": "inland_west"}
]

# Cache files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "dataset_cache.csv")
BASELINE_FILE = os.path.join(DATA_DIR, "baseline_data.csv")


# ─── Weather Data (Open-Meteo Historical API) ─────────────────────────

def fetch_weather_data(city_info, start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API.
    Source: Open-Meteo (uses ERA5 reanalysis data from ECMWF, incorporating 
    India Meteorological Department observations).
    
    API docs: https://open-meteo.com/en/docs/historical-weather-api
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "surface_pressure_mean",
        ]),
        "timezone": "Asia/Kolkata",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        if not dates:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "City": city_info["city"],
            "Latitude": city_info["lat"],
            "Longitude": city_info["lon"],
            "Date": dates,
            "Temperature": daily.get("temperature_2m_max"),
            "Rainfall": daily.get("precipitation_sum"),
            "Humidity": daily.get("relative_humidity_2m_mean"),
            "Wind_Speed": daily.get("wind_speed_10m_max"),
            "Surface_Pressure": daily.get("surface_pressure_mean"),
        })
        
        return df
        
    except Exception as e:
        print(f"  ⚠ Weather fetch failed for {city_info['city']}: {e}")
        return pd.DataFrame()


# ─── Earthquake Data (USGS Earthquake Catalog) ────────────────────────

def fetch_earthquake_data(start_date, end_date, min_magnitude=3.0):
    """
    Fetch earthquake data from USGS Earthquake Catalog API for the India region.
    Source: US Geological Survey (earthquake.usgs.gov) — authoritative global 
    catalog also referenced by India's National Center for Seismology (seismo.gov.in).
    
    API docs: https://earthquake.usgs.gov/fdsnws/event/1/
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv",
        "starttime": start_date,
        "endtime": end_date,
        "minlatitude": 6.0,
        "maxlatitude": 38.0,
        "minlongitude": 68.0,
        "maxlongitude": 98.0,
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        from io import StringIO
        eq_df = pd.read_csv(StringIO(resp.text))
        
        print(f"  📊 Fetched {len(eq_df)} earthquakes (M≥{min_magnitude}) in India region")
        return eq_df
        
    except Exception as e:
        print(f"  ⚠ Earthquake fetch failed: {e}")
        return pd.DataFrame()


def compute_earthquake_frequency(city_lat, city_lon, eq_df, radius_km=200):
    """
    Count earthquakes within a radius of a city.
    Uses Haversine approximation for distance.
    """
    if eq_df.empty or "latitude" not in eq_df.columns:
        return 0
    
    # Haversine distance approximation
    lat_diff = np.radians(eq_df["latitude"] - city_lat)
    lon_diff = np.radians(eq_df["longitude"] - city_lon)
    a = (np.sin(lat_diff / 2) ** 2 +
         np.cos(np.radians(city_lat)) * np.cos(np.radians(eq_df["latitude"])) *
         np.sin(lon_diff / 2) ** 2)
    distances_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    
    return int((distances_km <= radius_km).sum())


# ─── Derived Indices ───────────────────────────────────────────────────

def compute_drought_index(rainfall_series):
    """
    Compute Standardized Precipitation Index (SPI) as drought indicator.
    Lower rainfall → higher drought index (0 = no drought, 1 = extreme drought).
    Based on standard meteorological SPI methodology.
    """
    if rainfall_series.empty or rainfall_series.std() == 0:
        return pd.Series(0.3, index=rainfall_series.index)
    
    mean_rain = rainfall_series.mean()
    std_rain = rainfall_series.std()
    
    # SPI: negative = dry, positive = wet
    spi = (rainfall_series - mean_rain) / std_rain
    
    # Convert to 0-1 drought index: more negative SPI = higher drought
    drought = np.clip((-spi + 2) / 4, 0, 1)
    return drought


def compute_cyclone_risk(wind_speed_series, pressure_series=None):
    """
    Compute cyclone risk based on IMD cyclone classification thresholds.
    
    IMD classification (wind speed in km/h):
      - Low Pressure: < 31
      - Depression: 31-49
      - Deep Depression: 50-61
      - Cyclonic Storm: 62-88
      - Severe Cyclonic Storm: 89-117
      - Very Severe: 118-166
      - Extremely Severe: 167-221
      - Super Cyclone: > 221
    
    Source: India Meteorological Department classification system
    """
    risk = pd.Series(0.0, index=wind_speed_series.index)
    
    risk = np.where(wind_speed_series >= 221, 1.0,
           np.where(wind_speed_series >= 167, 0.9,
           np.where(wind_speed_series >= 118, 0.8,
           np.where(wind_speed_series >= 89, 0.65,
           np.where(wind_speed_series >= 62, 0.5,
           np.where(wind_speed_series >= 50, 0.35,
           np.where(wind_speed_series >= 31, 0.2,
           np.where(wind_speed_series >= 20, 0.1, 0.05))))))))
    
    # Add pressure-based adjustment if available
    if pressure_series is not None:
        # Lower pressure → higher cyclone risk
        pressure_factor = np.clip((1013 - pressure_series) / 50, 0, 0.3)
        risk = np.clip(risk + pressure_factor, 0, 1)
    
    return pd.Series(risk, index=wind_speed_series.index)


# ─── Risk Label Assignment ────────────────────────────────────────────

def assign_risk_label(row):
    """Assign risk label based on feature thresholds."""
    score = 0
    
    # Rainfall contribution
    if row["Rainfall"] > 100:
        score += 3
    elif row["Rainfall"] > 40:
        score += 1.5
    
    # Wind speed contribution
    if row["Wind_Speed"] > 60:
        score += 2.5
    elif row["Wind_Speed"] > 35:
        score += 1.5
    elif row["Wind_Speed"] > 20:
        score += 0.5
    
    # Earthquake frequency contribution
    if row["Earthquake_Frequency"] > 15:
        score += 3
    elif row["Earthquake_Frequency"] > 5:
        score += 1.5
    elif row["Earthquake_Frequency"] > 0:
        score += 0.5
    
    # Cyclone risk
    score += row["Cyclone_Risk"] * 4
    
    # Drought index
    if row["Drought_Index"] > 0.7:
        score += 2
    elif row["Drought_Index"] > 0.4:
        score += 1
    
    # Temperature extremes
    if row["Temperature"] > 42 or row["Temperature"] < 0:
        score += 1.5
    elif row["Temperature"] > 38:
        score += 0.5
    
    # Humidity extremes
    if row["Humidity"] > 90:
        score += 1
    
    if score >= 6:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


# ─── Main Dataset Builder ─────────────────────────────────────────────

def build_dataset(start_date="2023-01-01", end_date="2023-12-31", use_cache=True):
    """
    Build the multi-hazard dataset by fetching real data from government sources.
    
    Data sources:
      1. Open-Meteo Historical API (weather — ERA5/IMD reanalysis)
      2. USGS Earthquake Catalog (seismic — earthquake.usgs.gov)
      3. Derived: Drought Index (SPI from rainfall)
      4. Derived: Cyclone Risk (IMD wind speed classification)
    
    Args:
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)
        use_cache: If True, return cached dataset if available
    
    Returns:
        pd.DataFrame with all features and risk labels
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(CACHE_FILE):
        print("📁 Loading cached dataset...")
        df = pd.read_csv(CACHE_FILE)
        if len(df) > 0:
            print(f"  ✅ Loaded {len(df)} rows from cache")
            return df
    
    # Fallback to committed baseline if cache is missing (useful for first deploy on Render)
    if use_cache and os.path.exists(BASELINE_FILE):
        print(f"📁 Cache missing. Loading baseline dataset for instant start...")
        df = pd.read_csv(BASELINE_FILE)
        if len(df) > 0:
            print(f"  ✅ Loaded {len(df)} cities from baseline")
            return df
    
    print(f"🌐 Fetching real data from government sources ({start_date} to {end_date})...")
    print(f"   Sources: Open-Meteo (ERA5/IMD), USGS Earthquake Catalog")
    
    # 1. Fetch earthquake data (one call for all of India)
    print("\n🔬 Fetching earthquake data from USGS...")
    eq_df = fetch_earthquake_data(start_date, end_date, min_magnitude=2.5)
    
    # 2. Fetch weather for each city
    all_city_data = []
    total_cities = len(INDIAN_CITIES)
    
    for i, city_info in enumerate(INDIAN_CITIES):
        print(f"\n🌤️  [{i+1}/{total_cities}] Fetching weather for {city_info['city']}...")
        
        weather_df = fetch_weather_data(city_info, start_date, end_date)
        
        if weather_df.empty:
            print(f"  ⚠ No data returned for {city_info['city']}, skipping")
            continue
        
        # 3. Compute earthquake frequency for this city
        eq_freq = compute_earthquake_frequency(
            city_info["lat"], city_info["lon"], eq_df, radius_km=200
        )
        weather_df["Earthquake_Frequency"] = eq_freq
        print(f"  🔬 Earthquakes within 200km: {eq_freq}")
        
        # 4. Compute derived indices
        rainfall = weather_df["Rainfall"].fillna(0)
        wind = weather_df["Wind_Speed"].fillna(0)
        pressure = weather_df.get("Surface_Pressure")
        
        weather_df["Drought_Index"] = compute_drought_index(rainfall).round(3)
        weather_df["Cyclone_Risk"] = compute_cyclone_risk(
            wind,
            pressure if pressure is not None else None
        ).round(3)
        
        # Drop the intermediate pressure column
        weather_df = weather_df.drop(columns=["Surface_Pressure"], errors="ignore")
        
        all_city_data.append(weather_df)
        
        # Rate limiting: Open-Meteo recommends <1 req/sec for free tier
        time.sleep(0.3)
    
    if not all_city_data:
        print("❌ No data fetched. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Combine all city data
    df = pd.concat(all_city_data, ignore_index=True)
    
    # Handle any remaining missing values for label assignment
    df_for_labels = df.copy()
    for col in ["Temperature", "Rainfall", "Humidity", "Wind_Speed"]:
        df_for_labels[col] = df_for_labels[col].fillna(df_for_labels[col].median())
    
    # 5. Assign risk labels
    print("\n🏷️  Assigning risk labels...")
    df["Risk_Level"] = df_for_labels.apply(assign_risk_label, axis=1)
    
    # 6. Cache the dataset
    df.to_csv(CACHE_FILE, index=False)
    
    print(f"\n✅ Dataset built: {len(df)} rows across {df['City'].nunique()} cities")
    print(f"   Risk distribution: {df['Risk_Level'].value_counts().to_dict()}")
    print(f"   Cached to: {CACHE_FILE}")
    
    return df


def get_city_list():
    """Return list of available cities with coordinates and states."""
    return [{"city": c["city"], "state": c.get("state", "Unknown"), "lat": c["lat"], "lon": c["lon"], "zone": c["zone"]} for c in INDIAN_CITIES]


def clear_cache():
    """Clear the cached dataset to force re-fetch."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("🗑️  Cache cleared")


if __name__ == "__main__":
    df = build_dataset("2023-01-01", "2023-12-31")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample:\n{df.head()}")
