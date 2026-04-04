import sys
from geopy.geocoders import Nominatim
import time

cities_to_add = [
    "Karjat", "Khopoli", "Alibaug", "Mahabaleshwar", "Lonavala", 
    "Igatpuri", "Bhusawal", "Amalner", "Shirpur", "Shrirampur", 
    "Kopargaon", "Sangamner", "Manchar", "Junnar", "Ambajogai", 
    "Parli", "Sawantwadi", "Khed", "Dapoli", "Guhagar",
    "Shrigonda", "Matheran", "Panchgani", "Wai", "Dahanu"
]

geolocator = Nominatim(user_agent="disaster_risk_app")

with open("extra_cities.txt", "w", encoding="utf-8") as f:
    for city in cities_to_add:
        try:
            location = geolocator.geocode(f"{city}, Maharashtra, India")
            if location:
                zone = "inland_west"
                if city in ["Alibaug", "Dapoli", "Guhagar", "Sawantwadi", "Dahanu"]:
                    zone = "coastal_west"
                f.write(f'    {{"city": "{city}", "state": "Maharashtra", "lat": {round(location.latitude, 4)}, "lon": {round(location.longitude, 4)}, "zone": "{zone}"}},\n')
            time.sleep(1)
        except Exception as e:
            f.write(f'# Failed {city}: {e}\n')
