import sys
from geopy.geocoders import Nominatim
import time

cities_to_add = ["Jat", "Islampur", "Palus"]

geolocator = Nominatim(user_agent="disaster_risk_app_v2")

with open("more_cities.txt", "w", encoding="utf-8") as f:
    for city in cities_to_add:
        try:
            location = geolocator.geocode(f"{city}, Maharashtra, India")
            if location:
                f.write(f'    {{"city": "{city}", "state": "Maharashtra", "lat": {round(location.latitude, 4)}, "lon": {round(location.longitude, 4)}, "zone": "inland_west"}},\n')
            time.sleep(1)
        except Exception as e:
            f.write(f'# Failed {city}: {e}\n')
