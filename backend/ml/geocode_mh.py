import time
from geopy.geocoders import Nominatim

districts = [
    "Ahmednagar", "Akola", "Amravati", "Chhatrapati Sambhajinagar", "Beed", 
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", 
    "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Nanded", 
    "Nandurbar", "Dharashiv", "Palghar", "Parbhani", "Raigad", "Ratnagiri", 
    "Sangli", "Satara", "Sindhudurg", "Solapur", "Wardha", "Washim", "Yavatmal",
    # adding some major talukas too
    "Kalyan", "Ulhasnagar", "Bhiwandi", "Vasai", "Virar", "Mira Bhayandar", "Panvel", 
    "Malegaon", "Ichalkaranji", "Pimpri-Chinchwad", "Baramati", "Karad", "Pandharpur", 
    "Chiplun", "Kudal", "Kankavli", "Tuljapur", "Ausa", "Udgir", "Khamgaon", "Malkapur"
]

geolocator = Nominatim(user_agent="mh_geocoder_agent")
results = []

file_path = r'c:\Users\athar\OneDrive\Desktop\ML Project\backend\ml\data_fetcher.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for d in districts:
    if f'"city": "{d}"' in content: 
        print(f"Skipping {d}, already implicitly there.")
        continue
    try:
        location = geolocator.geocode(f"{d}, Maharashtra, India")
        if location:
            lat = round(location.latitude, 4)
            lon = round(location.longitude, 4)
            zone = 'coastal_west' if lon < 73.5 else ('inland_west' if lon < 76.0 else 'inland_central')
            results.append(f'    {{"city": "{d}", "lat": {lat}, "lon": {lon}, "zone": "{zone}"}},')
            print(f"Geocoded: {d} -> {lat}, {lon}")
        else:
            print(f"Not found: {d}")
    except Exception as e:
        print(f"Geocode error for {d}: {e}")
    time.sleep(1) # Be nice to Nominatim API (1 request per second)

if results:
    new_cities_str = '\n'.join(results) + '\n'
    insert_marker = ']\n\n# Cache directory'
    if insert_marker in content:
        content = content.replace(insert_marker, new_cities_str + insert_marker)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nSUCCESS: Added {len(results)} new locations to data_fetcher.py")
    else:
        print("ERROR: Could not find insert marker.")
else:
    print("No new locations to add.")
