
import os
import sys
import json
import time
import requests
from geopy.geocoders import Nominatim

# Add parent directory to path to import ml module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_fetcher import (
    INDIAN_CITIES, 
    CITIES_METADATA_FILE,
    load_indian_cities
)
from update_baseline import update_baseline

DISTRICTS_URL = "https://cdn.jsdelivr.net/gh/sab99r/Indian-States-And-Districts@master/states-and-districts.json"
CITIES_URL = "https://cdn.jsdelivr.net/gh/nshntarora/Indian-Cities-JSON@master/cities.json"

def get_zone(lat, lon):
    """Determine a rough geographic zone for hazard classification."""
    if lat > 28.5:
        return "himalayan"
    if lon < 73.5:
        return "coastal_west"
    if lon > 85.0:
        return "coastal_east" if lat < 22 else "inland_northeast"
    if lat < 15:
        return "inland_south"
    if lon < 77.5:
        return "inland_west"
    if lat > 26.0:
        return "inland_north"
    return "inland_central"

def expand_nationwide():
    print("Starting Nationwide Expansion (Districts + Major Cities)...")
    
    geolocator = Nominatim(user_agent="disaster_risk_research_expansion_v2")
    
    # 1. Load existing
    existing_cities = load_indian_cities()
    existing_names = {c["city"].lower() for c in existing_cities}
    
    new_cities = list(existing_cities)
    cities_added = 0
    
    # 2. Fetch Major Cities (High priority)
    print("\nFetching Tier 1/2 Cities catalog via jsDelivr...")
    try:
        resp = requests.get(CITIES_URL, timeout=30)
        resp.raise_for_status()
        remote_data = resp.json()
        
        for c in remote_data:
            name = c.get("name") or c.get("city")
            if name and name.lower() not in existing_names:
                # Check for lat/lng with various possible keys
                lat_str = c.get("lat") or c.get("latitude")
                lon_str = c.get("lng") or c.get("lon") or c.get("longitude")
                
                if lat_str and lon_str:
                    try:
                        lat, lon = float(lat_str), float(lon_str)
                        state = c.get("state", "Unknown")
                        zone = get_zone(lat, lon)
                        new_cities.append({
                            "city": name,
                            "state": state,
                            "lat": round(lat, 4),
                            "lon": round(lon, 4),
                            "zone": zone
                        })
                        existing_names.add(name.lower())
                        cities_added += 1
                    except ValueError:
                        continue
        print(f"  Successfully gathered {cities_added} major cities from remote catalog.")
    except Exception as e:
        print(f"  Warning: Failed to fetch cities JSON: {e}")

    # 3. Fetch All Districts (Coverage priority)
    print("\nFetching Complete District Catalog via jsDelivr (sab99r source)...")
    districts_added = 0
    try:
        resp = requests.get(DISTRICTS_URL, timeout=30)
        resp.raise_for_status()
        # sab99r structure: {"states": [{"state": "StateName", "districts": ["Dist1", ...]}, ...]}
        states_data = resp.json().get("states", [])
        
        for state_obj in states_data:
            state_name = state_obj["state"]
            districts = state_obj["districts"]
            
            print(f"  Processing state: {state_name} ({len(districts)} districts)")
            for dist_name in districts:
                if dist_name.lower() not in existing_names:
                    print(f"    Geocoding district: {dist_name}...")
                    try:
                        query = f"{dist_name}, {state_name}, India"
                        location = geolocator.geocode(query, timeout=10)
                        
                        if not location:
                            location = geolocator.geocode(f"{dist_name}, India", timeout=10)
                        
                        if location:
                            lat, lon = location.latitude, location.longitude
                            if 8.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0:
                                zone = get_zone(lat, lon)
                                new_cities.append({
                                    "city": dist_name,
                                    "state": state_name,
                                    "lat": round(lat, 4),
                                    "lon": round(lon, 4),
                                    "zone": zone
                                })
                                existing_names.add(dist_name.lower())
                                districts_added += 1
                                time.sleep(1.2)
                            else:
                                print(f"      Warning: Coordinates outside India range for {dist_name}: {lat}, {lon}")
                        else:
                            print(f"      Warning: Could not geocode {dist_name}")
                    except Exception as ge:
                        print(f"      Error geocoding {dist_name}: {ge}")
                        time.sleep(2)
        
        print(f"\n  Added {districts_added} new districts via geocoding.")
    except Exception as e:
        print(f"  Warning: Failed to fetch districts JSON: {e}")

    # 4. Save expanded metadata
    if districts_added > 0 or cities_added > 0:
        print(f"\nSaving expanded metadata ({len(new_cities)} total locations)...")
        with open(CITIES_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(new_cities, f, indent=4)
        
        print("Metadata expansion complete.")
        
        # 5. Trigger Baseline Update for all new cities
        print("\nStarting Hazard Data Synchronization for the new nationwide dataset...")
        update_baseline()
    else:
        print("\nNo new locations were added. System is already comprehensive.")

if __name__ == "__main__":
    expand_nationwide()
