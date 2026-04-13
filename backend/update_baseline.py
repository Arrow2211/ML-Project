
import os
import sys
import json
import pandas as pd
import time

# Add parent directory to path to import ml module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.data_fetcher import (
    INDIAN_CITIES, 
    fetch_weather_data, 
    fetch_earthquake_data, 
    compute_earthquake_frequency,
    compute_drought_index,
    compute_cyclone_risk,
    assign_risk_label,
    BASELINE_FILE
)

def update_baseline():
    print(f"Loading metadata and current baseline...")
    meta_cities = INDIAN_CITIES
    meta_dict = {c["city"]: c for c in meta_cities}
    
    if os.path.exists(BASELINE_FILE):
        df_base = pd.read_csv(BASELINE_FILE)
        
        # Backfill State/Zone for existing rows if missing or "Unknown"
        print("Backfilling State/Zone for existing rows...")
        for i, row in df_base.iterrows():
            city = row["City"]
            if city in meta_dict:
                if "State" not in df_base.columns or pd.isna(df_base.loc[i, "State"]) or df_base.loc[i, "State"] == "Unknown":
                    df_base.loc[i, "State"] = meta_dict[city].get("state", "Unknown")
                if "Zone" not in df_base.columns or pd.isna(df_base.loc[i, "Zone"]) or df_base.loc[i, "Zone"] == "Unknown":
                    df_base.loc[i, "Zone"] = meta_dict[city].get("zone", "Unknown")
        
        existing_cities = set(df_base["City"].unique())
    else:
        df_base = pd.DataFrame()
        existing_cities = set()
        
    missing_cities = [c for c in meta_cities if c["city"] not in existing_cities]
    
    if not missing_cities:
        print("✅ All cities are already present in the baseline.")
        if not df_base.empty:
            df_base = df_base.sort_values(["State", "City"])
            df_base.to_csv(BASELINE_FILE, index=False)
            print("Sorted and saved existing baseline with state metadata.")
        return

    print(f"Found {len(missing_cities)} missing cities. Starting backfill...")
    
    # Fetch earthquake data once for all new cities
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    print(f"Fetching global earthquake data for {start_date} to {end_date}...")
    eq_df = fetch_earthquake_data(start_date, end_date, min_magnitude=2.5)
    
    new_rows = []
    
    for i, city_info in enumerate(missing_cities):
        city_name = city_info["city"]
        print(f"[{i+1}/{len(missing_cities)}] Processing {city_name} ({city_info.get('state', 'Unknown')})...")
        
        try:
            # 1. Weather data
            weather_df = fetch_weather_data(city_info, start_date, end_date)
            if weather_df.empty:
                print(f"  Warning: No weather data for {city_name}, skipping.")
                continue
                
            # 2. Earthquake frequency
            eq_freq = compute_earthquake_frequency(
                city_info["lat"], city_info["lon"], eq_df, radius_km=200
            )
            weather_df["Earthquake_Frequency"] = eq_freq
            
            # 3. Derived Indices
            rainfall = weather_df["Rainfall"].fillna(0)
            wind = weather_df["Wind_Speed"].fillna(0)
            pressure = weather_df.get("Surface_Pressure")
            
            weather_df["Drought_Index"] = compute_drought_index(rainfall).round(3)
            weather_df["Cyclone_Risk"] = compute_cyclone_risk(
                wind,
                pressure if pressure is not None else None
            ).round(3)
            
            # 4. Aggregation (1 representative row per city)
            agg_map = {
                'Temperature': 'mean',
                'Rainfall': 'sum',
                'Humidity': 'mean',
                'Wind_Speed': 'mean',
                'Latitude': 'first',
                'Longitude': 'first',
                'Earthquake_Frequency': 'max',
                'Drought_Index': 'mean',
                'Cyclone_Risk': 'mean'
            }
            
            df_city_agg = weather_df.groupby('City').agg({k: v for k, v in agg_map.items() if k in weather_df.columns}).reset_index()
            
            # Add state and zone from metadata
            df_city_agg["State"] = city_info.get("state", "Unknown")
            df_city_agg["Zone"] = city_info.get("zone", "Unknown")
            
            # 5. Risk Label
            df_city_agg["Risk_Level"] = df_city_agg.apply(assign_risk_label, axis=1)
            
            new_rows.append(df_city_agg)
            print(f"  Added {city_name}: Risk Level = {df_city_agg['Risk_Level'].iloc[0]}")
            
            # Rate limit
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Failed to process {city_name}: {e}")

    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        df_final = pd.concat([df_base, df_new], ignore_index=True)
        
        # Sort and save
        df_final = df_final.sort_values(["State", "City"])
        df_final.to_csv(BASELINE_FILE, index=False)
        print(f"\nSUCCESS: Added {len(new_rows)} new cities. Baseline now contains {len(df_final)} cities.")
    else:
        # Save backfilled existing data if any
        df_base = df_base.sort_values(["State", "City"])
        df_base.to_csv(BASELINE_FILE, index=False)
        print("\nNo new cities added, but existing cities were updated with state metadata.")


if __name__ == "__main__":
    update_baseline()
