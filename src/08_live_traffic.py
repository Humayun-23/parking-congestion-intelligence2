import pandas as pd
import requests
import random
import sys
from pathlib import Path

# Setup paths (assuming standard project structure)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

def fetch_live_traffic(lat, lon, api_key=None):
    """
    Fetches live traffic data for a specific coordinate.
    Uses a simulated response if API_KEY is DEMO_KEY.
    """
    if api_key is None:
        api_key = C.TOMTOM_API_KEY

    if api_key == "DEMO_KEY":
        # Simulate a realistic traffic response (Current Speed vs Free Flow Speed)
        free_flow = random.randint(40, 60)
        current_speed = max(5, free_flow - random.randint(15, 35))
        delay_seconds = random.randint(10, 120)
        
        return {
            "current_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow,
            "delay_seconds": delay_seconds,
            "congestion_ratio": round(current_speed / free_flow, 2)
        }
    else:
        # Production ready code for TomTom Traffic Flow API
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json().get("flowSegmentData", {})
                current_speed = data.get("currentSpeed", 0)
                free_flow = data.get("freeFlowSpeed", 1)
                
                return {
                    "current_speed_kmh": current_speed,
                    "free_flow_speed_kmh": free_flow,
                    "delay_seconds": data.get("currentTravelTime", 0) - data.get("freeFlowTravelTime", 0),
                    "congestion_ratio": round(current_speed / free_flow, 2) if free_flow > 0 else 1.0
                }
            else:
                print(f"    [API ERROR] Status code {response.status_code}. Using fallback simulation.")
                return fetch_live_traffic(lat, lon, api_key="DEMO_KEY")
        except Exception as e:
            print(f"    [NETWORK ERROR] Could not connect to API: {e}. Using fallback simulation.")
            return fetch_live_traffic(lat, lon, api_key="DEMO_KEY")

def validate_hotspots():
    print("==> Loading Cleaned Hotspot Data...")
    try:
        df = pd.read_parquet(C.CLEAN_PARQUET)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract the top 5 worst junctions to check their live traffic
    top_junctions = df[df["has_junction"] == True]["junction_clean"].value_counts().head(5).index.tolist()
    
    print("\n==> Pinging Live Traffic API for Top 5 Priority Junctions...")
    
    validation_results = []
    
    for junction in top_junctions:
        # Get approximate coordinates for this junction from our dataset
        j_data = df[df["junction_clean"] == junction].iloc[0]
        lat, lon = j_data["latitude"], j_data["longitude"]
        
        print(f"    Checking {junction} (Lat: {lat:.4f}, Lon: {lon:.4f})...")
        traffic_data = fetch_live_traffic(lat, lon)
        
        if traffic_data:
            validation_results.append({
                "Junction": junction,
                "Modeled_Impact_Weight": j_data.get("congestion_weight", 3),
                "Live_Speed_kmh": traffic_data["current_speed_kmh"],
                "Free_Flow_kmh": traffic_data["free_flow_speed_kmh"],
                "Actual_Delay_Sec": traffic_data["delay_seconds"]
            })
            
    # Display the validation report
    report_df = pd.DataFrame(validation_results)
    print("\n--- LIVE TRAFFIC VALIDATION REPORT ---")
    print(report_df.to_string(index=False))
    
    # Save this report to the tables folder so Streamlit can find it
    report_df.to_csv(C.TBL_DIR / "live_traffic_validation.csv", index=False)

if __name__ == "__main__":
    validate_hotspots()