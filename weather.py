# Game-day weather from the Open-Meteo historical weather API (free, no key)
# Run with: python weather.py
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

# where each school plays its home games (latitude, longitude)
CAMPUS = {
    "ncf": (27.385, -82.561),          # Sarasota, FL
    "keiser": (26.734, -80.120),       # West Palm Beach, FL
    "stthomas": (25.922, -80.252),     # Miami Gardens, FL
    "scad": (32.081, -81.091),         # Savannah, GA
    "southeastern": (28.031, -81.944), # Lakeland, FL
    "fmu": (25.932, -80.268),          # Miami Gardens, FL
    "webber": (27.844, -81.527),       # Babson Park, FL
    "avemaria": (26.334, -81.439),     # Ave Maria, FL
    "warner": (27.876, -81.596),       # Lake Wales, FL
}

URL = "https://archive-api.open-meteo.com/v1/archive"


def get_weather(team, lat, lon, start, end, tries=3):
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }
    for attempt in range(tries):
        response = requests.get(URL, params=params, timeout=30)
        if response.status_code == 200:
            daily = pd.DataFrame(response.json()["daily"])
            daily["location"] = team
            return daily
        print(f"{team}: status {response.status_code}, trying again...")
        time.sleep(5 * (attempt + 1))  # wait a bit longer each time
    return None


if __name__ == "__main__":
    # the archive is a few days behind, so stop 6 days ago
    end = (date.today() - timedelta(days=6)).isoformat()
    frames = []
    for team, (lat, lon) in CAMPUS.items():
        w = get_weather(team, lat, lon, "2023-08-01", end)
        if w is not None:
            frames.append(w)
            print(f"{team}: {len(w)} days")
        time.sleep(1)
    weather = pd.concat(frames).rename(columns={
        "time": "date", "temperature_2m_max": "temp_max_f", "relative_humidity_2m_mean": "humidity_pct",
        "precipitation_sum": "rain_in", "wind_speed_10m_max": "wind_max_mph"})
    os.makedirs("data/raw", exist_ok=True)
    weather.to_csv("data/raw/weather.csv", index=False)
    print("saved data/raw/weather.csv", weather.shape)
