import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

# ============================================================
# STATION ID — CHANGE THIS
# ============================================================
# @545140 = "NED University City Campus" — confirmed unreliable
# (shows "no data" / stopped reporting on IQAir & AQICN's own map).
# That's why "aqi" kept coming back as None even though the request
# itself succeeded (no crash, just an empty reading).
#
# Go to https://aqicn.org/map/karachi/ , click a pin near Saddar that's
# CURRENTLY showing a live number (not a dash), open its station page,
# and copy the number from its URL (aqicn.org/station/@XXXXX). Put that
# number below.
AQICN_STATION_ID ="162592"  # <-- replace with the confirmed active station's ID


def get_aqicn_data() -> dict | None:
    aqicn_api_key = os.getenv("AQICN_API_KEY")

    if not aqicn_api_key:
        print("Error: AQICN_API_KEY environment variable is missing.")
        return None

    aqicn_url = f"https://api.waqi.info/feed/@{AQICN_STATION_ID}/?token={aqicn_api_key}"

    try:
        aqicn_response = requests.get(aqicn_url, timeout=10)
        aqicn_response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching AQICN data: {e}")
        return None

    aqicn_data = aqicn_response.json()

    if aqicn_data.get("status") != "ok":
        error_info = aqicn_data.get("data", "Unknown API error")
        print(f"AQICN API returned an error status: {error_info}")
        return None

    data = aqicn_data.get("data", {})

    # Safely retrieve nested values using .get() to prevent KeyErrors
    city_info = data.get("city", {})
    station_name = city_info.get("name", "Unknown Station")
    aqi = data.get("aqi")

    print("Station:", station_name)
    print("AQI:", aqi)

    if aqi is None:
        print(
            "Warning: station returned no AQI value. It may be inactive "
            "or temporarily offline — consider swapping AQICN_STATION_ID "
            "above for a station that's currently reporting live data."
        )

    iaqi = data.get("iaqi", {})

    pm1 = iaqi.get("pm1", {}).get("v")
    pm25 = iaqi.get("pm25", {}).get("v")
    pm10 = iaqi.get("pm10", {}).get("v")

    humidity = iaqi.get("h", {}).get("v")
    temperature = iaqi.get("t", {}).get("v")

    timestamp = data.get("time", {}).get("iso")

    return {
        "station_id": f"A{AQICN_STATION_ID}",
        "station_name": station_name,
        "aqi": aqi,
        "pm1": pm1,
        "pm25": pm25,
        "pm10": pm10,
        "aqi_temperature": temperature,
        "aqi_humidity": humidity,
        "timestamp": timestamp
    }


if __name__ == "__main__":
    print(get_aqicn_data())
