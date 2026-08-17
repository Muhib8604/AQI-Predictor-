import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Priority order — first successful station will be used
AQICN_STATIONS = [
    "162592",     # Zafar Memon DHA
    "A471613",    # Saddar Town (Clarity)    
]


def get_aqicn_data(station_id: str = None) -> dict | None:
    aqicn_api_key = os.getenv("AQICN_API_KEY")

    if not aqicn_api_key:
        print("Error: AQICN_API_KEY environment variable is missing.")
        return None

    stations_to_try = [station_id] if station_id else AQICN_STATIONS

    for sid in stations_to_try:
        # Remove @ if user accidentally put it
        sid = str(sid).lstrip("@")

        url = f"https://api.waqi.info/feed/@{sid}/?token={aqicn_api_key}"

        try:
            print(f"Trying AQICN station: {sid}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                print(f"  Station {sid} returned error status")
                continue

            payload = data.get("data", {})
            aqi = payload.get("aqi")

            if aqi is None:
                print(f"  Station {sid} has no AQI value (offline?)")
                continue

            station_name = payload.get("city", {}).get("name", "Unknown Station")
            print(f"  Success → Station: {station_name} | AQI: {aqi}")

            iaqi = payload.get("iaqi", {})

            return {
                "station_id": f"@{sid}",
                "station_name": station_name,
                "aqi": aqi,
                "pm1": iaqi.get("pm1", {}).get("v"),
                "pm25": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),
                "aqi_temperature": iaqi.get("t", {}).get("v"),
                "aqi_humidity": iaqi.get("h", {}).get("v"),
                "timestamp": payload.get("time", {}).get("iso"),
            }

        except Exception as e:
            print(f"  Station {sid} failed: {e}")
            continue

    print("All AQICN stations failed or returned no AQI.")
    return None


if __name__ == "__main__":
    print(get_aqicn_data())