import requests
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from datetime import timezone

load_dotenv()

#AQICN api key 
def get_aqicn_data():

    aqicn_api_key = os.getenv("AQICN_API_KEY")

    aqicn_url = f"https://api.waqi.info/feed/A545140/?token={aqicn_api_key}"

    aqicn_response = requests.get(aqicn_url)

    if aqicn_response.status_code != 200:
        print("Error fetching AQICN data")
        return None

    aqicn_data = aqicn_response.json()
    
    if aqicn_data.get("status") != "ok":
        print("AQICN API returned an error")
        return None

    data = aqicn_data["data"]

    aqi = data.get("aqi")

    iaqi = data.get("iaqi", {})

    pm1 = iaqi.get("pm1", {}).get("v")
    pm25 = iaqi.get("pm25", {}).get("v")
    pm10 = iaqi.get("pm10", {}).get("v")

    humidity = iaqi.get("h", {}).get("v")
    temperature = iaqi.get("t", {}).get("v")

    timestamp = data.get("time", {}).get("iso")

    city = data.get("city", {})

    station_name = city.get("name")

    clean_data = {
        "station_id": "A545140",
        "station_name": station_name,
        "aqi": aqi,
        "pm1": pm1,
        "pm25": pm25,
        "pm10": pm10,
        "aqi_temperature": temperature,
        "aqi_humidity": humidity,
        "timestamp": timestamp
    }

    return clean_data