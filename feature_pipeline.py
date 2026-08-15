"""
feature_pipeline.py
--------------------
Runs HOURLY (scheduled by .github/workflows/feature_pipeline.yml).

1. Fetches the current weather + pollutant reading from OpenWeather.
2. Fetches the current ground-truth AQI reading from AQICN.
3. Computes engineered features:
     - time-based: hour, day, month, year, day_of_week
     - derived:    aqi_lag_1, aqi_rolling_mean_3, aqi_change
4. Appends the observed AQI to the local lag/rolling history (aqi_history.json).
5. Writes the full feature row to the Hopsworks Feature Store.

This directly satisfies:
  "Fetch raw weather and pollutant data from external APIs like AQICN or OpenWeather"
  "Compute features from raw data including time-based features (hour, day,
   month) and derived features like AQI change rate"
  "Store processed features in Feature Store (Hopsworks or Vertex AI)"
"""

import os
import traceback
from datetime import datetime, timezone
import hopsworks
import pandas as pd
from dotenv import load_dotenv

from weather import get_current_weather
from aqi import get_aqicn_data
from history import load_history, update_history, get_aqi_lag_1, get_rolling_mean_3

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def connect_feature_store():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        engine="python",
        cert_folder=r"E:\AQI Predictor\hops_certs",
    )
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather + pollutant + engineered AQI features for Karachi",
        primary_key=["date","hour"],
        event_time="date",
        online_enabled=True,
    )


def build_feature_row():
    """Build one hourly feature row. AQICN supplies the ground-truth AQI
    reading (used both as the training target and to update lag/rolling
    history); OpenWeather supplies the raw weather + pollutant inputs."""

    weather_now = get_current_weather()
    if weather_now is None:
        raise RuntimeError("Could not fetch current weather from OpenWeather — aborting this run.")

    aqicn = get_aqicn_data()
    observed_aqi = aqicn["aqi"] if (aqicn and aqicn.get("aqi") is not None) else None

    history = load_history()

    lag_1 = history[-1] if len(history) >= 1 else 0
    lag_2 = history[-2] if len(history) >= 2 else 0
    lag_3 = history[-3] if len(history) >= 3 else 0

    rolling_3 = (
    sum(history[-3:]) / 3
    if len(history) >= 3
    else lag_1
)

    aqi_change = lag_1 - lag_2
    now = datetime.now(timezone.utc)

    row = {
        "date": now.date(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "year": now.year,
        "day_of_week": now.weekday(),
        "city": "Karachi",

        "temperature": weather_now["temperature"],
        "humidity": weather_now["humidity"],
        "pressure": weather_now["pressure"],
        "wind_speed": weather_now["wind_speed"],
        "rain": weather_now["rain"],

        "pm10": weather_now["pm10"],
        "pm2_5": weather_now["pm25"],
        "carbon_monoxide": weather_now["carbon_monoxide"],
        "nitrogen_dioxide": weather_now["nitrogen_dioxide"],
        "sulphur_dioxide": weather_now["sulphur_dioxide"],
        "ozone": weather_now["ozone"],

        "aqi_lag_1": lag_1,
        "aqi_lag_2": lag_2,
        "aqi_lag_3": lag_3,
        "aqi_rolling_mean_3": rolling_3,
        "aqi_change": aqi_change,

        "aqi": observed_aqi,
    }

    # Keep the local lag/rolling history in sync with what we just observed
    if observed_aqi is not None:
        update_history(observed_aqi)

    return row, observed_aqi, now

    
def main():
    try:
        row, observed_aqi, now = build_feature_row()
        df_row = pd.DataFrame([row])

        backup = "feature_backup.csv"

        if os.path.exists(backup):
            df_row.to_csv(
                backup,
                mode="a",
                header=False,
                index=False,
            )
        else:
            df_row.to_csv(
                backup,
                index=False,
            )

        # ALWAYS upload
        fs = connect_feature_store()
        fg = get_or_create_feature_group(fs)
        print(df_row.dtypes)
        print()
        print(df_row)
        fg.insert(df_row)
        
        print(f"Inserted AQI={observed_aqi}")

        
    except Exception:
        print("\nFeature pipeline failed.\n")
        traceback.print_exc()
        

if __name__ == "__main__":
    main()
