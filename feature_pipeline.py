"""
feature_pipeline.py
--------------------
Runs HOURLY.
"""

import os
import sys
import traceback
from feature_schema import align_to_hopsworks_schema
import numpy as np   
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from weather import get_current_weather
from aqi import get_aqicn_data
from history import load_history, update_history

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2

BACKUP_PATH = os.getenv("FEATURE_BACKUP_PATH", "feature_backup.csv")


def connect_feature_store():
    cert_folder = os.getenv(
        "HOPSWORKS_CERT_FOLDER",
        os.path.join(os.getcwd(), "hops_certs"),
    )
    os.makedirs(cert_folder, exist_ok=True)

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        engine="python",
        cert_folder=cert_folder,
    )
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly basic weather + pollutant + simple AQI features for Karachi (v2)",
        primary_key=["date", "hour"],
        event_time="date",
        online_enabled=True,
    )


def build_feature_row() -> Tuple[Dict[str, Any], Optional[float], datetime]:
    weather_now = get_current_weather()
    if weather_now is None:
        raise RuntimeError("Could not fetch current weather from OpenWeather")

    aqicn = get_aqicn_data()

    observed_aqi: Optional[float] = None
    if aqicn and aqicn.get("aqi") is not None:
        observed_aqi = float(aqicn["aqi"])

    history = load_history()

    lag_1 = history[-1] if len(history) >= 1 else 0
    lag_2 = history[-2] if len(history) >= 2 else 0
    lag_3 = history[-3] if len(history) >= 3 else 0

    rolling_3 = sum(history[-3:]) / 3 if len(history) >= 3 else lag_1
    aqi_change = lag_1 - lag_2

    now = datetime.now(timezone.utc)

    row: Dict[str, Any] = {
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

        # Important: use np.nan instead of None
        "aqi": observed_aqi if observed_aqi is not None else np.nan,
    }

    if observed_aqi is not None:
        update_history(observed_aqi)

    print(f"Observed AQI being saved: {observed_aqi}")
    return row, observed_aqi, now


def main():
    try:
        row, observed_aqi, _ = build_feature_row()
        df_row = pd.DataFrame([row])
        df_row["aqi"] = df_row["aqi"].astype("float64")
        # Local backup
        if os.path.exists(BACKUP_PATH):
            df_row.to_csv(BACKUP_PATH, mode="a", header=False, index=False)
        else:
            df_row.to_csv(BACKUP_PATH, index=False)

        fs = connect_feature_store()
        fg = get_or_create_feature_group(fs)

        df_row = align_to_hopsworks_schema(df_row, fg)

        print("Final dataframe dtypes:")
        print(df_row.dtypes)

        print("\nFinal dataframe:")
        print(df_row)

        # Important: Do NOT call append_features on a brand new FG.
        # First insert will create the schema automatically.
        # Make sure AQI is populated BEFORE this point
        df_row["aqi"] = pd.to_numeric(df_row["aqi"], errors="coerce")

        for col in [
            "aqi_lag_1",
            "aqi_lag_2",
            "aqi_lag_3",
            "aqi_change",
        ]:
            df_row[col] = df_row[col].astype("int64")

        df_row["aqi"] = df_row["aqi"].astype("float64")
        df_row["aqi_rolling_mean_3"] = df_row["aqi_rolling_mean_3"].astype("float64")

        print("\nData going into Hopsworks:")
        print(df_row[["date", "hour", "aqi"]].tail())

                # ---------- ROBUST INSERT WITH RETRIES ----------
        max_retries = 3
        inserted = False

        for attempt in range(1, max_retries + 1):
            try:
                print(f"Insert attempt {attempt}/{max_retries}...")

                fg.insert(
                    df_row,
                    write_options={
                        "wait_for_job": False,                  # important
                        "wait_for_online_ingestion": False,     # important
                    },
                )

                print("Feature Group insert successful.")
                print(f"Successfully inserted AQI={observed_aqi}")
                inserted = True
                break

            except Exception as e:
                print(f"Insert attempt {attempt} failed: {e}")

                if attempt < max_retries:
                    import time
                    time.sleep(8 * attempt)   # 8s, 16s
                else:
                    print("All insert attempts failed. Data saved only to local backup.")
                    # Do NOT sys.exit(1) — local backup already written


if __name__ == "__main__":
    main()