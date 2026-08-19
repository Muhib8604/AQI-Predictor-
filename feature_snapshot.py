"""
feature_snapshot.py
"""

import time
from datetime import datetime, timezone

import pandas as pd

from feature_store import connect_feature_store
from weather import get_current_weather
from aqi import get_aqicn_data
from training_pipeline import prepare_clean_dataset, FEATURE_COLS

fs = connect_feature_store()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2

feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)


def build_today_features():
    current_weather = get_current_weather()
    if current_weather is None:
        return None

    # ----------------------------------------------------------
    # Retry Hopsworks read
    # ----------------------------------------------------------
    history_df = None
    last_error = None

    for attempt in range(3):
        try:
            history_df = feature_group.read()
            break
        except Exception as e:
            last_error = e
            print(f"Hopsworks read failed (attempt {attempt + 1}/3): {e}")
            time.sleep(2)

    if history_df is None or history_df.empty:
        print(f"Could not read feature group after retries: {last_error}")
        return None

    history_df["aqi"] = pd.to_numeric(history_df["aqi"], errors="coerce")

    known_aqi_df = (
        history_df
        .dropna(subset=["aqi"])
        .sort_values(["date", "hour"])
    )

    if known_aqi_df.empty:
        return None

    # Prefer LIVE station AQI when available
    live_aqi = None
    try:
        aqicn = get_aqicn_data()
        if aqicn and aqicn.get("aqi") is not None:
            live_aqi = float(aqicn["aqi"])
            print(f"Using live AQICN AQI as latest reference: {live_aqi}")
    except Exception as e:
        print(f"Could not fetch live AQI for feature snapshot: {e}")

    last_known_aqi = live_aqi if live_aqi is not None else float(known_aqi_df["aqi"].iloc[-1])

    now = datetime.now(timezone.utc)

    today_row = {
        "date": now.date(),
        "temperature": current_weather["temperature"],
        "humidity": current_weather["humidity"],
        "pressure": current_weather["pressure"],
        "wind_speed": current_weather["wind_speed"],
        "rain": current_weather["rain"],
        "pm10": current_weather["pm10"],
        "pm2_5": current_weather["pm25"],
        "carbon_monoxide": current_weather["carbon_monoxide"],
        "nitrogen_dioxide": current_weather["nitrogen_dioxide"],
        "sulphur_dioxide": current_weather["sulphur_dioxide"],
        "ozone": current_weather["ozone"],
        "aqi": last_known_aqi,
    }

    combined_raw = pd.concat(
        [history_df, pd.DataFrame([today_row])],
        ignore_index=True
    )

    engineered = prepare_clean_dataset(combined_raw)

    if engineered.empty:
        return None

    today_engineered = engineered.iloc[-1]
    features = {col: today_engineered[col] for col in FEATURE_COLS}

    # Keep live AQI available for caller (main.py blend)
    features["aqi"] = last_known_aqi
    if live_aqi is not None:
        features["live_aqi"] = live_aqi

    return features