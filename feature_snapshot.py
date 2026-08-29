

import time
from datetime import datetime, timezone

import pandas as pd

from feature_store import connect_feature_store
from weather import get_current_weather
from aqi import get_aqicn_data
from training_pipeline import prepare_clean_dataset, FEATURE_COLS


FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2



def build_today_features():

    
    try:
        fs = connect_feature_store()

        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )

    except Exception as e:
        print(f"Hopsworks connection failed: {e}")
        return None
    
    current_weather = get_current_weather()

    if current_weather is None:
        print("Unable to fetch current weather.")
        return None

    

    history_df = None
    last_error = None

    

    try:

        print(
            "Trying Hopsworks Feature Query Service..."
        )

        history_df = feature_group.read(
            online=True,
            read_options={
                "external": True,
                "arrow_flight_config": {
                    "timeout": 30
                }
            }
        )

        print(
            "Hopsworks Feature Query Service read succeeded."
        )

    except Exception as e:

        last_error = e

        print(
            f"Hopsworks Feature Query Service failed: {e}"
        )

        history_df = None

    if history_df is None or history_df.empty:

        print(
            "Could not read feature group from Hopsworks."
        )

        print(
            f"Last Hopsworks error: {last_error}"
        )

        return None

    

    history_df = history_df.copy()

    history_df["date"] = pd.to_datetime(
        history_df["date"]
    )

    history_df["aqi"] = pd.to_numeric(
        history_df["aqi"],
        errors="coerce"
    )

    known_aqi_df = (
        history_df
        .dropna(subset=["aqi"])
        .sort_values(["date", "hour"])
    )

    if known_aqi_df.empty:

        print("No historical AQI values available.")

        return None

    

    live_aqi = None

    try:

        aqicn = get_aqicn_data()

        if aqicn and aqicn.get("aqi") is not None:

            live_aqi = float(aqicn["aqi"])

            print(
                f"Live AQICN AQI: {live_aqi:.1f}"
            )

    except Exception as e:

        print(
            f"Could not fetch live AQI: {e}"
        )

    
    if live_aqi is not None:

        today_aqi = live_aqi

    else:

        today_aqi = float(
            known_aqi_df["aqi"].iloc[-1]
        )

        print(
            f"Live AQI unavailable. "
            f"Using latest historical AQI: {today_aqi:.1f}"
        )

    

    now = datetime.now(timezone.utc)

    today_row = {

        "date": pd.Timestamp(
            now.date()
        ),

        "hour": now.hour,

        "temperature":
            current_weather["temperature"],

        "humidity":
            current_weather["humidity"],

        "pressure":
            current_weather["pressure"],

        "wind_speed":
            current_weather["wind_speed"],

        "rain":
            current_weather["rain"],

        "pm10":
            current_weather["pm10"],

        "pm2_5":
            current_weather["pm25"],

        "carbon_monoxide":
            current_weather["carbon_monoxide"],

        "nitrogen_dioxide":
            current_weather["nitrogen_dioxide"],

        "sulphur_dioxide":
            current_weather["sulphur_dioxide"],

        "ozone":
            current_weather["ozone"],

        "aqi":
            today_aqi,
    }

    
    combined_raw = pd.concat(
        [
            history_df,
            pd.DataFrame([today_row])
        ],
        ignore_index=True
    )

    
    engineered = prepare_clean_dataset(
        combined_raw
    )

    if engineered.empty:

        print(
            "Feature engineering produced no rows."
        )

        return None

    

    today_engineered = engineered.iloc[-1]

    
    features = {}

    for col in FEATURE_COLS:

        value = today_engineered.get(
            col,
            0.0
        )

        if pd.isna(value):

            value = 0.0

        features[col] = float(value)

    
    features["aqi"] = float(
        today_aqi
    )

    if live_aqi is not None:

        features["live_aqi"] = float(
            live_aqi
        )

    

    print(
        "\n===== TODAY FEATURE SNAPSHOT ====="
    )

    print(
        f"Today's AQI: {today_aqi:.2f}"
    )

    print(
        f"AQI lag 1: "
        f"{features.get('aqi_lag_1', 0):.2f}"
    )

    print(
        f"AQI lag 2: "
        f"{features.get('aqi_lag_2', 0):.2f}"
    )

    print(
        f"AQI rolling mean 3: "
        f"{features.get('aqi_rolling_mean_3', 0):.2f}"
    )

    print(
        f"AQI EMA 3: "
        f"{features.get('aqi_ema_3', 0):.2f}"
    )

    print(
        "==================================\n"
    )

    return features