"""
feature_snapshot.py
Builds today's prediction features using the same feature engineering
logic used during training.
"""

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

    # ==========================================================
    # 1. CURRENT WEATHER
    # ==========================================================
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

    # ==========================================================
    # 2. READ HISTORICAL DATA
    # ==========================================================

    history_df = None
    last_error = None

    # ==========================================================
    # READ HISTORICAL DATA FROM HOPSWORKS
    # ==========================================================

    # First try the normal Hopsworks Feature Query Service.
    # If its Arrow Flight connection is unavailable, fall back
    # to Hive. Both paths still read from Hopsworks Feature Store.

    try:

        print(
            "Trying Hopsworks Feature Query Service..."
        )

        history_df = feature_group.read(
            online=True,
            read_options={
                "arrow_flight_config": {
                    "timeout": 900
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

        # ======================================================
        # FALLBACK: HIVE
        # ======================================================

        try:

            print(
                "Falling back to Hopsworks Hive read..."
            )

            history_df = feature_group.read(
                read_options={
                    "use_hive": True
                }
            )

            print(
                "Hopsworks Hive read succeeded."
            )

        except Exception as hive_error:

            last_error = hive_error

            print(
                f"Hopsworks Hive read also failed: "
                f"{hive_error}"
            )


    if history_df is None or history_df.empty:

        print(
            "Could not read feature group from Hopsworks."
        )

        print(
            f"Last Hopsworks error: {last_error}"
        )

        return None

    # ==========================================================
    # 3. CLEAN HISTORICAL AQI
    # ==========================================================

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

    # ==========================================================
    # 4. GET LIVE AQI
    # ==========================================================

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

    # ==========================================================
    # 5. USE LIVE AQI AS TODAY'S ACTUAL AQI
    # ==========================================================

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

    # ==========================================================
    # 6. BUILD TODAY'S RAW ROW
    # ==========================================================

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

    # ==========================================================
    # 7. APPEND TODAY TO HISTORICAL DATA
    # ==========================================================

    combined_raw = pd.concat(
        [
            history_df,
            pd.DataFrame([today_row])
        ],
        ignore_index=True
    )

    # ==========================================================
    # 8. RUN EXACT TRAINING FEATURE ENGINEERING
    # ==========================================================

    engineered = prepare_clean_dataset(
        combined_raw
    )

    if engineered.empty:

        print(
            "Feature engineering produced no rows."
        )

        return None

    # ==========================================================
    # 9. GET THE MOST RECENT ENGINEERED ROW
    # ==========================================================

    today_engineered = engineered.iloc[-1]

    # ==========================================================
    # 10. BUILD MODEL INPUT
    # ==========================================================

    features = {}

    for col in FEATURE_COLS:

        value = today_engineered.get(
            col,
            0.0
        )

        if pd.isna(value):

            value = 0.0

        features[col] = float(value)

    # ==========================================================
    # 11. KEEP TODAY'S LIVE AQI SEPARATELY
    # ==========================================================

    features["aqi"] = float(
        today_aqi
    )

    if live_aqi is not None:

        features["live_aqi"] = float(
            live_aqi
        )

    # ==========================================================
    # 12. DEBUG OUTPUT
    # ==========================================================

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