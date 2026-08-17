"""
feature_pipeline.py
--------------------
Runs HOURLY (scheduled by .github/workflows/feature_pipeline.yml).
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

import hopsworks
import pandas as pd
from dotenv import load_dotenv
from hsfs.feature import Feature

from weather import get_current_weather
from aqi import get_aqicn_data
from history import load_history, update_history


load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Backup file location — override via env if you want it written
# somewhere persistent. On GitHub Actions runners this file does
# NOT survive between runs unless cached/uploaded/committed.
BACKUP_PATH = os.getenv("FEATURE_BACKUP_PATH", "feature_backup.csv")


def connect_feature_store():
    # cert_folder: relative, auto-created folder — works on both
    # Windows (local) and Linux (GitHub Actions). Avoids the
    # hopsworks library's broken '/tmp\\...' default on Windows.
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
        description="Hourly weather + pollutant + engineered AQI features for Karachi",
        primary_key=["date", "hour"],
        event_time="date",
        online_enabled=True,
    )


def build_feature_row() -> Tuple[Dict[str, Any], Optional[int], datetime]:

    weather_now = get_current_weather()

    if weather_now is None:
        raise RuntimeError(
            "Could not fetch current weather from OpenWeather — "
            "aborting this run."
        )

    aqicn = get_aqicn_data()

    observed_aqi: Optional[int] = (
        int(aqicn["aqi"])
        if aqicn and aqicn.get("aqi") is not None
        else None
    )

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

    row: Dict[str, Any] = {
        # Matches the feature group's existing 'date' type in
        # Hopsworks (a plain date, not a timestamp).
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

    if observed_aqi is not None:
        update_history(observed_aqi)

    print(f"Observed AQI being saved: {observed_aqi}")

    return row, observed_aqi, now


def main():
    try:

        # Build the feature row
        row, observed_aqi, now = build_feature_row()

        # Convert the row into a DataFrame
        df_row = pd.DataFrame([row])

        # --------------------------------------------------------
        # Local backup
        # --------------------------------------------------------

        if os.path.exists(BACKUP_PATH):
            df_row.to_csv(
                BACKUP_PATH,
                mode="a",
                header=False,
                index=False,
            )
        else:
            df_row.to_csv(
                BACKUP_PATH,
                index=False,
            )

        # --------------------------------------------------------
        # Connect to Hopsworks
        # --------------------------------------------------------

        fs = connect_feature_store()
        fg = get_or_create_feature_group(fs)

        # --------------------------------------------------------
        # Append missing features automatically
        # --------------------------------------------------------

        existing_feature_names = {
            f.name.lower()
            for f in fg.features
        }

        features_to_append = []

        for col in df_row.columns:

            if col.lower() not in existing_feature_names:

                dtype = df_row[col].dtype

                if pd.api.types.is_integer_dtype(dtype):
                    hs_type = "bigint"

                elif pd.api.types.is_float_dtype(dtype):
                    hs_type = "double"

                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    hs_type = "timestamp"

                else:
                    hs_type = "string"

                features_to_append.append(
                    Feature(
                        name=col,
                        type=hs_type,
                    )
                )

        if features_to_append:

            print(
                f"Appending {len(features_to_append)} "
                "missing features to Feature Group..."
            )

            fg.append_features(features_to_append)

            print("Features appended successfully.")

        else:

            print(
                "All features already exist in the Feature Group."
            )

        # --------------------------------------------------------
        # Debug output
        # --------------------------------------------------------

        print(df_row.dtypes)
        print(df_row)

        # --------------------------------------------------------
        # Insert into Hopsworks
        # --------------------------------------------------------
        # NOTE: no "start_offline_materialization": False here — that key
        # was removed. It was a leftover workaround from before the real
        # HDFS/deltalake fix; leaving it in would silently stop new hourly
        # rows from ever reaching the OFFLINE store, which is the ONLY
        # store training_pipeline.py and feature_snapshot.py read from.
        # Every future row would exist online only and training would
        # never see it again.

        fg.insert(
            df_row,
            write_options={"wait_for_job": True},
        )

        print(
            f"Successfully inserted AQI={observed_aqi}"
        )

    except Exception:

        print(
            "\nFeature pipeline failed.\n"
        )

        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
