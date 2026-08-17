"""
backfill_historical.py
-----------------------
ONE-TIME script. Merges your two historical CSVs (pollutants+aqi, and
weather) and bulk-inserts them into feature group v2 — the SAME one
feature_pipeline.py writes to hourly and training_pipeline.py /
feature_snapshot.py read from.

WHY THIS IS SIMPLER THAN THE OLD VERSION: the old script carefully
walked forward through time computing real aqi_lag_1/aqi_lag_2/aqi_lag_3/
aqi_rolling_mean_3/aqi_change values, row by row. That turned out to be
unnecessary — training_pipeline.py's prepare_clean_dataset() OVERWRITES
these exact columns by recomputing them fresh from the raw "aqi" column
every time it runs. Whatever values sit in the feature store for these 5
columns are never actually read by anything downstream. So backfilled
rows just fill them with 0.0 placeholders — the schema requires the
columns to exist with a numeric value, but their content doesn't matter.

Expects two files in this folder:
  pollutants_csv: date,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aqi
  weather_csv:     date,temperature,humidity,pressure,wind_speed,rain
"""

import sys
import pandas as pd
from feature_store import connect_feature_store

POLLUTANTS_CSV = "pollutants_historical.csv"   # <-- rename to match your actual filename
WEATHER_CSV = "weather_historical.csv"          # <-- rename to match your actual filename

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2  # MUST MATCH feature_pipeline.py / training_pipeline.py / feature_snapshot.py


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly basic weather + pollutant + simple AQI features for Karachi (v2)",
        primary_key=["date", "hour"],
        event_time="date",
        online_enabled=True,
    )


def build_backfill_rows(pollutants_df, weather_df):
    pollutants_df["date"] = pd.to_datetime(pollutants_df["date"]).dt.date
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    merged = pd.merge(pollutants_df, weather_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    dropped = len(pollutants_df) - len(merged)
    if dropped > 0:
        print(f"Note: {dropped} date(s) in the pollutants file had no matching "
              f"weather row (or vice versa) and were skipped.")

    rows = []
    for _, r in merged.iterrows():
        d = r["date"]
        rows.append({
            "date": d,
            "hour": 0,  # backfilled rows represent a daily snapshot; safe since these are past dates only
            "day": d.day,
            "month": d.month,
            "year": d.year,
            "day_of_week": d.weekday(),
            "city": "Karachi",

            "temperature": r["temperature"],
            "humidity": r["humidity"],
            "pressure": r["pressure"],
            "wind_speed": r["wind_speed"],
            "rain": r["rain"],

            "pm10": r["pm10"],
            "pm2_5": r["pm2_5"],
            "carbon_monoxide": r["carbon_monoxide"],
            "nitrogen_dioxide": r["nitrogen_dioxide"],
            "sulphur_dioxide": r["sulphur_dioxide"],
            "ozone": r["ozone"],

            # Placeholders — never actually read downstream, see module docstring.
            "aqi_lag_1": 0.0,
            "aqi_lag_2": 0.0,
            "aqi_lag_3": 0.0,
            "aqi_rolling_mean_3": 0.0,
            "aqi_change": 0.0,

            "aqi": r["aqi"],
        })

    result = pd.DataFrame(rows)

    # Match the live pipeline's dtypes as closely as possible. If Hopsworks
    # still rejects the insert with a type-mismatch error on a specific
    # column, that tells us its ACTUAL locked-in schema type for that
    # column — flip int64 <-> float64 for just that column and retry.
    result["pressure"] = result["pressure"].round().astype("int64")
    result["rain"] = result["rain"].round().astype("int64")
    for col in ["aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change", "aqi"]:
        result[col] = result[col].astype("float64")

    return result


def main():
    try:
        pollutants_df = pd.read_csv(POLLUTANTS_CSV)
        weather_df = pd.read_csv(WEATHER_CSV)
    except FileNotFoundError as e:
        print(f"Could not find a CSV file: {e}")
        print(f"Check the POLLUTANTS_CSV / WEATHER_CSV filenames at the top of this script.")
        sys.exit(1)

    backfill_df = build_backfill_rows(pollutants_df, weather_df)
    print(f"Built {len(backfill_df)} historical rows spanning "
          f"{backfill_df['date'].min()} to {backfill_df['date'].max()}")

    fs = connect_feature_store()
    fg = get_or_create_feature_group(fs)

    print("Uploading to Hopsworks (this is one bulk insert, not one-by-one)...")
    fg.insert(backfill_df, write_options={"wait_for_job": True})

    print(f"Backfill complete: inserted {len(backfill_df)} rows.")


if __name__ == "__main__":
    main()
