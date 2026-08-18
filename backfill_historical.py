"""
backfill_historical.py
-----------------------
ONE-TIME script. Merges your two historical CSVs (pollutants+aqi, and
weather) and bulk-inserts them into feature group v2 — the SAME one
feature_pipeline.py writes to hourly and training_pipeline.py /
feature_snapshot.py read from.

Expects two files in this folder:
  pollutants_csv: date,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aqi
  weather_csv:     date,temperature,humidity,pressure,wind_speed,rain
"""

import sys
import pandas as pd
from feature_schema import align_to_hopsworks_schema
from feature_store import connect_feature_store

POLLUTANTS_CSV = "historical_aqi_clean.csv"   # <-- rename to match your actual filename
WEATHER_CSV = "historical_weather.csv"          # <-- rename to match your actual filename

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
            "hour": 0,
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

            # Placeholders — training_pipeline recomputes these later
            "aqi_lag_1": 0.0,
            "aqi_lag_2": 0.0,
            "aqi_lag_3": 0.0,
            "aqi_rolling_mean_3": 0.0,
            "aqi_change": 0.0,

            "aqi": r["aqi"],
        })

    return pd.DataFrame(rows)


def main():
    try:
        pollutants_df = pd.read_csv(POLLUTANTS_CSV)
        weather_df = pd.read_csv(WEATHER_CSV)
    except FileNotFoundError as e:
        print(f"Could not find a CSV file: {e}")
        print("Check the POLLUTANTS_CSV / WEATHER_CSV filenames at the top of this script.")
        sys.exit(1)

    backfill_df = build_backfill_rows(pollutants_df, weather_df)

    print(
        f"Built {len(backfill_df)} historical rows spanning "
        f"{backfill_df['date'].min()} to {backfill_df['date'].max()}"
    )

    fs = connect_feature_store()
    fg = get_or_create_feature_group(fs)

    # Match the EXISTING V2 Hopsworks schema
    backfill_df = align_to_hopsworks_schema(backfill_df, fg)

    print("\nBackfill dtypes:")
    print(backfill_df.dtypes)

    print("\nUploading to Hopsworks V2...")
    fg.insert(
        backfill_df,
        write_options={"wait_for_job": True}
    )

    print(f"Backfill complete: inserted {len(backfill_df)} rows.")


if __name__ == "__main__":
    main()