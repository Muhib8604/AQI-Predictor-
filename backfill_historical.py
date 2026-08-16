"""
backfill_historical.py
-----------------------
ONE-TIME script. Merges your two historical CSVs (pollutants+aqi, and
weather) and bulk-inserts them into the SAME Hopsworks feature group that
feature_pipeline.py writes to hourly — so training_pipeline.py has real
history to work with instead of waiting ~2 weeks for hourly rows to add up.

Expects two files in this folder:
  pollutants_csv: date,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aqi
  weather_csv:     date,temperature,humidity,pressure,wind_speed,rain

Computes aqi_lag_1/2/3, aqi_rolling_mean_3, aqi_change the EXACT same way
feature_pipeline.py's build_feature_row() does — walking forward through
time, only ever using aqi values that happened BEFORE the current row —
so backfilled rows are consistent with what real hourly rows would have
looked like.

Run this once, from wherever your writes currently work (GitHub Actions
manual trigger, or WSL) — same requirement as feature_pipeline.py, since
this also WRITES to Hopsworks.
"""

import sys
import pandas as pd
from feature_store import connect_feature_store

POLLUTANTS_CSV = "historical_aqi_clean.csv"   # <-- rename to match your actual filename
WEATHER_CSV = "historical_weather.csv"          # <-- rename to match your actual filename

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather + pollutant + engineered AQI features for Karachi",
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
    history = []  # running list of past aqi values, oldest first — mirrors aqi_history.json

    for _, r in merged.iterrows():
        lag_1 = history[-1] if len(history) >= 1 else 0
        lag_2 = history[-2] if len(history) >= 2 else 0
        lag_3 = history[-3] if len(history) >= 3 else 0
        rolling_3 = (sum(history[-3:]) / 3) if len(history) >= 3 else lag_1
        aqi_change = lag_1 - lag_2

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

            "aqi_lag_1": lag_1,
            "aqi_lag_2": lag_2,
            "aqi_lag_3": lag_3,
            "aqi_rolling_mean_3": rolling_3,
            "aqi_change": aqi_change,

            "aqi": r["aqi"],
        })

        history.append(r["aqi"])  # only NOW does today's own aqi become part of history

    return pd.DataFrame(rows)


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
