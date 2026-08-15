"""
feature_snapshot.py
--------------------
Single source of truth for building "today's" feature row — the exact input
fed to all 3 per-day models (main.py's /predict route) AND used by the
Explainability dashboard page to reconstruct what the model actually saw.

Why this file exists: previously main.py and the Explainability page each
built this row independently, and they drifted out of sync (the
Explainability page was missing aqi_change, causing a crash). Importing
this one function from both places means they can't drift again — change
the feature set once, here, and everything downstream stays consistent.
"""

from datetime import datetime
from feature_store import connect_feature_store
from weather import get_current_weather
from history import (
    load_history,
    get_aqi_lag_1,
    get_aqi_lag_2,
    get_aqi_lag_3,
    get_rolling_mean_3,
    get_aqi_trend,
)
fs = connect_feature_store()

feature_group = fs.get_feature_group(
    name="aqi_features",
    version=1
)

df = feature_group.read()

def build_today_features():
    """Returns a dict with today's real feature values (matching
    training_pipeline.py's FEATURE_COLUMNS exactly), or None if the live
    weather API call failed."""

    current_weather = get_current_weather()
    if current_weather is None:
        return None

    history = load_history()

    now = datetime.now()

    return {
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

        "year": now.year,
        "month": now.month,
        "day": now.day,
        "day_of_week": now.weekday(),

        "aqi_lag_1": get_aqi_lag_1(history),
        "aqi_lag_2": get_aqi_lag_2(history),
        "aqi_lag_3": get_aqi_lag_3(history),
        "aqi_rolling_mean_3": get_rolling_mean_3(history),
        "aqi_change": get_aqi_trend(history),
    }
