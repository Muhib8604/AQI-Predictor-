"""
feature_snapshot.py
--------------------
Single source of truth for building "today's" feature row — the exact input
fed to all 3 per-day models (main.py's /predict route) AND used by the
Explainability dashboard page to reconstruct what the model actually saw.

WHY THIS VERSION IS DIFFERENT: earlier versions hand-built this row with
their own copy of the lag/rolling/change formulas, and drifted out of sync
with training_pipeline.py every time that file's feature set grew (that's
what caused the "not in index" crash on the Explainability page). This
version instead imports and calls prepare_clean_dataset() and FEATURE_COLS
DIRECTLY from training_pipeline.py — the actual function training uses —
so there is now only one place these formulas are ever written. If
training_pipeline.py's feature set changes again in the future, this file
does not need to change at all.

HOW "TODAY" IS BUILT WITHOUT KNOWING TODAY'S OWN AQI:
We append one placeholder row to the real historical data pulled from
Hopsworks, using today's LIVE weather/pollutant reading. That placeholder
row's own "aqi" value is a dummy (we don't know it yet — that's what we're
forecasting) — but every column in FEATURE_COLS (aqi_lag_*, rolling means/
stds, emas, changes) is built from aqi.shift(1) onward, i.e. strictly PAST
days only, so the dummy value never actually reaches any FEATURE_COLS
column. It exists purely so prepare_clean_dataset()'s final dropna() step
doesn't discard this last row for lacking a same-day AQI we can't know yet.
"""

from datetime import datetime, timezone

import pandas as pd

from feature_store import connect_feature_store
from weather import get_current_weather
from training_pipeline import prepare_clean_dataset, FEATURE_COLS

fs = connect_feature_store()
# VERSION MUST MATCH feature_pipeline.py, training_pipeline.py, and
# backfill_historical.py — this was still hardcoded to v1 (the old,
# schema-polluted feature group) while everything else moved to v2,
# which is exactly why "unable to fetch aqi" was happening on the
# dashboard — this function was reading empty/wrong data from v1.
feature_group = fs.get_feature_group(name="aqi_features", version=2)


def build_today_features():
    """Returns a dict with today's real feature values, computed by the
    SAME function training_pipeline.py uses, or None if the live weather
    API call failed or there isn't enough history yet."""

    current_weather = get_current_weather()
    if current_weather is None:
        return None

    # Read fresh on every call (not once at import time) so this always reflects
    # the latest data actually saved to Hopsworks, even in a long-running
    # server process.
    history_df = feature_group.read()
    if history_df.empty:
        return None

    now = datetime.now(timezone.utc)
    last_known_aqi = history_df.sort_values("date")["aqi"].iloc[-1]

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
        "aqi": last_known_aqi,  # dummy placeholder — see module docstring
    }

    combined_raw = pd.concat(
        [history_df, pd.DataFrame([today_row])], ignore_index=True
    )
    engineered = prepare_clean_dataset(combined_raw)

    if engineered.empty:
        # Not enough history yet (e.g. fewer than 14 days) for the longest
        # lag/rolling features to have real values.
        return None

    today_engineered = engineered.iloc[-1]
    return {col: today_engineered[col] for col in FEATURE_COLS}
