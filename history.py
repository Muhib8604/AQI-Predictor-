import json
import os

HISTORY_FILE = "aqi_history.json"


def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r") as file:
        return json.load(file)


def save_history(history):

    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)


def update_history(today_aqi):

    history = load_history()

    history.append(today_aqi)

    save_history(history)

    return history


def get_aqi_lag_1(history):

    if len(history) == 0:
        return 0

    return history[-1]


def get_rolling_mean_3(history):

    if len(history) == 0:
        return 0

    if len(history) < 3:
        return sum(history) / len(history)

    return sum(history[-3:]) / 3


# ---- NEW: added for the richer multi-horizon feature set ----

def get_aqi_lag_2(history):
    """AQI from 2 readings ago. Falls back to lag_1 if not enough history yet,
    same fallback style as the existing lag/rolling functions above."""

    if len(history) < 2:
        return get_aqi_lag_1(history)

    return history[-2]


def get_aqi_lag_3(history):
    """AQI from 3 readings ago. Falls back to lag_2 if not enough history yet."""

    if len(history) < 3:
        return get_aqi_lag_2(history)

    return history[-3]


def get_aqi_trend(history):
    """Simple momentum indicator: lag_1 - lag_2. Positive = AQI rising,
    negative = AQI falling. Returns 0 if there isn't enough history to
    compute a meaningful trend yet."""

    if len(history) < 2:
        return 0

    return get_aqi_lag_1(history) - get_aqi_lag_2(history)
