"""
predict.py

Day 1/2/3 predictions with live-AQI anchoring.
"""

import os
import json
import joblib
import pandas as pd
import torch

from model_definition import AQINet


FEATURES_PATH = "model_features.pkl"
MANIFEST_PATH = "model_manifest.json"

_FEATURE_COLUMNS = None
_MANIFEST = None


def _ensure_loaded():
    global _FEATURE_COLUMNS, _MANIFEST

    if _FEATURE_COLUMNS is not None:
        return

    if (
        not os.path.exists(FEATURES_PATH)
        or not os.path.exists(MANIFEST_PATH)
    ):
        raise RuntimeError(
            "Model files missing. Run training_pipeline.py first."
        )

    _FEATURE_COLUMNS = joblib.load(FEATURES_PATH)

    with open(MANIFEST_PATH) as f:
        _MANIFEST = json.load(f)


def _predict_raw(horizon_name, df):
    info = _MANIFEST[horizon_name]

    if info["kind"] == "sklearn":

        model = joblib.load(info["file"])

        return float(
            model.predict(df)[0]
        )

    scaler = joblib.load(
        info["scaler_file"]
    )

    scaled_x = scaler.transform(df)

    model = AQINet(
        len(_FEATURE_COLUMNS)
    )

    model.load_state_dict(
        torch.load(
            info["file"],
            map_location="cpu"
        )
    )

    model.eval()

    with torch.no_grad():

        scaled_pred = float(
            model(
                torch.tensor(
                    scaled_x,
                    dtype=torch.float32
                )
            ).item()
        )

    return (
        scaled_pred * info["target_std"]
    ) + info["target_mean"]


def _anchor_prediction(
    model_prediction,
    live_aqi,
    strength
):
    """
    Pull model prediction toward current live AQI.

    strength:
        0.0 = no correction
        1.0 = completely use live AQI
    """

    if live_aqi is None:
        return float(model_prediction)

    return (
        (1.0 - strength) * float(model_prediction)
        + strength * float(live_aqi)
    )


def predict_all_horizons(features: dict):

    _ensure_loaded()

    live_aqi = features.get(
        "live_aqi",
        features.get("aqi")
    )

    if live_aqi is not None:
        live_aqi = float(live_aqi)

    # ======================================================
    # REQUIRED FEATURES
    # ======================================================

    row = {}

    for col in _FEATURE_COLUMNS:

        if col not in features:
            raise ValueError(
                f"Missing required feature: {col}"
            )

        row[col] = float(features[col])

    # ======================================================
    # RAW MODEL PREDICTIONS
    # ======================================================

    df = pd.DataFrame(
        [row]
    )[_FEATURE_COLUMNS]

    # Day 1
    raw_day1 = _predict_raw(
        "day1",
        df
    )

    # Day 2
    raw_day2_delta = _predict_raw(
        "day2",
        df
    )

    raw_day2 = (
        raw_day1 + raw_day2_delta
        if _MANIFEST["day2"]["is_delta"]
        else raw_day2_delta
    )

    # Day 3
    raw_day3_delta = _predict_raw(
        "day3",
        df
    )

    raw_day3 = (
        raw_day2 + raw_day3_delta
        if _MANIFEST["day3"]["is_delta"]
        else raw_day3_delta
    )

    # ======================================================
    # LIVE AQI ANCHOR
    # ======================================================

    if live_aqi is not None:

        # Day 1 gets the strongest correction.
        # Future days receive progressively weaker correction.

        day1_strength = 0.70
        day2_strength = 0.55
        day3_strength = 0.40

        pred_day1 = _anchor_prediction(
            raw_day1,
            live_aqi,
            day1_strength
        )

        pred_day2 = _anchor_prediction(
            raw_day2,
            live_aqi,
            day2_strength
        )

        pred_day3 = _anchor_prediction(
            raw_day3,
            live_aqi,
            day3_strength
        )

    else:

        pred_day1 = raw_day1
        pred_day2 = raw_day2
        pred_day3 = raw_day3

    # ======================================================
    # SAFETY LIMITS
    # ======================================================

    pred_day1 = max(0.0, pred_day1)
    pred_day2 = max(0.0, pred_day2)
    pred_day3 = max(0.0, pred_day3)

    average_aqi = (
        pred_day1
        + pred_day2
        + pred_day3
    ) / 3.0

    print(
        f"\nPrediction correction:"
        f"\n  Live AQI  : {live_aqi}"
        f"\n  Raw Day1  : {raw_day1:.2f}"
        f"\n  Final Day1: {pred_day1:.2f}"
        f"\n  Raw Day2  : {raw_day2:.2f}"
        f"\n  Final Day2: {pred_day2:.2f}"
        f"\n  Raw Day3  : {raw_day3:.2f}"
        f"\n  Final Day3: {pred_day3:.2f}"
    )

    return {

        "day1": {
            "predicted_aqi": pred_day1,
            "model_used":
                _MANIFEST["day1"]["model_type"]
        },

        "day2": {
            "predicted_aqi": pred_day2,
            "model_used":
                _MANIFEST["day2"]["model_type"]
        },

        "day3": {
            "predicted_aqi": pred_day3,
            "model_used":
                _MANIFEST["day3"]["model_type"]
        },

        "average_aqi": average_aqi
    }