"""
predict.py — matches training_pipeline v5
Day1 = absolute prediction
Day2/3 = residual prediction → add to previous prediction
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
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(MANIFEST_PATH):
        raise RuntimeError("Model files missing. Run training_pipeline.py first.")
    _FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
    with open(MANIFEST_PATH) as f:
        _MANIFEST = json.load(f)


def _predict_raw(horizon_name, df):
    info = _MANIFEST[horizon_name]
    if info["kind"] == "sklearn":
        model = joblib.load(info["file"])
        return float(model.predict(df)[0])
    else:
        scaler = joblib.load(info["scaler_file"])
        scaled_x = scaler.transform(df)
        model = AQINet(len(_FEATURE_COLUMNS))
        model.load_state_dict(torch.load(info["file"], map_location="cpu"))
        model.eval()
        with torch.no_grad():
            scaled_pred = float(model(torch.tensor(scaled_x, dtype=torch.float32)).item())
        return (scaled_pred * info["target_std"]) + info["target_mean"]


def predict_all_horizons(features: dict):
    """
    features must contain all columns that appear in model_features.pkl
    except prev_aqi (filled automatically).  It must also contain the key
    "aqi" = today's observed AQI.
    """
    _ensure_loaded()

    row = {col: float(features.get(col, 0.0)) for col in _FEATURE_COLUMNS}

    # Day 1 – absolute, prev_aqi = today's AQI
    row["prev_aqi"] = float(features.get("aqi", 0.0))
    df = pd.DataFrame([row])[_FEATURE_COLUMNS]
    pred_day1 = _predict_raw("day1", df)

    # Day 2 – residual, add to day1
    row["prev_aqi"] = pred_day1
    df = pd.DataFrame([row])[_FEATURE_COLUMNS]
    delta2 = _predict_raw("day2", df)
    pred_day2 = pred_day1 + delta2 if _MANIFEST["day2"]["is_delta"] else delta2

    # Day 3 – residual, add to day2
    row["prev_aqi"] = pred_day2
    df = pd.DataFrame([row])[_FEATURE_COLUMNS]
    delta3 = _predict_raw("day3", df)
    pred_day3 = pred_day2 + delta3 if _MANIFEST["day3"]["is_delta"] else delta3

    return {
        "day1": {"predicted_aqi": pred_day1, "model_used": _MANIFEST["day1"]["model_type"]},
        "day2": {"predicted_aqi": pred_day2, "model_used": _MANIFEST["day2"]["model_type"]},
        "day3": {"predicted_aqi": pred_day3, "model_used": _MANIFEST["day3"]["model_type"]},
        "average_aqi": (pred_day1 + pred_day2 + pred_day3) / 3.0,
    }