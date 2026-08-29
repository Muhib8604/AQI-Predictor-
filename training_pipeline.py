

import os
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="joblib")

import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from feature_store import connect_feature_store
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_definition import AQINet

load_dotenv()

TRAINING_DATA_PATH = "training_dataset.csv"
N_SPLITS = 5

PYTORCH_PARAM_CANDIDATES = [
    {"epochs": 120, "lr": 0.0015, "weight_decay": 1e-3},
    {"epochs": 180, "lr": 0.001,  "weight_decay": 5e-4},
    {"epochs": 220, "lr": 0.0008, "weight_decay": 1e-3},
    {"epochs": 280, "lr": 0.0005, "weight_decay": 1e-4},
]


def recency_weights(dates, half_life_days=75):
    """
    Recent rows get higher weight.
    half_life_days=75 → ~2.5 months purani row ka weight ~0.5.
    """
    dates = pd.to_datetime(dates)
    age_days = (dates.max() - dates).dt.days.astype(float)
    return np.exp(-np.log(2) * age_days / half_life_days)

def prepare_clean_dataset(df_raw):

    # Make a working copy BEFORE using df
    df = df_raw.copy()

    print("\n===== AQI DISTRIBUTION =====")
    print(df["aqi"].describe())

    print("\n===== AQI RANGES =====")
    print("AQI < 20:", (df["aqi"] < 20).sum())
    print("AQI 20-40:", ((df["aqi"] >= 20) & (df["aqi"] < 40)).sum())
    print("AQI 40-60:", ((df["aqi"] >= 40) & (df["aqi"] < 60)).sum())
    print("AQI 60-80:", ((df["aqi"] >= 60) & (df["aqi"] < 80)).sum())
    print("AQI 80-100:", ((df["aqi"] >= 80) & (df["aqi"] < 100)).sum())
    print("AQI 100+:", (df["aqi"] >= 100).sum())
    print("============================\n")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Daily aggregation
    df = df.resample("D").mean(numeric_only=True)

    
    df = df.dropna(subset=["aqi"])

    # Interpolate/fill only non-target features
    feature_fill_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    for col in feature_fill_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time")
            df[col] = df[col].ffill().bfill()

    df = df.reset_index()

    
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    aqi_lag1 = df["aqi"].shift(1)

    df["aqi_rolling_mean_3"] = aqi_lag1.rolling(3).mean()
    df["aqi_rolling_std_3"] = aqi_lag1.rolling(3).std()

    df["aqi_rolling_mean_7"] = aqi_lag1.rolling(7).mean()
    df["aqi_rolling_std_7"] = aqi_lag1.rolling(7).std()

    df["aqi_rolling_mean_14"] = aqi_lag1.rolling(14).mean()
    df["aqi_rolling_std_14"] = aqi_lag1.rolling(14).std()

    df["aqi_ema_3"] = aqi_lag1.ewm(span=3, adjust=False).mean()
    df["aqi_ema_7"] = aqi_lag1.ewm(span=7, adjust=False).mean()

    df["aqi_change"] = df["aqi_lag_1"] - df["aqi_lag_2"]
    df["aqi_change_3"] = df["aqi_lag_1"] - df["aqi_lag_3"]
    df["aqi_change_7"] = df["aqi_lag_1"] - df["aqi_lag_7"]

    
    for col in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_change_1"] = df[col] - df[col].shift(1)

    
    for col in ["temperature", "humidity", "wind_speed", "pressure"]:
        df[f"{col}_change_1"] = df[col] - df[col].shift(1)
        df[f"{col}_change_3"] = df[col] - df[col].shift(3)

    # Remove rows where lag/rolling features aren't available
    df = df.dropna().reset_index(drop=True)

    return df

FEATURE_COLS = [
    "temperature", "humidity", "pressure", "wind_speed", "rain",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
    "month", "day", "day_of_week", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_5", "aqi_lag_7", "aqi_lag_14",
    "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "aqi_rolling_mean_7", "aqi_rolling_std_7",
    "aqi_rolling_mean_14", "aqi_rolling_std_14",
    "aqi_ema_3", "aqi_ema_7",
    "aqi_change", "aqi_change_3", "aqi_change_7",
    "pm2_5_lag_1", "pm10_lag_1", "ozone_lag_1", "nitrogen_dioxide_lag_1",
    "pm2_5_change_1", "pm10_change_1", "ozone_change_1", "nitrogen_dioxide_change_1",
    "temperature_change_1", "humidity_change_1", "wind_speed_change_1", "pressure_change_1",
    "temperature_change_3", "humidity_change_3", "wind_speed_change_3", "pressure_change_3",
]


def _pooled_metrics(actuals, preds):
    actuals = np.asarray(actuals)
    preds   = np.asarray(preds)
    return {
        "mae":  float(mean_absolute_error(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "r2":   float(r2_score(actuals, preds)),
    }



def train_random_forest(X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight):
    param_grid = {
        "n_estimators":     [200, 300],
        "max_depth":        [8, 12, None],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", 0.7],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y_target, sample_weight=sample_weight)
    best_rf = search.best_estimator_

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_level = pd.Series(np.nan, index=X.index, dtype=float)
    pooled_actuals, pooled_preds = [], []

    for tr, te in tscv.split(X):
        m = clone(best_rf).fit(X.iloc[tr], y_target.iloc[tr], sample_weight=sample_weight.iloc[tr])
        preds = m.predict(X.iloc[te])
        recon = (base_pred.iloc[te].values + preds) if is_delta else preds
        oof_level.iloc[te] = recon
        valid = eval_mask.iloc[te].values
        if valid.any():
            pooled_actuals.extend(y_true_level.iloc[te].values[valid])
            pooled_preds.extend(np.asarray(recon)[valid])

    metrics = _pooled_metrics(pooled_actuals, pooled_preds)
    final_model = clone(best_rf).fit(X, y_target, sample_weight=sample_weight)
    return final_model, metrics, oof_level


def train_ridge(X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    param_grid = {"ridge__alpha": [0.5, 1.0, 3.0, 10.0, 30.0, 100.0]}
    search = GridSearchCV(
        pipe, param_grid,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y_target, **{"ridge__sample_weight": sample_weight})
    best_pipe = search.best_estimator_

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_level = pd.Series(np.nan, index=X.index, dtype=float)
    pooled_actuals, pooled_preds = [], []

    for tr, te in tscv.split(X):
        m = clone(best_pipe).fit(X.iloc[tr], y_target.iloc[tr], **{"ridge__sample_weight": sample_weight.iloc[tr]})
        preds = m.predict(X.iloc[te])
        recon = (base_pred.iloc[te].values + preds) if is_delta else preds
        oof_level.iloc[te] = recon
        valid = eval_mask.iloc[te].values
        if valid.any():
            pooled_actuals.extend(y_true_level.iloc[te].values[valid])
            pooled_preds.extend(np.asarray(recon)[valid])

    metrics = _pooled_metrics(pooled_actuals, pooled_preds)
    final_model = clone(best_pipe).fit(X, y_target, **{"ridge__sample_weight": sample_weight})
    return final_model, metrics, oof_level


def _fit_pytorch_once(X_tr, y_tr, X_te, epochs, lr, weight_decay, sample_weight=None):
    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)

    yt_m = float(y_tr.mean())
    yt_s = float(y_tr.std()) if y_tr.std() > 1e-6 else 1.0
    yt_tr_s = (y_tr.values - yt_m) / yt_s

    model = AQINet(X_tr.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0, reduction="none")

    if sample_weight is None:
        w = torch.ones(len(y_tr), dtype=torch.float32)
    else:
        w = torch.tensor(np.asarray(sample_weight, dtype=np.float32))
        w = w / (w.mean() + 1e-8)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(torch.tensor(X_tr_s, dtype=torch.float32))
        target = torch.tensor(yt_tr_s, dtype=torch.float32).view(-1, 1)
        per_sample = criterion(out, target).view(-1)
        loss = (per_sample * w).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_s = model(torch.tensor(X_te_s, dtype=torch.float32)).numpy().flatten()
    raw_preds = (preds_s * yt_s) + yt_m
    return model, raw_preds, x_scaler, yt_m, yt_s


def train_pytorch(X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight,
                  candidates=PYTORCH_PARAM_CANDIDATES):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    best = None

    for params in candidates:
        oof_level = pd.Series(np.nan, index=X.index, dtype=float)
        pooled_actuals, pooled_preds = [], []

        for tr, te in tscv.split(X):
            _, raw_preds, _, _, _ = _fit_pytorch_once(
                X.iloc[tr], y_target.iloc[tr], X.iloc[te], sample_weight=sample_weight.iloc[tr], **params
            )
            recon = (base_pred.iloc[te].values + raw_preds) if is_delta else raw_preds
            oof_level.iloc[te] = recon
            valid = eval_mask.iloc[te].values
            if valid.any():
                pooled_actuals.extend(y_true_level.iloc[te].values[valid])
                pooled_preds.extend(np.asarray(recon)[valid])

        metrics = _pooled_metrics(pooled_actuals, pooled_preds)
        if best is None or metrics["mae"] < best["metrics"]["mae"]:
            best = {"params": params, "metrics": metrics, "oof_level": oof_level}

    final_model, _, x_scaler, yt_m, yt_s = _fit_pytorch_once(
        X, y_target, X.tail(5), sample_weight=sample_weight, **best["params"]
    )
    return (final_model, x_scaler, yt_m, yt_s,
            best["metrics"], best["oof_level"], best["params"])


def run_tournament_for_horizon(daily_df, horizon_name, h, upstream_oof_level_by_date):
    print(f"\n=== Tournament for {horizon_name} (AQI {h} day(s) ahead) ===")

    df_h = daily_df.copy()
    df_h["true_level"] = df_h["aqi"].shift(-h)

    if h == 1:
        df_h["target"] = df_h["true_level"]
        is_delta = False
    else:
        prev_actual = df_h["aqi"].shift(-(h - 1))
        df_h["target"] = df_h["true_level"] - prev_actual
        is_delta = True

    df_h = df_h.dropna(subset=FEATURE_COLS + ["target", "true_level"]).reset_index(drop=True)

    if is_delta:
        df_h["base_pred"] = df_h["date"].map(upstream_oof_level_by_date)
    else:
        df_h["base_pred"] = 0.0

    eval_mask = df_h["base_pred"].notna() if is_delta else pd.Series(True, index=df_h.index)

    print(f"  training rows (full): {len(df_h)}")
    if is_delta:
        print(f"  honest-eval rows (chained): {int(eval_mask.sum())}")
    
    sample_weight = pd.Series(
        recency_weights(df_h["date"], half_life_days=75),
        index=df_h.index,
    )
    print(f"  sample_weight: min={sample_weight.min():.3f} max={sample_weight.max():.3f} mean={sample_weight.mean():.3f}")
    
    X            = df_h[FEATURE_COLS]
    y_target     = df_h["target"]
    y_true_level = df_h["true_level"]
    base_pred    = df_h["base_pred"]

    rf_m, rf_metrics, rf_oof = train_random_forest(
        X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight
    )
    print(f"  RandomForest CV: MAE={rf_metrics['mae']:6.2f} | R2={rf_metrics['r2']:.3f}")

    ridge_m, ridge_metrics, ridge_oof = train_ridge(
        X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight
    )
    print(f"  Ridge CV: MAE={ridge_metrics['mae']:6.2f} | R2={ridge_metrics['r2']:.3f}")

    (pt_m, pt_scaler, pt_m_val, pt_s_val,
     pt_metrics, pt_oof, pt_params) = train_pytorch(
        X, y_target, y_true_level, base_pred, eval_mask, is_delta, sample_weight
    )
    print(f"  PyTorch CV: MAE={pt_metrics['mae']:6.2f} | R2={pt_metrics['r2']:.3f} params={pt_params}")

    candidates = {
        "RandomForest": {"model": rf_m, "kind": "sklearn", "oof": rf_oof, **rf_metrics},
        "Ridge":        {"model": ridge_m, "kind": "sklearn", "oof": ridge_oof, **ridge_metrics},
        "PyTorch": {
            "model": pt_m, "kind": "pytorch",
            "scaler": pt_scaler,
            "target_mean": pt_m_val, "target_std": pt_s_val,
            "oof": pt_oof, **pt_metrics
        },
    }

    champion_name = min(candidates, key=lambda k: candidates[k]["mae"])
    champion_data = candidates[champion_name]
    print(f"  -> Champion for {horizon_name}: {champion_name} "
          f"(CV MAE={champion_data['mae']:.2f}, R2={champion_data['r2']:.3f})")

    all_candidate_metrics = {
        name: {"mae": c["mae"], "rmse": c["rmse"], "r2": c["r2"]}
        for name, c in candidates.items()
    }

    oof_level_by_date = pd.Series(champion_data["oof"].values, index=df_h["date"].values)
    return champion_name, champion_data, is_delta, oof_level_by_date, all_candidate_metrics


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Karachi_AQI_Chained_Tournament")

    # Connect to Hopsworks
    FEATURE_GROUP_NAME = "aqi_features"
    FEATURE_GROUP_VERSION = 2
    fs = connect_feature_store()

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    
    df_raw = None
    last_error = None

    for attempt in range(1, 4):
        try:
            print(
                f"Read attempt {attempt}/3 "
                f"(Feature Query Service)..."
            )

            df_raw = feature_group.read(
                read_options={
                    "arrow_flight_config": {
                        "timeout": 90
                    }
                }
            )

            print("Feature Query Service read succeeded.")
            break

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} failed: {e}")

            if attempt < 3:
                import time
                wait_time = 10 * attempt
                print(
                    f"Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)

    
    if df_raw is None:
        try:
            print("Falling back to Hive read...")

            df_raw = feature_group.read(
                read_options={
                    "use_hive": True
                }
            )

            print("Hive read succeeded.")

        except Exception as hive_err:
            last_error = hive_err
            print(f"Hive also failed: {hive_err}")

    
    if df_raw is None or df_raw.empty:
        raise RuntimeError(
            "Could not read feature group after "
            f"retries and Hive fallback. "
            f"Last error: {last_error}"
        )

    raw_row_count = len(df_raw)

    print(
        f"Raw rows read from Hopsworks V2: "
        f"{raw_row_count}"
    )

    if raw_row_count < 23:
        raise RuntimeError(
            f"Not enough historical daily data. "
            f"Need at least 23 raw daily rows for the "
            f"3-day horizon with 5-fold CV. "
            f"Found {raw_row_count} rows."
        )

    if not df_raw.empty:
        print(
            f"Raw date range: "
            f"{df_raw['date'].min()} -> "
            f"{df_raw['date'].max()}"
        )

    df = prepare_clean_dataset(df_raw)

    print(
        f"Training data source: Hopsworks Feature Store V2 "
        f"({len(df)} clean daily rows)"
    )
    manifest = {}
    comparison = {}
    upstream_oof_level_by_date = None

    for horizon_name, h in [("day1", 1), ("day2", 2), ("day3", 3)]:
        champ_name, champ_data, is_delta, oof_level_by_date, all_candidate_metrics = run_tournament_for_horizon(
            df, horizon_name, h, upstream_oof_level_by_date
        )
        upstream_oof_level_by_date = oof_level_by_date
        comparison[horizon_name] = {
            "candidates": all_candidate_metrics,
            "champion": champ_name,
        }

        if champ_data["kind"] == "sklearn":
            file_name = f"aqi_model_{horizon_name}.pkl"
            joblib.dump(champ_data["model"], file_name)
            manifest[horizon_name] = {
                "model_type": champ_name, "kind": "sklearn", "file": file_name,
                "is_delta": is_delta, "mae": champ_data["mae"], "rmse": champ_data["rmse"], "r2": champ_data["r2"],
            }
        else:
            file_name   = f"aqi_model_{horizon_name}.pt"
            scaler_name = f"aqi_scaler_{horizon_name}.pkl"
            torch.save(champ_data["model"].state_dict(), file_name)
            joblib.dump(champ_data["scaler"], scaler_name)
            manifest[horizon_name] = {
                "model_type": champ_name, "kind": "pytorch",
                "file": file_name, "scaler_file": scaler_name,
                "target_mean": champ_data["target_mean"],
                "target_std":   champ_data["target_std"],
                "is_delta": is_delta,
                "mae": champ_data["mae"], "rmse": champ_data["rmse"], "r2": champ_data["r2"],
            }

        manifest[horizon_name]["evaluation"] = (
            "Day1 = absolute. Day2/3 = residual + prev_aqi feature. "
            "All rows used. Metrics on reconstructed absolute AQI."
        )

        with mlflow.start_run(run_name=f"{horizon_name}_{champ_name}"):
            mlflow.set_tags({"horizon": horizon_name, "model_type": champ_name})
            mlflow.log_metrics({
                "MAE": champ_data["mae"],
                "RMSE": champ_data["rmse"],
                "R2": champ_data["r2"],
            })

    with open("model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open("model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    joblib.dump(FEATURE_COLS, "model_features.pkl")

    print("\nSaved updated model_manifest.json, model_comparison.json, and model_features.pkl.")
    print("\nFinal Lineup (Tournament Winners):")
    for h_name in ["day1", "day2", "day3"]:
        c = manifest[h_name]
        print(f"  {h_name}: {c['model_type']} (MAE={c['mae']:.2f}, R2={c['r2']:.3f})")


if __name__ == "__main__":
    main()