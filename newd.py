import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Optional explainability dependencies. The dashboard still runs if SHAP artifacts
# are not present; it will show a clear message instead of crashing.
try:
    import joblib
except Exception:
    joblib = None

try:
    import shap
except Exception:
    shap = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM DARK / GLASS UI
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --panel: rgba(19, 32, 52, 0.78);
    --panel2: rgba(24, 40, 64, 0.78);
    --border: rgba(148, 163, 184, 0.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --cyan: #22d3ee;
    --blue: #38bdf8;
    --purple: #818cf8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(34,211,238,.10), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(129,140,248,.12), transparent 30%),
        linear-gradient(135deg, #050b14 0%, #07111f 45%, #0b1729 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    line-height: 1.08;
    background: linear-gradient(90deg, #22d3ee, #38bdf8, #818cf8, #22d3ee);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 7s ease infinite;
}

@keyframes shine {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.live-badge, .chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid rgba(34,211,238,.28);
    background: rgba(34,211,238,.08);
    color: #a5f3fc;
    font-size: .76rem;
    font-weight: 700;
}

.live-badge { margin-left: 12px; vertical-align: middle; }

.pulse-dot, .monitor-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 rgba(34,197,94,.6);
    animation: pulse 1.7s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,.55); }
    70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

.glass-card, .metric-card {
    background: linear-gradient(145deg, rgba(25,42,66,.82), rgba(11,24,41,.82));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,.20);
    backdrop-filter: blur(16px);
}

.glass-card:hover, .metric-card:hover {
    border-color: rgba(56,189,248,.32);
}

.forecast-card {
    background: linear-gradient(160deg, #182840, #0d192a);
    border: 1px solid #26384f;
    border-radius: 18px;
    padding: 20px;
    min-height: 310px;
    box-shadow: 0 12px 30px rgba(0,0,0,.22);
}

.forecast-aqi {
    font-family: 'Poppins', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 7px 0 0;
}

.monitor {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 12px 0 18px;
    padding: 11px 15px;
    border: 1px solid rgba(56,189,248,.24);
    background: rgba(56,189,248,.07);
    border-radius: 12px;
    color: #bfdbfe;
}

.alert-banner {
    border: 1px solid rgba(239,68,68,.55);
    border-left: 5px solid #ef4444;
    background: linear-gradient(90deg, rgba(239,68,68,.20), rgba(239,68,68,.07));
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0 18px;
    color: #fecaca;
}

.section-note {
    color: var(--muted);
    font-size: .9rem;
}

.small-muted { color: #94a3b8; font-size: .78rem; }

.legend-wrap {
    display: flex;
    height: 12px;
    overflow: hidden;
    border-radius: 999px;
    margin: 12px 0 5px;
}

.legend-seg { flex: 1; }
.legend-labels {
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
    font-size: .68rem;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 12px 0 0;
    font-size: .78rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1423 0%, #060d17 100%);
    border-right: 1px solid rgba(148,163,184,.10);
}

[data-testid="stMetric"] {
    background: rgba(24,40,64,.62);
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 16px;
    padding: 14px 16px;
}

button[kind="primary"] {
    border-radius: 10px;
}

.stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
}

hr { border-color: rgba(148,163,184,.12); }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS / PROJECT SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "model_manifest.json"
ALERT_LOG_PATH = BASE_DIR / "alert_log.csv"
DEFAULT_BACKEND = "http://127.0.0.1:8000"
SADDAR_LAT = 24.8576
SADDAR_LON = 67.0302
VERSION = "1.0"

FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "rain",
    "pm25",
    "pm10",
    "ozone",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
]

AQI_BANDS = [
    (50, "🟢 Good", "#22c55e"),
    (100, "🟡 Moderate", "#eab308"),
    (150, "🟠 Unhealthy for Sensitive Groups", "#f97316"),
    (200, "🔴 Unhealthy", "#ef4444"),
    (300, "🟣 Very Unhealthy", "#a855f7"),
    (float("inf"), "⚫ Hazardous", "#4b5563"),
]

POLLUTANT_LIMITS = {
    "pm25": 60.0,
    "pm10": 100.0,
    "ozone": 100.0,
}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=35, r=20, t=55, b=35),
    font=dict(family="Inter", color="#e2e8f0"),
)


# ============================================================
# HELPERS
# ============================================================
def safe_float(value, default=None):
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def aqi_status(aqi):
    value = safe_float(aqi, 0.0)
    for upper, label, _ in AQI_BANDS:
        if value <= upper:
            return label
    return "⚫ Hazardous"


def aqi_color(aqi):
    value = safe_float(aqi, 0.0)
    for upper, _, color in AQI_BANDS:
        if value <= upper:
            return color
    return "#4b5563"


def fmt_number(value, digits=1, fallback="N/A"):
    number = safe_float(value)
    if number is None:
        return fallback
    return f"{number:.{digits}f}"


def get_value(row, key, default=None):
    value = row.get(key, default) if isinstance(row, dict) else default
    return value if value is not None else default


def load_json(path):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def model_name(manifest, day_key):
    if not manifest or not isinstance(manifest.get(day_key), dict):
        return "Unknown"
    return str(manifest[day_key].get("model_type") or manifest[day_key].get("model") or "Unknown")


def model_metric(manifest, day_key, metric):
    if not manifest or not isinstance(manifest.get(day_key), dict):
        return None
    return safe_float(manifest[day_key].get(metric))


def average_metric(manifest, metric):
    values = [model_metric(manifest, day, metric) for day in ("day1", "day2", "day3")]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def backend_url():
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND).rstrip("/")


# ============================================================
# ALERT LOG
# ============================================================
def log_alert(forecast_date, predicted_aqi, threshold):
    value = safe_float(predicted_aqi)
    if value is None:
        return
    entry = pd.DataFrame(
        [{
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_date": str(forecast_date),
            "predicted_aqi": round(value, 1),
            "threshold": int(threshold),
            "status": aqi_status(value),
        }]
    )

    try:
        if ALERT_LOG_PATH.exists():
            existing = pd.read_csv(ALERT_LOG_PATH)
            for col in ["forecast_date", "predicted_aqi"]:
                if col not in existing.columns:
                    existing[col] = None
            existing["predicted_aqi"] = pd.to_numeric(existing["predicted_aqi"], errors="coerce").round(1)
            duplicate = existing[
                (existing["forecast_date"].astype(str) == str(forecast_date))
                & (existing["predicted_aqi"] == round(value, 1))
            ]
            if not duplicate.empty:
                return
            entry = pd.concat([existing, entry], ignore_index=True)
        entry.to_csv(ALERT_LOG_PATH, index=False)
    except (OSError, ValueError, pd.errors.ParserError):
        pass


def load_alert_log():
    columns = ["logged_at", "forecast_date", "predicted_aqi", "threshold", "status"]
    try:
        if ALERT_LOG_PATH.exists():
            df = pd.read_csv(ALERT_LOG_PATH)
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df[columns]
    except (OSError, pd.errors.ParserError):
        pass
    return pd.DataFrame(columns=columns)


# ============================================================
# FORECAST FETCH
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast(url):
    started = time.perf_counter()
    try:
        response = requests.get(f"{url}/predict", timeout=20)
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body, elapsed_ms, None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return None, None, elapsed_ms, str(exc)


# ============================================================
# SHAP / EXPLAINABILITY HELPERS
# ============================================================
def resolve_model_path(day_key, info):
    candidates = []
    if isinstance(info, dict):
        for key in ("model_path", "path", "file", "model_file", "artifact"):
            value = info.get(key)
            if value:
                candidates.append(Path(str(value)))

    candidates.extend([
        BASE_DIR / f"{day_key}_model.pkl",
        BASE_DIR / f"model_{day_key}.pkl",
        BASE_DIR / "models" / f"{day_key}_model.pkl",
        BASE_DIR / "models" / f"{day_key}.pkl",
        BASE_DIR / "models" / f"model_{day_key}.pkl",
        BASE_DIR / "aqi_prediction_model.pkl" if day_key == "day1" else BASE_DIR / "aqi_prediction_model.pkl",
    ])

    for path in candidates:
        if not path.is_absolute():
            path = BASE_DIR / path
        if path.exists():
            return path
    return None


def load_feature_names():
    candidates = [
        BASE_DIR / "model_features.pkl",
        BASE_DIR / "features.pkl",
        BASE_DIR / "training_features.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".pkl" and joblib is not None:
                data = joblib.load(path)
                if isinstance(data, (list, tuple)):
                    return [str(x) for x in data]
                if hasattr(data, "columns"):
                    return [str(x) for x in data.columns]
            elif path.suffix == ".csv":
                return [str(x) for x in pd.read_csv(path, nrows=1).columns]
        except Exception:
            continue
    return None


def model_feature_importance(day_key, info):
    """Return native model feature importance when the champion exposes it."""
    if joblib is None:
        return None
    model_path = resolve_model_path(day_key, info)
    if model_path is None:
        return None
    try:
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            return None
        values = list(model.feature_importances_)
        names = list(getattr(model, "feature_names_in_", []))
        if len(names) != len(values):
            names = load_feature_names() or FEATURES
        if len(names) != len(values):
            names = [f"Feature {i + 1}" for i in range(len(values))]
        return pd.Series(values, index=[str(x) for x in names]).sort_values(ascending=False)
    except Exception:
        return None


def shap_importance_from_local_model(day_key, info):
    if joblib is None or shap is None:
        return None, "SHAP or joblib is not installed."

    model_path = resolve_model_path(day_key, info)
    if model_path is None:
        return None, "No local model artifact was found for this forecast horizon."

    training_path = BASE_DIR / "training_features.csv"
    if not training_path.exists():
        return None, "training_features.csv was not found."

    try:
        model = joblib.load(model_path)
        X = pd.read_csv(training_path)
        feature_names = list(X.columns)

        # Keep the sample small enough for a dashboard while retaining a useful
        # explanation. SHAP handles a full matrix, but this is much faster.
        sample = X.tail(min(300, len(X))).copy()
        if sample.empty:
            return None, "training_features.csv contains no rows."

        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            common = [c for c in expected if c in sample.columns]
            if common:
                sample = sample[common]
                feature_names = common

        if "RandomForest" in type(model).__name__ or "ExtraTrees" in type(model).__name__ or hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(sample)
        elif "Ridge" in type(model).__name__ or "Linear" in type(model).__name__:
            explainer = shap.LinearExplainer(model, sample)
            values = explainer.shap_values(sample)
        else:
            explainer = shap.Explainer(model, sample)
            values = explainer(sample).values

        if isinstance(values, list):
            values = values[0]
        values = getattr(values, "values", values)
        importance = pd.Series(abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False)
        return importance, None
    except Exception as exc:
        return None, f"SHAP could not evaluate the local model: {exc}"


# ============================================================
# LOAD CONFIG
# ============================================================
manifest = load_json(MANIFEST_PATH)
avg_mae = average_metric(manifest, "mae")
avg_rmse = average_metric(manifest, "rmse")
avg_r2 = average_metric(manifest, "r2")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🌍 Karachi AQI Predictor")
st.sidebar.caption(f"Saddar, Karachi • Version {VERSION}")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Location")
st.sidebar.success("Saddar, Karachi")
st.sidebar.caption(f"Coordinates: {SADDAR_LAT:.4f}, {SADDAR_LON:.4f}")

st.sidebar.markdown("### 🤖 Champion Models")
if manifest:
    for day in ("day1", "day2", "day3"):
        st.sidebar.write(f"**{day.replace('day', 'Day ')}:** {model_name(manifest, day)}")
else:
    st.sidebar.warning("model_manifest.json not found")

st.sidebar.markdown("### 📈 Performance")
if avg_r2 is not None:
    st.sidebar.metric("Average R²", f"{avg_r2:.3f}")
if avg_mae is not None:
    st.sidebar.metric("Average MAE", f"{avg_mae:.2f}")
if avg_rmse is not None:
    st.sidebar.metric("Average RMSE", f"{avg_rmse:.2f}")

st.sidebar.markdown("### ⚙️ System")
st.sidebar.write(f"**Forecast horizon:** 3 days")
st.sidebar.write(f"**Input features:** {len(FEATURES)}")
st.sidebar.write("**Backend:** FastAPI")
st.sidebar.write("**Data:** OpenWeather + air-pollution forecast")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Alert Settings")
alert_threshold = st.sidebar.slider(
    "Hazard alert threshold (AQI)",
    min_value=50,
    max_value=300,
    value=150,
    step=10,
    help="An alert is triggered when any forecast day reaches or exceeds this AQI.",
)
enable_toast = st.sidebar.checkbox("🔔 Toast notification on alert", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Forecast", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Last dashboard refresh: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")


# ============================================================
# HERO
# ============================================================
st.markdown(
    '<span class="hero-title">🌍 Karachi Air Quality Prediction Dashboard</span>'
    '<span class="live-badge"><span class="pulse-dot"></span> LIVE</span>',
    unsafe_allow_html=True,
)
st.caption(
    "AI-powered 3-Day AQI Forecast • Weather + Air Pollution Forecast • Machine Learning"
)
st.markdown(
    f'<div class="monitor"><span class="monitor-dot"></span>'
    f'<span>🛡️ Hazard monitoring active — threshold <b>{alert_threshold} AQI</b></span></div>',
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📍 Location", "Saddar", "Karachi")
k2.metric("🤖 Forecast", "3 Days", "1 model per day")
k3.metric("🎯 Avg R²", f"{avg_r2:.3f}" if avg_r2 is not None else "N/A", "Validation")
k4.metric("📏 Avg MAE", f"{avg_mae:.2f}" if avg_mae is not None else "N/A", "AQI points")
k5.metric("🕒 Updated", datetime.now().strftime("%H:%M"), datetime.now().strftime("%d %b"))

st.markdown(
    """
<div class="legend-wrap">
  <div class="legend-seg" style="background:#22c55e"></div>
  <div class="legend-seg" style="background:#eab308"></div>
  <div class="legend-seg" style="background:#f97316"></div>
  <div class="legend-seg" style="background:#ef4444"></div>
  <div class="legend-seg" style="background:#a855f7"></div>
  <div class="legend-seg" style="background:#4b5563"></div>
</div>
<div class="legend-labels">
  <span>0 Good</span><span>50 Moderate</span><span>100 USG</span>
  <span>150 Unhealthy</span><span>200 Very Unhealthy</span><span>300+ Hazardous</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FETCH BACKEND
# ============================================================
status_code, payload, response_ms, fetch_error = fetch_forecast(backend_url())

if fetch_error:
    st.error(f"Unable to reach FastAPI backend at {backend_url()}/predict")
    st.code(fetch_error)
    st.info("Start the FastAPI server and then use **Refresh Forecast** in the sidebar.")
    st.stop()

if status_code != 200:
    detail = payload.get("error") if isinstance(payload, dict) else None
    st.error(f"FastAPI returned HTTP {status_code}.")
    if detail:
        st.warning(str(detail))
    st.info("Check the backend terminal for the exact API error, then refresh the dashboard.")
    st.stop()

if not isinstance(payload, dict):
    st.error("The backend returned an invalid JSON response.")
    st.stop()

if "error" in payload and "3_day_AQI_forecast" not in payload:
    st.error(f"⚠️ Backend error: {payload.get('error')}")
    st.info(
        "If the error says that models are missing, run the training pipeline so that "
        "model_manifest.json and the per-day model artifacts are generated."
    )
    st.stop()

forecast_data = payload.get("3_day_AQI_forecast", [])
if not isinstance(forecast_data, list) or not forecast_data:
    st.error("No 3-day forecast was returned by the backend.")
    st.stop()

# Normalize the backend response without assuming every optional field exists.
df = pd.DataFrame(forecast_data).copy()
if "date" not in df.columns or "predicted_aqi" not in df.columns:
    st.error("Backend response is missing the required fields: date and predicted_aqi.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["predicted_aqi"] = pd.to_numeric(df["predicted_aqi"], errors="coerce")
df = df.dropna(subset=["date", "predicted_aqi"]).reset_index(drop=True)
if df.empty:
    st.error("The forecast contains no valid AQI values.")
    st.stop()

latest = float(df.iloc[0]["predicted_aqi"])
live_aqi = safe_float(payload.get("live_aqi"))
live_station = payload.get("live_station_name") or "configured station"


# ============================================================
# ALERT CHECK
# ============================================================
exceeded = df[df["predicted_aqi"] >= alert_threshold]
if not exceeded.empty:
    worst = exceeded.loc[exceeded["predicted_aqi"].idxmax()]
    st.markdown(
        f'<div class="alert-banner">🚨 <b>HAZARD ALERT</b> — '
        f'{worst["date"].strftime("%d %b")} forecast AQI is '
        f'<b>{worst["predicted_aqi"]:.0f}</b>, meeting/exceeding your '
        f'<b>{alert_threshold}</b> threshold. Status: {aqi_status(worst["predicted_aqi"])}</div>',
        unsafe_allow_html=True,
    )
    if enable_toast:
        st.toast(f"Hazard AQI forecast: {worst['predicted_aqi']:.0f}", icon="🚨")
    for _, row in exceeded.iterrows():
        log_alert(row["date"].strftime("%Y-%m-%d"), row["predicted_aqi"], alert_threshold)


# ============================================================
# TABS
# ============================================================
tab_overview, tab_forecast, tab_explain, tab_alerts, tab_models = st.tabs(
    ["🏠 Overview", "📈 Forecast & Trends", "🧠 Explainability", "🚨 Alerts & Data", "🤖 Models & System"]
)


# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:
    lc1, lc2, lc3 = st.columns(3)
    if live_aqi is not None:
        lc1.metric(f"📡 Live AQI ({live_station})", f"{live_aqi:.0f}")
        lc2.metric("🔮 Predicted AQI — Day 1", f"{latest:.1f}")
        lc3.metric("Δ Difference", f"{latest - live_aqi:+.1f}", "Predicted − live")
    else:
        lc1.warning("Live station reading is unavailable right now.")
        lc2.metric("🔮 Predicted AQI — Day 1", f"{latest:.1f}")
        lc3.metric("API response", f"{response_ms:.0f} ms")

    st.markdown("### 📊 Current Forecast Snapshot")
    left, right = st.columns([1, 2])

    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest,
                number={"font": {"size": 48}},
                title={"text": "Day 1 Forecast AQI"},
                gauge={
                    "axis": {"range": [0, max(300, latest * 1.15)]},
                    "bar": {"color": aqi_color(latest)},
                    "steps": [
                        {"range": [0, 50], "color": "#14532d"},
                        {"range": [50, 100], "color": "#713f12"},
                        {"range": [100, 150], "color": "#7c2d12"},
                        {"range": [150, 200], "color": "#7f1d1d"},
                        {"range": [200, 300], "color": "#581c87"},
                    ],
                },
            )
        )
        gauge.update_layout(**PLOT_LAYOUT, height=370)
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"### {aqi_status(latest)}")

    with right:
        chart_df = df.copy()
        fig = px.line(chart_df, x="date", y="predicted_aqi", markers=True, title="3-Day AQI Forecast")
        fig.update_traces(
            line=dict(width=4, color="#22d3ee"),
            marker=dict(size=10, color="#818cf8"),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.08)",
        )
        if live_aqi is not None:
            fig.add_hline(
                y=live_aqi,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Live {live_aqi:.0f}",
                annotation_position="top left",
            )
        fig.update_layout(**PLOT_LAYOUT, height=370, yaxis_title="AQI")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📅 3-Day Forecast")
    cards = st.columns(min(3, len(df)))
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        with cards[i]:
            value = float(row["predicted_aqi"])
            glow = aqi_color(value)
            date_label = row["date"].strftime("%a, %d %b %Y")
            model_used = row.get("model_used", model_name(manifest, f"day{i + 1}"))
            st.markdown(
                f'<div class="forecast-card" style="border-top:4px solid {glow}">'
                f'<div class="small-muted">{date_label}</div>'
                f'<div class="forecast-aqi" style="color:{glow}">{value:.1f}</div>'
                f'<b>{aqi_status(value)}</b><br>'
                f'<span class="chip">🤖 {model_used}</span><hr>'
                f'🌡 Temperature: {fmt_number(row.get("temperature"))} °C<br>'
                f'💧 Humidity: {fmt_number(row.get("humidity"), 0)}%<br>'
                f'🌬 Wind: {fmt_number(row.get("wind_speed"), 1)} m/s<br>'
                f'🌫 PM2.5: {fmt_number(row.get("pm25"), 1)}<br>'
                f'🌫 PM10: {fmt_number(row.get("pm10"), 1)}<br>'
                f'☁ Clouds: {fmt_number(row.get("clouds"), 0)}%<br>'
                f'🌧 Rain: {fmt_number(row.get("rain"), 2)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 📌 Summary Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Average AQI", f"{df['predicted_aqi'].mean():.1f}")
    s2.metric("Maximum AQI", f"{df['predicted_aqi'].max():.1f}")
    s3.metric("Minimum AQI", f"{df['predicted_aqi'].min():.1f}")
    backend_avg = safe_float(payload.get("average_aqi"), df["predicted_aqi"].mean())
    s4.metric("Backend Average AQI", f"{backend_avg:.1f}")

    st.markdown("### ❤️ Health Advisory")
    if latest <= 50:
        st.success("Excellent air quality. Outdoor activities are encouraged.")
    elif latest <= 100:
        st.info("Air quality is acceptable. Sensitive individuals should monitor conditions.")
    elif latest <= 150:
        st.warning("Sensitive groups should reduce prolonged outdoor exposure and monitor symptoms.")
    elif latest <= 200:
        st.error("Unhealthy air. Consider reducing unnecessary outdoor activity and using protection outdoors.")
    else:
        st.error("Very unhealthy/hazardous air. Avoid unnecessary outdoor exposure and follow local health guidance.")

    st.markdown("### 🗺 Forecast Location")
    map_df = pd.DataFrame({"lat": [SADDAR_LAT], "lon": [SADDAR_LON]})
    st.map(map_df, zoom=13)


# ============================================================
# FORECAST & TRENDS TAB
# ============================================================
with tab_forecast:
    st.markdown("### 📈 Forecast Trends")
    st.caption("All charts use the values returned by the FastAPI /predict endpoint.")

    def trend_chart(column, title, color, y_title=None):
        if column not in df.columns:
            st.info(f"{title}: this field was not returned by the backend.")
            return
        local = df[["date", column]].copy()
        local[column] = pd.to_numeric(local[column], errors="coerce")
        local = local.dropna()
        if local.empty:
            st.info(f"{title}: no numeric values are available.")
            return
        fig = px.line(local, x="date", y=column, markers=True, title=title)
        fig.update_traces(line=dict(color=color, width=4), marker=dict(size=9))
        fig.update_layout(**PLOT_LAYOUT, yaxis_title=y_title or title)
        st.plotly_chart(fig, use_container_width=True)

    trend_chart("predicted_aqi", "AQI", "#22d3ee", "AQI")
    c1, c2 = st.columns(2)
    with c1:
        trend_chart("temperature", "Temperature", "#fb923c", "°C")
    with c2:
        trend_chart("humidity", "Humidity", "#38bdf8", "%")
    c3, c4 = st.columns(2)
    with c3:
        trend_chart("pm25", "PM2.5", "#f87171", "PM2.5")
    with c4:
        trend_chart("pm10", "PM10", "#facc15", "PM10")
    c5, c6 = st.columns(2)
    with c5:
        trend_chart("wind_speed", "Wind Speed", "#a3e635", "m/s")
    with c6:
        trend_chart("pressure", "Pressure", "#c084fc", "hPa")

    st.markdown("### 🧪 Pollutant Comparison")
    pollutant_columns = [
        ("pm25", "PM2.5"),
        ("pm10", "PM10"),
        ("ozone", "Ozone"),
        ("carbon_monoxide", "CO"),
        ("nitrogen_dioxide", "NO₂"),
        ("sulphur_dioxide", "SO₂"),
    ]
    available = [item for item in pollutant_columns if item[0] in df.columns]
    if available:
        latest_row = df.iloc[0]
        bar_data = pd.DataFrame(
            {"Pollutant": [label for key, label in available], "Value": [safe_float(latest_row.get(key), 0) for key, _ in available]}
        )
        bar = px.bar(bar_data, x="Pollutant", y="Value", title="Latest Forecast Pollutant Levels")
        bar.update_layout(**PLOT_LAYOUT, yaxis_title="Forecast concentration / value")
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No pollutant fields were returned by the backend.")

    st.markdown("### 📡 API Performance")
    p1, p2 = st.columns(2)
    p1.metric("/predict response time", f"{response_ms:.0f} ms")
    p2.metric("HTTP status", str(status_code), "Healthy" if status_code == 200 else "Check backend")


# ============================================================
# EXPLAINABILITY TAB
# ============================================================
with tab_explain:
    st.markdown("### 🧠 Model Explainability")
    st.caption(
        "SHAP shows which input features contribute most strongly to the AQI prediction. "
        "The dashboard reads the active champion model from model_manifest.json when the required artifacts are available."
    )

    # If the backend exposes an /explain endpoint, use it first. This is optional and
    # does not break the dashboard when the endpoint does not exist.
    explain_payload = None
    explain_error = None
    try:
        started = time.perf_counter()
        explain_response = requests.get(f"{backend_url()}/explain", timeout=12)
        explain_ms = (time.perf_counter() - started) * 1000
        if explain_response.status_code == 200:
            explain_payload = explain_response.json()
        elif explain_response.status_code not in (404, 405):
            explain_error = f"/explain returned HTTP {explain_response.status_code}."
    except requests.RequestException:
        explain_ms = None

    if explain_payload:
        st.success("Explainability data received from FastAPI /explain endpoint.")
        importance_data = explain_payload.get("feature_importance") or explain_payload.get("shap_importance")
        if isinstance(importance_data, dict):
            imp = pd.DataFrame({"Feature": list(importance_data.keys()), "Importance": list(importance_data.values())})
        elif isinstance(importance_data, list) and importance_data and isinstance(importance_data[0], dict):
            imp = pd.DataFrame(importance_data)
            rename_map = {"feature": "Feature", "name": "Feature", "value": "Importance", "importance": "Importance"}
            imp = imp.rename(columns=rename_map)
        else:
            imp = pd.DataFrame()
        if not imp.empty and {"Feature", "Importance"}.issubset(imp.columns):
            imp["Importance"] = pd.to_numeric(imp["Importance"], errors="coerce")
            imp = imp.dropna().sort_values("Importance", ascending=False).head(12)
            fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top SHAP Features")
            fig.update_layout(**PLOT_LAYOUT, height=480)
            st.plotly_chart(fig, use_container_width=True)
            if explain_ms is not None:
                st.caption(f"Explainability API response: {explain_ms:.0f} ms")
        else:
            st.warning("/explain responded successfully, but no recognizable feature-importance payload was returned.")

    # Local fallback using the established SHAP artifacts.
    if explain_payload is None:
        st.info("FastAPI /explain is not required. Trying local SHAP artifacts next…")

        horizon = st.selectbox("Explain forecast horizon", ["day1", "day2", "day3"], index=0)
        info = manifest.get(horizon, {}) if manifest else {}
        importance, shap_error = shap_importance_from_local_model(horizon, info)

        if importance is not None and not importance.empty:
            top = importance.head(12).sort_values(ascending=True)
            imp_df = top.rename("Mean |SHAP value|").reset_index()
            imp_df.columns = ["Feature", "Mean |SHAP value|"]
            fig = px.bar(imp_df, x="Mean |SHAP value|", y="Feature", orientation="h", title=f"Top SHAP Features — {horizon}")
            fig.update_layout(**PLOT_LAYOUT, height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🔎 Interpretation")
            top_feature = importance.index[0]
            top_value = float(importance.iloc[0])
            st.success(f"The strongest feature by mean absolute SHAP value is **{top_feature}** ({top_value:.4f}).")
            st.caption("A larger mean |SHAP| value means the feature has a larger average influence on the model output; it does not by itself indicate whether the feature raises or lowers AQI.")
        else:
            st.warning("SHAP explanation is not available yet for this horizon.")
            st.code(shap_error or "No explainability artifacts found.")
            st.markdown(
                "**Expected local artifacts:** `model_manifest.json`, the champion model artifact(s), "
                "`training_features.csv`, and `model_features.pkl` (when used by the training pipeline)."
            )

    st.markdown("### 📊 Native Model Feature Importance")
    if manifest:
        importance_horizon = st.selectbox(
            "Feature-importance horizon", ["day1", "day2", "day3"], key="native_importance_horizon"
        )
        native_importance = model_feature_importance(
            importance_horizon, manifest.get(importance_horizon, {})
        )
        if native_importance is not None and not native_importance.empty:
            top_native = native_importance.head(12).sort_values(ascending=True)
            native_df = top_native.rename("Importance").reset_index()
            native_df.columns = ["Feature", "Importance"]
            fig = px.bar(
                native_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top Native Feature Importance — {importance_horizon}",
            )
            fig.update_layout(**PLOT_LAYOUT, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "The selected champion does not expose native feature_importances_, "
                "or its model artifact path is not available. SHAP remains the preferred explanation when available."
            )


# ============================================================
# ALERTS & DATA TAB
# ============================================================
with tab_alerts:
    st.markdown("### 🚨 Alert History")
    alert_log = load_alert_log()
    if alert_log.empty:
        st.info("No hazard alerts have been triggered yet.")
    else:
        display_alerts = alert_log.sort_values("logged_at", ascending=False).reset_index(drop=True)
        st.dataframe(display_alerts, use_container_width=True, hide_index=True)
        a1, a2 = st.columns(2)
        a1.metric("Total Alerts Logged", len(display_alerts))
        if a2.button("🗑 Clear Alert Log", use_container_width=True):
            try:
                ALERT_LOG_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            st.rerun()

    st.markdown("### 📋 Forecast Data")
    table_df = df.copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv_bytes,
        file_name="karachi_aqi_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# MODEL & SYSTEM TAB
# ============================================================
with tab_models:
    st.markdown("### 🤖 Champion Model Information")
    if manifest:
        model_rows = []
        for day in ("day1", "day2", "day3"):
            model_rows.append(
                {
                    "Horizon": day.replace("day", "Day "),
                    "Champion Model": model_name(manifest, day),
                    "MAE": model_metric(manifest, day, "mae"),
                    "RMSE": model_metric(manifest, day, "rmse"),
                    "R²": model_metric(manifest, day, "r2"),
                }
            )
        model_df = pd.DataFrame(model_rows)
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        metric_cols = st.columns(3)
        for idx, metric in enumerate(["mae", "rmse", "r2"]):
            values = [model_metric(manifest, day, metric) for day in ("day1", "day2", "day3")]
            if any(v is not None for v in values):
                chart_data = pd.DataFrame({"Horizon": ["Day 1", "Day 2", "Day 3"], metric.upper(): values})
                fig = px.bar(chart_data, x="Horizon", y=metric.upper(), title=metric.upper())
                fig.update_layout(**PLOT_LAYOUT, height=300)
                metric_cols[idx].plotly_chart(fig, use_container_width=True)
    else:
        st.warning("model_manifest.json was not found beside DASHBOARD.py.")

    st.markdown("### 🧩 Model Pipeline")
    st.markdown(
        """
- **Target:** AQI prediction
- **Forecast horizon:** 3 days
- **Champion selection:** one independently trained champion model per forecast day
- **Candidate algorithms:** Random Forest, Ridge Regression, and PyTorch
- **Core inputs:** temperature, humidity, pressure, wind speed, rain, PM2.5, PM10, ozone, CO, NO₂, SO₂
- **Historical features:** lag features, rolling AQI averages, and AQI change-rate features when produced by the training pipeline
- **Backend:** FastAPI
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Weather / air-pollution source:** OpenWeather API
- **Explainability:** SHAP
"""
    )

    st.markdown("### ⚙️ Feature Inventory")
    feature_df = pd.DataFrame({"Feature": FEATURES})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown("### 📦 Runtime Information")
    r1, r2, r3 = st.columns(3)
    r1.metric("Backend URL", backend_url())
    r2.metric("API response", f"{response_ms:.0f} ms")
    r3.metric("Dashboard version", VERSION)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '<span class="chip">⚡ FastAPI</span> '
    '<span class="chip">🎈 Streamlit</span> '
    '<span class="chip">🌲 Scikit-Learn</span> '
    '<span class="chip">🔥 PyTorch</span> '
    '<span class="chip">🧠 SHAP</span> '
    '<span class="chip">☁️ OpenWeather API</span> '
    '<span class="chip">📊 Plotly</span><br><br>'
    'Karachi AQI Predictor • Saddar, Karachi • Version 1.0'
    '</div>',
    unsafe_allow_html=True,
)
