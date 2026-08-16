"""
Karachi AQI Predictor — Professional 4-Page Dashboard
Pages: Overview | Model Comparison | Historical Analysis | Explainability & Custom Prediction
Theme: Dark glassmorphism (preserved from original)
All charts: Interactive Plotly
"""

import os
import json
import warnings
import requests
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# THEME — preserved exactly from original + tab styles
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html,body,[class*="css"]{
    background:#07111F;color:white;
    font-family:'Inter','Poppins',sans-serif;
}
.stApp{
    background:linear-gradient(-45deg,#07111F,#0b1a2e,#091422,#0d2338);
    background-size:400% 400%;
    animation:gradientShift 18s ease infinite;
}
@keyframes gradientShift{
    0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}
}
.main{background:transparent;}
.block-container{padding-top:1.5rem;padding-bottom:2rem;}
h1,h2,h3,h4{color:white;font-family:'Poppins',sans-serif;}

.hero-title{
    font-size:2.4rem;font-weight:800;
    background:linear-gradient(90deg,#22d3ee,#38bdf8,#818cf8,#22d3ee);
    background-size:300% 300%;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:shine 6s ease infinite;margin-bottom:0;
}
@keyframes shine{
    0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}
}

.live-badge{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.4);
    color:#4ade80;padding:4px 14px;border-radius:999px;
    font-size:.8rem;font-weight:600;margin-left:14px;vertical-align:middle;
}
.pulse-dot{
    width:9px;height:9px;border-radius:50%;background:#22c55e;
    box-shadow:0 0 0 0 rgba(34,197,94,.7);animation:pulse 1.6s infinite;
}
@keyframes pulse{
    0%{box-shadow:0 0 0 0 rgba(34,197,94,.6);}
    70%{box-shadow:0 0 0 10px rgba(34,197,94,0);}
    100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}
}

[data-testid="stMetric"]{
    background:rgba(30,41,59,.55);backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.08);border-radius:16px;
    padding:16px 18px 10px 18px;transition:.3s ease;
    box-shadow:0 4px 18px rgba(0,0,0,.25);
}
[data-testid="stMetric"]:hover{
    transform:translateY(-4px);border-color:rgba(56,189,248,.4);
    box-shadow:0 8px 26px rgba(56,189,248,.15);
}

.card{
    background:linear-gradient(160deg,#182840,#101c2e);
    border-radius:18px;padding:20px;border:1px solid #2B3A4D;
    transition:.35s ease;animation:fadeInUp .6s ease;
}
.card:hover{
    transform:translateY(-5px) scale(1.01);
    border-color:rgba(56,189,248,.5);
    box-shadow:0 10px 30px rgba(56,189,248,.15);
}
@keyframes fadeInUp{
    from{opacity:0;transform:translateY(14px);}
    to{opacity:1;transform:translateY(0);}
}

.metric-card{
    background:rgba(30,41,59,.65);backdrop-filter:blur(18px);
    border-radius:18px;padding:20px;
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 0 25px rgba(0,255,255,.08);transition:.3s;
}
.metric-card:hover{transform:translateY(-6px);box-shadow:0 0 35px rgba(0,255,255,.20);}

.legend-wrap{
    display:flex;border-radius:12px;overflow:hidden;
    height:14px;margin-top:10px;box-shadow:0 0 15px rgba(0,0,0,.4);
}
.legend-seg{flex:1;}
.legend-labels{
    display:flex;justify-content:space-between;
    font-size:.72rem;color:#9CA3AF;margin-top:6px;
}

.pollutant-row{margin-bottom:14px;}
.pollutant-name{
    display:flex;justify-content:space-between;
    font-size:.85rem;color:#cbd5e1;margin-bottom:4px;
}
.bar-track{background:rgba(255,255,255,.08);border-radius:8px;height:10px;overflow:hidden;}
.bar-fill{height:100%;border-radius:8px;transition:width 1s ease;}

.model-badge{
    display:inline-block;background:rgba(129,140,248,.15);
    border:1px solid rgba(129,140,248,.4);color:#c7d2fe;
    padding:2px 10px;border-radius:999px;font-size:.72rem;margin-top:6px;
}
.model-compare-card{
    background:linear-gradient(160deg,#0f2035,#091525);
    border-radius:16px;padding:18px;border:1px solid #1e3a5f;
    text-align:center;transition:.3s;
}
.model-compare-card:hover{
    border-color:rgba(56,189,248,.5);
    box-shadow:0 8px 24px rgba(56,189,248,.12);
}

hr{border:none;border-top:1px solid #2B3A4D;}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1524,#070f1c);
    border-right:1px solid rgba(255,255,255,.06);
}

.stTabs [data-baseweb="tab-list"]{
    background:rgba(255,255,255,.04);border-radius:14px;
    padding:4px;border:1px solid rgba(255,255,255,.07);gap:4px;
}
.stTabs [data-baseweb="tab"]{
    color:#94a3b8;font-weight:600;border-radius:10px;
    padding:8px 20px;font-size:.88rem;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,rgba(34,211,238,.18),rgba(129,140,248,.18)) !important;
    color:#e2e8f0 !important;border:1px solid rgba(56,189,248,.3) !important;
}

.alert-banner{
    background:linear-gradient(90deg,rgba(239,68,68,.18),rgba(239,68,68,.08));
    border:1px solid rgba(239,68,68,.55);border-left:5px solid #ef4444;
    border-radius:14px;padding:16px 20px;margin:16px 0;
    font-size:1.02rem;color:#fecaca;
    animation:alertPulse 2s ease-in-out infinite;
}
@keyframes alertPulse{
    0%{box-shadow:0 0 0 0 rgba(239,68,68,.35);}
    70%{box-shadow:0 0 0 14px rgba(239,68,68,0);}
    100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}
}

.monitor-badge{
    display:inline-flex;align-items:center;gap:10px;
    background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.3);
    border-radius:12px;padding:10px 16px;font-size:.85rem;color:#93c5fd;margin:12px 0;
}
.monitor-dot{
    width:8px;height:8px;border-radius:50%;background:#38bdf8;
    box-shadow:0 0 0 0 rgba(56,189,248,.6);animation:pulse 1.6s infinite;
}

.section-header{
    font-family:'Poppins',sans-serif;font-size:1.15rem;
    font-weight:700;color:#e2e8f0;letter-spacing:.02em;
    padding:10px 0 6px;border-bottom:1px solid rgba(56,189,248,.2);
    margin-bottom:14px;
}

.footer-chip{
    display:inline-block;background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.1);padding:4px 12px;
    border-radius:999px;font-size:.78rem;margin:3px;color:#93c5fd;
}

.stDownloadButton button{
    background:linear-gradient(90deg,#0ea5e9,#22d3ee);
    color:#04121b;font-weight:700;border:none;border-radius:10px;transition:.3s;
}
.stDownloadButton button:hover{
    transform:translateY(-2px);box-shadow:0 6px 18px rgba(34,211,238,.35);
}

::-webkit-scrollbar{width:10px;}
::-webkit-scrollbar-track{background:#0a1626;}
::-webkit-scrollbar-thumb{background:#22405e;border-radius:6px;}
::-webkit-scrollbar-thumb:hover{background:#2f5a82;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════
API_KEY      = os.getenv("OPENWEATHER_API_KEY")
LAT, LON     = 24.7967, 67.0728
LOCATION     = "Defence Phase 7, Karachi"
ALERT_LOG    = "alert_log.csv"

FEATURE_COLS = [
    "temperature","humidity","pressure","wind_speed","rain",
    "pm2_5","pm10","ozone","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide",
    "year","month","day","day_of_week","day_of_year","weekend",
    "AQI_lag_1","AQI_lag_2","AQI_lag_3","AQI_3day_mean","AQI_7day_mean",
    "rain_flag","temp_humidity",
]

POLLUTANT_LIMITS = {"pm2_5":60,"pm10":100,"ozone":100}

KARACHI_AREAS = {
    "Defence Phase 7 (Default)": (24.7967, 67.0728),
    "Clifton":                   (24.8067, 67.0300),
    "Saddar":                    (24.8559, 67.0106),
    "Gulshan-e-Iqbal":           (24.9215, 67.1024),
    "Malir":                     (24.8928, 67.2009),
    "Korangi":                   (24.8186, 67.1308),
    "North Nazimabad":           (24.9356, 67.0435),
    "Orangi Town":               (24.9480, 66.9950),
    "Lyari":                     (24.8450, 67.0050),
    "Baldia Town":               (24.9000, 66.9700),
}

MODEL_COLORS = {
    "PyTorch LSTM":     "#22d3ee",
    "Random Forest":    "#818cf8",
    "Ridge Regression": "#4ade80",
}

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#cbd5e1", size=12),
)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def aqi_color(v):
    if v <= 50:  return "#22c55e"
    if v <= 100: return "#eab308"
    if v <= 150: return "#f97316"
    if v <= 200: return "#ef4444"
    if v <= 300: return "#a855f7"
    return "#4b5563"

def aqi_label(v):
    if v <= 50:  return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive Groups"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

def aqi_alert(v):
    if v <= 50:  return "🟢 Good"
    if v <= 100: return "🟡 Moderate"
    if v <= 150: return "🟠 Unhealthy for Sensitive"
    if v <= 200: return "🔴 Unhealthy"
    if v <= 300: return "🟣 Very Unhealthy"
    return "⚫ Hazardous"

def pm25_to_aqi(c):
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for cl,ch,il,ih in bp:
        if cl <= c <= ch:
            return round(((ih-il)/(ch-cl))*(c-cl)+il, 1)
    return 500.0

def log_alert(forecast_date, predicted_aqi, threshold):
    entry = pd.DataFrame([{
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_date": forecast_date,
        "predicted_aqi": round(float(predicted_aqi),1),
        "threshold": threshold,
        "status": aqi_alert(predicted_aqi)
    }])
    if os.path.exists(ALERT_LOG):
        existing = pd.read_csv(ALERT_LOG)
        if ((existing["forecast_date"]==forecast_date) &
            (existing["predicted_aqi"]==entry.iloc[0]["predicted_aqi"])).any():
            return
        entry = pd.concat([existing,entry], ignore_index=True)
    entry.to_csv(ALERT_LOG, index=False)

def load_alert_log():
    if os.path.exists(ALERT_LOG):
        return pd.read_csv(ALERT_LOG)
    return pd.DataFrame(columns=["logged_at","forecast_date","predicted_aqi","threshold","status"])


# ══════════════════════════════════════════════════════════════
# PYTORCH LSTM MODEL DEFINITION
# ══════════════════════════════════════════════════════════════
class AQI_LSTM(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.lstm1 = nn.LSTM(inp, 128, batch_first=True, bidirectional=True)
        self.d1    = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=False)
        self.d2    = nn.Dropout(0.2)
        self.fc1   = nn.Linear(64, 32)
        self.bn    = nn.BatchNorm1d(32)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.lstm1(x);  o = self.d1(o)
        o, _ = self.lstm2(o);  o = self.d2(o[:,-1,:])
        return self.fc2(torch.relu(self.bn(self.fc1(o))))


# ══════════════════════════════════════════════════════════════
# DATA & MODEL LOADERS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_features():
    df = pd.read_csv("data/processed/final_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_registry():
    return pd.read_csv("data/registry/model_registry.csv")

@st.cache_resource
def load_all_models():
    models = {}
    # Random Forest
    for f in ["random_forest_v8.pkl","rf_model.pkl"]:
        p = f"data/models/{f}"
        if os.path.exists(p):
            models["Random Forest"] = joblib.load(p)
            break
    # Ridge
    if os.path.exists("data/models/ridge_model.pkl"):
        models["Ridge Regression"] = joblib.load("data/models/ridge_model.pkl")
    # PyTorch LSTM
    pt_path  = "data/models/pytorch_lstm.pt"
    sc_path  = "data/models/pytorch_scaler.pkl"
    if os.path.exists(pt_path) and os.path.exists(sc_path):
        ckpt   = torch.load(pt_path, map_location="cpu", weights_only=False)
        lstm   = AQI_LSTM(ckpt["input_size"])
        lstm.load_state_dict(ckpt["model_state"])
        lstm.eval()
        scaler = joblib.load(sc_path)
        models["PyTorch LSTM"] = {"type":"pytorch","model":lstm,"scaler":scaler}
    return models

@st.cache_data
def load_forecast():
    p = "data/processed/aqi_forecast.csv"
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_csv(p); df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= pd.Timestamp.now().normalize()].reset_index(drop=True)

def predict_single(model_obj, X: pd.DataFrame) -> float:
    if isinstance(model_obj, dict) and model_obj.get("type") == "pytorch":
        sc  = model_obj["scaler"]
        m   = model_obj["model"]
        Xs  = sc.transform(X).astype(np.float32)
        t   = torch.tensor(Xs).reshape(-1,1,Xs.shape[1])
        with torch.no_grad():
            return float(m(t).cpu().numpy().flatten()[0])
    if isinstance(model_obj, dict) and "model" in model_obj:
        sc = model_obj.get("scaler")
        return float(model_obj["model"].predict(sc.transform(X) if sc else X)[0])
    return float(model_obj.predict(X)[0])

def get_feature_importance(model_obj, feat_names):
    if isinstance(model_obj, dict) and model_obj.get("type") == "pytorch":
        return None
    obj = model_obj.get("model", model_obj) if isinstance(model_obj, dict) else model_obj
    if hasattr(obj, "feature_importances_"):
        return obj.feature_importances_
    if hasattr(obj, "coef_"):
        return np.abs(obj.coef_)
    return None

@st.cache_data(ttl=300)
def fetch_live_aqi(lat=LAT, lon=LON):
    if not API_KEY: return None
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat":lat,"lon":lon,"appid":API_KEY}, timeout=8)
        if r.status_code != 200: return None
        c = r.json()["list"][0]["components"]
        return {"pm2_5":c["pm2_5"],"pm10":c["pm10"],"ozone":c["o3"],
                "carbon_monoxide":c["co"],"nitrogen_dioxide":c["no2"],"sulphur_dioxide":c["so2"],
                "aqi": pm25_to_aqi(c["pm2_5"])}
    except: return None

def plotly_layout(h=320, **kwargs):
    d = dict(**PLOTLY_BASE, height=h, margin=dict(t=40,b=30,l=50,r=20))
    d.update(kwargs)
    return d


# ══════════════════════════════════════════════════════════════
# LOAD ALL DATA ONCE
# ══════════════════════════════════════════════════════════════
df_hist     = load_features()
registry    = load_registry()
models      = load_all_models()
forecast_df = load_forecast()
live_poll   = fetch_live_aqi()
FCOLS       = [c for c in df_hist.columns if c not in ("date","AQI")]

prod_row = registry[registry["status"]=="Production"].iloc[-1] \
           if "status" in registry.columns and len(registry[registry["status"]=="Production"]) > 0 \
           else registry.iloc[-1]

live_aqi = live_poll["aqi"] if live_poll else float(df_hist["AQI"].iloc[-1])
today_pred = float(forecast_df.iloc[0]["Predicted_AQI"]) if len(forecast_df) > 0 else live_aqi


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:12px 0 6px;'>"
        "<span style='font-size:32px;'>🌍</span>"
        "<p style='font-family:Poppins,sans-serif;font-weight:800;font-size:15px;"
        "color:#fff;margin:6px 0 2px;'>Pearls AQI Predictor</p>"
        f"<p style='font-size:9px;color:#64748b;letter-spacing:.12em;margin:0;'>📍 {LOCATION.upper()}</p>"
        "</div>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Model Comparison",
        "📈 Historical Analysis",
        "🔬 Explainability & Custom Prediction",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='background:rgba(30,41,59,.6);border-radius:14px;padding:14px;
    border:1px solid rgba(255,255,255,.08);'>
        <p style='font-size:9px;font-weight:700;letter-spacing:.12em;
        text-transform:uppercase;color:#475569;margin:0 0 8px;'>⚙️ Production Model</p>
        <p style='color:#e2e8f0;font-weight:700;margin:0;font-size:13px;'>
        {prod_row['algorithm']} <span style='color:#64748b;'>v{prod_row['version']}</span></p>
        <div style='display:flex;gap:18px;margin-top:8px;'>
            <div><p style='font-size:8px;color:#475569;margin:0;text-transform:uppercase;'>R²</p>
                 <p style='font-size:13px;color:#22d3ee;font-weight:700;margin:0;'>{prod_row['r2']:.3f}</p></div>
            <div><p style='font-size:8px;color:#475569;margin:0;text-transform:uppercase;'>MAE</p>
                 <p style='font-size:13px;color:#22d3ee;font-weight:700;margin:0;'>{prod_row['mae']:.2f}</p></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    alert_threshold = st.slider("🚨 Alert Threshold (AQI)",
        min_value=50, max_value=300, value=150, step=10)
    enable_toast = st.checkbox("🔔 Toast on alert", value=True)
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption(f"Updated {datetime.now().strftime('%d %b %Y %H:%M')}")


# ══════════════════════════════════════════════════════════════
# HERO HEADER (all pages)
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<span class="hero-title">🌍 Karachi Air Quality Prediction</span>'
    '<span class="live-badge"><span class="pulse-dot"></span> LIVE</span>',
    unsafe_allow_html=True)
st.caption("AI-powered 3-Day AQI Forecast · PyTorch LSTM · Random Forest · Ridge Regression")

st.markdown(f"""
<div class="monitor-badge">
    <span class="monitor-dot"></span>
    🛡️ Hazard monitoring active — alert threshold <b>{alert_threshold} AQI</b>
</div>""", unsafe_allow_html=True)

# AQI scale legend
st.markdown("""
<div class="legend-wrap">
    <div class="legend-seg" style="background:#22c55e;"></div>
    <div class="legend-seg" style="background:#eab308;"></div>
    <div class="legend-seg" style="background:#f97316;"></div>
    <div class="legend-seg" style="background:#ef4444;"></div>
    <div class="legend-seg" style="background:#a855f7;"></div>
    <div class="legend-seg" style="background:#4b5563;"></div>
</div>
<div class="legend-labels">
    <span>0 Good</span><span>50 Moderate</span><span>100 USG</span>
    <span>150 Unhealthy</span><span>200 Very Unhealthy</span><span>300+ Hazardous</span>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# Alert banner
if len(forecast_df) > 0:
    exceeded = forecast_df[forecast_df["Predicted_AQI"] >= alert_threshold]
    if not exceeded.empty:
        worst = exceeded.loc[exceeded["Predicted_AQI"].idxmax()]
        st.markdown(f"""<div class="alert-banner">
        🚨 <b>HAZARD ALERT</b> — {worst['date'].strftime('%d %b')} forecast AQI
        <b>{worst['Predicted_AQI']:.0f}</b> ≥ threshold {alert_threshold}.
        Status: {aqi_alert(worst['Predicted_AQI'])}</div>""", unsafe_allow_html=True)
        if enable_toast:
            st.toast(f"Hazardous AQI: {worst['Predicted_AQI']:.0f}", icon="🚨")
        for _, row in exceeded.iterrows():
            log_alert(row["date"].strftime("%Y-%m-%d"), row["Predicted_AQI"], alert_threshold)

st.markdown("---")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — OVERVIEW                                          ║
# ╚══════════════════════════════════════════════════════════════╝
if page == "🏠 Overview":

    # ── KPI row ──────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("📍 Location",   "Defence Ph 7", "Karachi")
    k2.metric("🌡 Live AQI",   f"{live_aqi:.0f}", aqi_alert(live_aqi))
    k3.metric("🔮 Today's Pred", f"{today_pred:.0f}",
              f"{'↑' if today_pred>live_aqi else '↓'}{abs(today_pred-live_aqi):.1f}")
    k4.metric("🤖 Production", prod_row["algorithm"], f"R²={prod_row['r2']:.3f}")
    k5.metric("🕒 Updated",    datetime.now().strftime("%H:%M"),
              datetime.now().strftime("%d %b"))

    st.markdown("---")

    # ── Live AQI gauge + 3-day forecast line ─────────────────
    lc = aqi_color(live_aqi)
    g1, g2 = st.columns([1,2], gap="large")

    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=live_aqi,
            delta={"reference": today_pred,
                   "increasing":{"color":"#ef4444"},
                   "decreasing":{"color":"#22c55e"}},
            number={"font":{"size":36,"color":lc,"family":"Poppins"}},
            title={"text":"Live AQI Now","font":{"size":13,"color":"#94a3b8"}},
            gauge={
                "axis":{"range":[0,300],"tickcolor":"#334155"},
                "bar":{"color":lc,"thickness":0.22},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,50],  "color":"rgba(34,197,94,.10)"},
                    {"range":[50,100], "color":"rgba(234,179,8,.10)"},
                    {"range":[100,150],"color":"rgba(249,115,22,.10)"},
                    {"range":[150,200],"color":"rgba(239,68,68,.10)"},
                    {"range":[200,300],"color":"rgba(168,85,247,.10)"},
                ],
                "threshold":{"line":{"color":"#fff","width":2},"value":today_pred}
            }))
        gauge.update_layout(**plotly_layout(280, margin=dict(t=30,b=10,l=20,r=20)))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption("White marker = today's model prediction")

    with g2:
        if len(forecast_df) > 0:
            fc = forecast_df.copy()
            mc = [aqi_color(v) for v in fc["Predicted_AQI"]]
            fig_fc = go.Figure()
            for lo,hi,col,lbl in [(0,50,"rgba(34,197,94,.06)","Good"),
                                   (50,100,"rgba(234,179,8,.06)","Moderate"),
                                   (100,150,"rgba(249,115,22,.06)","Sensitive"),
                                   (150,200,"rgba(239,68,68,.06)","Unhealthy"),
                                   (200,300,"rgba(168,85,247,.06)","Very Unhealthy")]:
                fig_fc.add_hrect(y0=lo,y1=hi,fillcolor=col,line_width=0,
                    annotation_text=lbl,annotation_position="right",
                    annotation_font_size=9,annotation_font_color="rgba(148,163,184,.5)")
            fig_fc.add_trace(go.Scatter(
                x=fc["date"], y=fc["Predicted_AQI"],
                mode="lines+markers+text",
                line=dict(color="#22d3ee",width=3),
                marker=dict(size=14,color=mc,line=dict(color="#fff",width=2)),
                text=[f"{v:.0f}" for v in fc["Predicted_AQI"]],
                textposition="top center",
                textfont=dict(size=12,color="#e2e8f0"),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
                hovertemplate="<b>%{x|%A %d %b}</b><br>Predicted AQI: %{y:.1f}<extra></extra>",
                name="Predicted AQI"))
            fig_fc.update_layout(
                **plotly_layout(280),
                title="3-Day AQI Forecast",
                xaxis=dict(showgrid=False,title="",tickformat="%a %d %b"),
                yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="AQI"),
                showlegend=False)
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Run the forecast pipeline to generate predictions.")

    # ── 3-day forecast cards ──────────────────────────────────
    st.markdown('<p class="section-header">📅 3-Day Forecast Cards</p>', unsafe_allow_html=True)
    next3 = forecast_df.iloc[1:4] if len(forecast_df) > 3 else forecast_df.iloc[:3]
    if len(next3):
        cols = st.columns(len(next3))
        for col,(_, row) in zip(cols, next3.iterrows()):
            dc = aqi_color(row["Predicted_AQI"])
            col.markdown(f"""
            <div class="card" style="border-top:3px solid {dc};text-align:center;">
                <p style="font-size:9px;font-weight:700;letter-spacing:.12em;
                text-transform:uppercase;color:#475569;margin:0 0 2px;">
                {row['date'].strftime('%A')}</p>
                <p style="font-size:11px;color:#64748b;margin:0 0 6px;">
                {row['date'].strftime('%d %b %Y')}</p>
                <span style="font-family:Poppins;font-size:42px;font-weight:800;
                color:{dc};line-height:1;">{row['Predicted_AQI']:.0f}</span>
                <p style="color:{dc};font-size:11px;font-weight:600;margin:6px 0 0;">
                {aqi_label(row['Predicted_AQI'])}</p>
                <p style="font-size:9px;color:#475569;margin:3px 0 0;">
                {aqi_alert(row['Predicted_AQI'])}</p>
                <span class="model-badge">🤖 {prod_row['algorithm']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Weather + Pollutants ──────────────────────────────────
    st.markdown('<p class="section-header">🌤️ Current Conditions</p>', unsafe_allow_html=True)
    try:
        lw = pd.read_csv("data/processed/weather_forecast.csv").iloc[0]
    except: lw = df_hist.iloc[-1]

    def gw(k):
        try: return float(lw[k])
        except: return float(df_hist.iloc[-1].get(k, 0))

    w1,w2,w3,w4,w5 = st.columns(5)
    w1.metric("🌡 Temperature", f"{gw('temperature'):.1f} °C")
    w2.metric("💧 Humidity",    f"{gw('humidity'):.0f} %")
    w3.metric("💨 Wind Speed",  f"{gw('wind_speed'):.1f} km/h")
    w4.metric("🌧 Rainfall",    f"{gw('rain'):.2f} mm")
    w5.metric("🔵 Pressure",    f"{gw('pressure'):.1f} hPa")

    if live_poll:
        st.markdown('<p class="section-header">🏭 Pollutant Concentrations</p>',
                    unsafe_allow_html=True)
        p1,p2,p3,p4,p5,p6 = st.columns(6)
        p1.metric("PM2.5", f"{live_poll['pm2_5']:.2f}")
        p2.metric("PM10",  f"{live_poll['pm10']:.2f}")
        p3.metric("O₃",    f"{live_poll['ozone']:.2f}")
        p4.metric("CO",    f"{live_poll['carbon_monoxide']:.2f}")
        p5.metric("NO₂",   f"{live_poll['nitrogen_dioxide']:.2f}")
        p6.metric("SO₂",   f"{live_poll['sulphur_dioxide']:.2f}")

        # Pollutant radar
        rc = ["PM2.5","PM10","Ozone","CO/10","NO₂","SO₂"]
        rv = [live_poll['pm2_5'], live_poll['pm10'], live_poll['ozone'],
              live_poll['carbon_monoxide']/10, live_poll['nitrogen_dioxide'], live_poll['sulphur_dioxide']]
        _, rcol, _ = st.columns([0.2,0.6,0.2])
        with rcol:
            fig_r = go.Figure(go.Scatterpolar(
                r=rv+[rv[0]], theta=rc+[rc[0]], fill="toself",
                fillcolor="rgba(34,211,238,.12)", line=dict(color="#22d3ee",width=2),
                hovertemplate="%{theta}: %{r:.2f} µg/m³<extra></extra>"))
            fig_r.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(tickfont=dict(size=8,color="#64748b"),
                                    gridcolor="rgba(51,65,85,.4)"),
                    angularaxis=dict(tickfont=dict(size=10,color="#94a3b8"),
                                     gridcolor="rgba(51,65,85,.4)")),
                **plotly_layout(300), title="Pollutant Radar",
                showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")

    # ── Forecast table ────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Forecast Table</p>', unsafe_allow_html=True)
    if len(forecast_df) > 0:
        tbl = forecast_df.copy()
        tbl["Date"]          = tbl["date"].dt.strftime("%A, %d %b %Y")
        tbl["Predicted AQI"] = tbl["Predicted_AQI"].round(1)
        tbl["Category"]      = tbl["Predicted_AQI"].apply(aqi_label)
        tbl["Alert"]         = tbl["Predicted_AQI"].apply(aqi_alert)
        try:
            wf = pd.read_csv("data/processed/weather_forecast.csv")
            wf["date"] = pd.to_datetime(wf["date"])
            tbl = tbl.merge(wf[["date","temperature","humidity","wind_speed","rain"]], on="date", how="left")
            tbl = tbl.rename(columns={"temperature":"Temp °C","humidity":"RH %",
                                      "wind_speed":"Wind km/h","rain":"Rain mm"})
            show = ["Date","Predicted AQI","Category","Alert","Temp °C","RH %","Wind km/h","Rain mm"]
        except:
            show = ["Date","Predicted AQI","Category","Alert"]
        st.dataframe(tbl[[c for c in show if c in tbl.columns]],
                     use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Forecast CSV",
            tbl.to_csv(index=False).encode(), "karachi_aqi_forecast.csv", "text/csv")

    # ── Location map ──────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📍 Monitoring Location</p>', unsafe_allow_html=True)
    map_df = pd.DataFrame({"lat":[LAT],"lon":[LON],"name":[LOCATION],"AQI":[live_aqi]})
    fig_map = px.scatter_mapbox(map_df,lat="lat",lon="lon",hover_name="name",
        hover_data={"AQI":True,"lat":False,"lon":False},
        color_discrete_sequence=[aqi_color(live_aqi)],zoom=12,height=340)
    fig_map.update_traces(marker=dict(size=18))
    fig_map.update_layout(mapbox_style="carto-darkmatter",
        mapbox_center={"lat":LAT,"lon":LON},
        paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=0,b=0,l=0,r=0),height=340)
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Alert history ─────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🚨 Alert History</p>', unsafe_allow_html=True)
    alert_log = load_alert_log()
    if alert_log.empty:
        st.info("No hazard alerts triggered yet.")
    else:
        st.dataframe(alert_log.sort_values("logged_at",ascending=False),
                     use_container_width=True, hide_index=True)
        h1,_ = st.columns([1,3])
        h1.metric("Total Alerts", len(alert_log))
        if h1.button("🗑 Clear Log"):
            os.remove(ALERT_LOG); st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""<div style="text-align:center;">
        <span class="footer-chip">⚡ FastAPI</span>
        <span class="footer-chip">🎈 Streamlit</span>
        <span class="footer-chip">🌲 Scikit-Learn</span>
        <span class="footer-chip">🔥 PyTorch</span>
        <span class="footer-chip">☁ OpenWeather</span>
        <span class="footer-chip">📊 Plotly</span>
    </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — MODEL COMPARISON                                  ║
# ╚══════════════════════════════════════════════════════════════╝
elif page == "📊 Model Comparison":

    st.markdown('<p class="section-header">📊 Model Registry</p>', unsafe_allow_html=True)
    st.dataframe(registry, use_container_width=True, hide_index=True,
        column_config={
            "r2":   st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.3f"),
            "mae":  st.column_config.NumberColumn("MAE",  format="%.2f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        })

    # ── Model capability cards ────────────────────────────────
    st.markdown('<p class="section-header">🤖 Model Capabilities</p>', unsafe_allow_html=True)
    mc1,mc2,mc3 = st.columns(3)
    caps = [
        ("PyTorch LSTM",     "#22d3ee", "🔥",
         "Bidirectional LSTM with BatchNorm & Dropout.",
         ["✅ Captures temporal dependencies","✅ Best R² & lowest MAE/RMSE",
          "✅ Production champion","⚡ Requires GPU for large datasets"]),
        ("Random Forest",    "#818cf8", "🌲",
         "Ensemble of 100+ decision trees.",
         ["✅ Robust to outliers & noise","✅ Built-in feature importance",
          "✅ No scaling required","⚡ Higher memory footprint"]),
        ("Ridge Regression", "#4ade80", "📐",
         "L2-regularised linear regression.",
         ["✅ Fast inference","✅ Interpretable coefficients",
          "✅ Stable on small datasets","⚡ Assumes linearity"]),
    ]
    for col,(name,clr,icon,desc,pts) in zip([mc1,mc2,mc3], caps):
        row = registry[registry["algorithm"].str.contains(name.split()[0],case=False,na=False)]
        r2  = row["r2"].values[0]  if len(row) else 0
        mae = row["mae"].values[0] if len(row) else 0
        col.markdown(f"""
        <div class="model-compare-card">
            <p style="font-size:28px;margin:0;">{icon}</p>
            <p style="font-family:Poppins;font-size:13px;font-weight:700;
            color:{clr};margin:4px 0;">{name}</p>
            <p style="font-size:11px;color:#64748b;margin:0 0 10px;">{desc}</p>
            <div style="display:flex;gap:18px;justify-content:center;margin-bottom:10px;">
                <div><p style="font-size:8px;color:#475569;margin:0;">R²</p>
                     <p style="font-size:16px;color:{clr};font-weight:700;margin:0;">{r2:.3f}</p></div>
                <div><p style="font-size:8px;color:#475569;margin:0;">MAE</p>
                     <p style="font-size:16px;color:{clr};font-weight:700;margin:0;">{mae:.2f}</p></div>
            </div>
            {''.join(f'<p style="font-size:11px;color:#94a3b8;margin:2px 0;text-align:left;">{p}</p>' for p in pts)}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric comparison bars ────────────────────────────────
    st.markdown('<p class="section-header">📈 Performance Metrics Comparison</p>',
                unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    clrs_reg = [MODEL_COLORS.get(a, "#38bdf8") for a in registry["algorithm"]]
    for col, met, title in [(m1,"r2","R² Score (higher = better)"),
                             (m2,"mae","MAE (lower = better)"),
                             (m3,"rmse","RMSE (lower = better)")]:
        with col:
            fig = go.Figure(go.Bar(
                x=registry["algorithm"], y=registry[met],
                marker_color=clrs_reg,
                text=registry[met].round(3), textposition="outside",
                hovertemplate="<b>%{x}</b><br>" + met.upper() + ": %{y:.4f}<extra></extra>"))
            fig.update_layout(**plotly_layout(280),
                title=title,
                xaxis=dict(showgrid=False,tickfont=dict(size=10)),
                yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)"),
                showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Radar capability chart ────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🕸️ Model Capability Radar</p>',
                unsafe_allow_html=True)
    radar_cats = ["R² Score","Low MAE","Low RMSE","Speed","Interpretability","Scalability"]
    radar_data = {
        "PyTorch LSTM":     [0.97, 0.95, 0.96, 0.55, 0.40, 0.70],
        "Random Forest":    [0.90, 0.80, 0.82, 0.75, 0.70, 0.65],
        "Ridge Regression": [0.96, 0.92, 0.93, 0.99, 0.99, 0.95],
    }
    fig_radar = go.Figure()
    for name, vals in radar_data.items():
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=radar_cats+[radar_cats[0]],
            fill="toself", name=name,
            line=dict(color=MODEL_COLORS[name], width=2),
            fillcolor=MODEL_COLORS[name].replace("#","rgba(").replace("ee","ee,0.10)")
                      if "#22d3ee" in MODEL_COLORS[name]
                      else MODEL_COLORS[name]+"18",
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0,1],tickfont=dict(size=8),
                            gridcolor="rgba(51,65,85,.4)"),
            angularaxis=dict(tickfont=dict(size=11,color="#94a3b8"),
                             gridcolor="rgba(51,65,85,.4)")),
        **plotly_layout(380),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Test-set predictions vs actual ────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔮 Predicted vs Actual AQI — Test Set</p>',
                unsafe_allow_html=True)
    split = int(len(df_hist)*0.80)
    X_test = df_hist[FCOLS].iloc[split:]
    y_test = df_hist["AQI"].iloc[split:]
    test_dates = df_hist["date"].iloc[split:].reset_index(drop=True)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_test.values, mode="lines", name="Actual AQI",
        line=dict(color="#ffffff",width=1.5,dash="dot"),
        hovertemplate="<b>Actual</b> %{x|%d %b}: %{y:.1f}<extra></extra>"))
    for name, mdl in models.items():
        try:
            preds = [predict_single(mdl, X_test.iloc[[i]]) for i in range(min(300, len(X_test)))]
            fig_pred.add_trace(go.Scatter(
                x=test_dates[:len(preds)], y=preds, mode="lines", name=name,
                line=dict(color=MODEL_COLORS.get(name,"#38bdf8"),width=2), opacity=0.85,
                hovertemplate=f"<b>{name}</b> %{{x|%d %b}}: %{{y:.1f}}<extra></extra>"))
        except: pass
    fig_pred.update_layout(**plotly_layout(380),
        title="Predicted vs Actual AQI on Test Set",
        xaxis=dict(showgrid=False,title=""),
        yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="AQI"),
        legend=dict(bgcolor="rgba(15,32,53,.8)",bordercolor="rgba(56,189,248,.3)",
                    borderwidth=1,font=dict(size=11)),
        hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption("White dotted = actual AQI · Coloured = model predictions on last 20% of data")

    # ── Scatter + residuals ───────────────────────────────────
    st.markdown("---")
    sc1,sc2 = st.columns(2)
    with sc1:
        st.markdown('<p class="section-header">🎯 Actual vs Predicted Scatter</p>',
                    unsafe_allow_html=True)
        fig_sc = go.Figure()
        lo,hi = float(y_test.min()), float(y_test.max())
        fig_sc.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",
            line=dict(color="rgba(255,255,255,.3)",dash="dot",width=1.5),
            name="Perfect",hoverinfo="skip"))
        for name, mdl in models.items():
            try:
                preds = [predict_single(mdl, X_test.iloc[[i]]) for i in range(min(300,len(X_test)))]
                fig_sc.add_trace(go.Scatter(
                    x=y_test.values[:len(preds)], y=preds, mode="markers",
                    marker=dict(color=MODEL_COLORS.get(name,"#38bdf8"),size=5,opacity=0.55),
                    name=name,
                    hovertemplate=f"<b>{name}</b><br>Actual: %{{x:.1f}}<br>Pred: %{{y:.1f}}<extra></extra>"))
            except: pass
        fig_sc.update_layout(**plotly_layout(340),
            xaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Actual AQI"),
            yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Predicted AQI"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
        st.plotly_chart(fig_sc, use_container_width=True)

    with sc2:
        st.markdown('<p class="section-header">📦 Residual Distribution</p>',
                    unsafe_allow_html=True)
        fig_res = go.Figure()
        for name, mdl in models.items():
            try:
                preds = [predict_single(mdl, X_test.iloc[[i]]) for i in range(min(300,len(X_test)))]
                resid = y_test.values[:len(preds)] - np.array(preds)
                fig_res.add_trace(go.Box(
                    y=resid, name=name, marker_color=MODEL_COLORS.get(name,"#38bdf8"),
                    boxmean=True,
                    hovertemplate="Residual: %{y:.2f}<extra></extra>"))
            except: pass
        fig_res.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.3)")
        fig_res.update_layout(**plotly_layout(340),
            yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Actual − Predicted"),
            xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature importance per model ──────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔑 Feature Importance by Model</p>',
                unsafe_allow_html=True)
    fi1, fi2 = st.columns(2)
    for col, (name, mdl) in zip([fi1,fi2],
        {k:v for k,v in models.items() if k != "PyTorch LSTM"}.items()):
        imps = get_feature_importance(mdl, FCOLS)
        if imps is not None:
            fi_df = pd.DataFrame({"Feature":FCOLS[:len(imps)],
                                   "Importance":imps[:len(FCOLS)]})
            fi_df = fi_df.nlargest(12,"Importance").sort_values("Importance")
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker=dict(color=fi_df["Importance"],
                            colorscale=[[0,"#0f2035"],[0.5,MODEL_COLORS.get(name,"#38bdf8")],[1,"#22d3ee"]]),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
            fig_fi.update_layout(**plotly_layout(360),
                title=f"{name} — Top Features",
                xaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Importance"),
                yaxis=dict(showgrid=False,automargin=True),
                margin=dict(t=40,b=30,l=140,r=20))
            col.plotly_chart(fig_fi, use_container_width=True)

    # ── Model metrics grouped bar ─────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📊 All Metrics Side-by-Side</p>',
                unsafe_allow_html=True)
    fig_grp = go.Figure()
    for met, clr in [("r2","#22d3ee"),("mae","#818cf8"),("rmse","#4ade80")]:
        fig_grp.add_trace(go.Bar(
            name=met.upper(), x=registry["algorithm"], y=registry[met],
            marker_color=clr, opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>{met.upper()}: %{{y:.3f}}<extra></extra>"))
    fig_grp.update_layout(**plotly_layout(320),
        barmode="group", title="R² / MAE / RMSE — All Models",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    st.plotly_chart(fig_grp, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — HISTORICAL ANALYSIS                               ║
# ╚══════════════════════════════════════════════════════════════╝
elif page == "📈 Historical Analysis":

    ds = df_hist.sort_values("date")

    # ── AQI trend ─────────────────────────────────────────────
    st.markdown('<p class="section-header">📅 AQI Trend Over Time</p>',
                unsafe_allow_html=True)
    roll30 = ds["AQI"].rolling(30, min_periods=1).mean()
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=ds["date"],y=ds["AQI"],mode="lines",name="Daily AQI",
        line=dict(color="#22d3ee",width=1.5),fill="tozeroy",
        fillcolor="rgba(34,211,238,0.07)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>AQI: %{y:.1f}<extra></extra>"))
    fig_t.add_trace(go.Scatter(x=ds["date"],y=roll30,mode="lines",name="30-day avg",
        line=dict(color="#fbbf24",width=2,dash="dash"),
        hovertemplate="30d avg %{x|%d %b}: %{y:.1f}<extra></extra>"))
    fig_t.update_layout(**plotly_layout(320),
        title="Historical AQI with 30-Day Rolling Average",
        xaxis=dict(showgrid=False,title=""),
        yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="AQI"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    st.plotly_chart(fig_t, use_container_width=True)
    st.caption("Cyan = daily AQI · Gold dashed = 30-day rolling average")

    # ── Monthly heatmap ───────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🗓️ Monthly AQI Heatmap</p>',
                unsafe_allow_html=True)
    df2 = df_hist.copy()
    df2["year"]  = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month
    pivot = df2.groupby(["year","month"])["AQI"].mean().reset_index().pivot(
        index="year",columns="month",values="AQI")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    fig_hm = px.imshow(pivot, text_auto=".0f", aspect="auto",
        color_continuous_scale=["#22c55e","#eab308","#f97316","#ef4444","#a855f7"],
        zmin=0, zmax=200,
        labels={"color":"AQI"})
    fig_hm.update_traces(
        hovertemplate="<b>%{y} %{x}</b><br>Avg AQI: %{z:.1f}<extra></extra>",
        textfont=dict(size=10))
    fig_hm.update_layout(**plotly_layout(240),
        coloraxis_colorbar=dict(title="AQI",thickness=12))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Distribution + outliers ───────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📦 Distribution & Outliers</p>',
                unsafe_allow_html=True)
    d1,d2 = st.columns(2)
    with d1:
        fig_h = px.histogram(df_hist, x="AQI", nbins=40,
            color_discrete_sequence=["#22d3ee"],
            labels={"AQI":"AQI","count":"Days"})
        fig_h.update_traces(
            marker_line_color="rgba(255,255,255,.15)",marker_line_width=0.5,
            hovertemplate="AQI %{x:.0f}<br>Days: %{y}<extra></extra>")
        fig_h.update_layout(**plotly_layout(300),
            title="AQI Distribution",
            xaxis=dict(showgrid=False,title="AQI"),
            yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Days"))
        st.plotly_chart(fig_h, use_container_width=True, key="hist_dist")

    with d2:
        fig_b = px.box(df_hist, y="AQI", points="outliers",
            color_discrete_sequence=["#818cf8"])
        fig_b.update_traces(
            marker=dict(color="#ef4444",size=5,opacity=0.7),
            line=dict(color="#818cf8"),
            hovertemplate="AQI: %{y:.1f}<extra></extra>")
        fig_b.update_layout(**plotly_layout(300),
            title="AQI Outliers",
            yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="AQI"))
        st.plotly_chart(fig_b, use_container_width=True, key="hist_box")

    # ── Correlation heatmap ───────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔗 Feature Correlation Heatmap</p>',
                unsafe_allow_html=True)
    ccols = ["AQI","temperature","humidity","wind_speed","rain","pressure",
             "pm2_5","pm10","ozone","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide"]
    ccols = [c for c in ccols if c in df_hist.columns]
    cm = df_hist[ccols].corr().round(2)
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
        color_continuous_scale=["#07111F","#0f2035","#818cf8","#22d3ee"],
        zmin=-1, zmax=1)
    fig_cm.update_traces(
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r=%{z:.2f}<extra></extra>",
        textfont=dict(size=9))
    fig_cm.update_layout(**plotly_layout(440),
        margin=dict(t=30,b=80,l=110,r=20),
        coloraxis_colorbar=dict(title="r",thickness=12))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Seasonal analysis ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🍂 AQI by Season</p>',
                unsafe_allow_html=True)
    df3 = df_hist.copy()
    df3["Season"] = df3["date"].dt.month.map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Autumn",10:"Autumn",11:"Autumn"})
    fig_s = px.box(df3, x="Season", y="AQI", color="Season", points="outliers",
        color_discrete_map={"Winter":"#60a5fa","Spring":"#4ade80",
                            "Summer":"#f97316","Autumn":"#f59e0b"},
        category_orders={"Season":["Winter","Spring","Summer","Autumn"]})
    fig_s.update_layout(**plotly_layout(320),
        xaxis=dict(showgrid=False,title=""),
        yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="AQI"),
        showlegend=False)
    st.plotly_chart(fig_s, use_container_width=True, key="seasonal_box")

    # ── Pollutant trends tabs ─────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Pollutant Trend Graphs</p>',
                unsafe_allow_html=True)
    poll_map = {"PM2.5":"pm2_5","PM10":"pm10","Ozone":"ozone",
                "CO":"carbon_monoxide","NO₂":"nitrogen_dioxide","SO₂":"sulphur_dioxide"}
    poll_colors = ["#22d3ee","#818cf8","#4ade80","#f59e0b","#ef4444","#06b6d4"]
    tabs = st.tabs(list(poll_map.keys()))
    for tab,(name,col),clr in zip(tabs, poll_map.items(), poll_colors):
        with tab:
            if col not in ds.columns: continue
            roll = ds[col].rolling(30,min_periods=1).mean()
            s1,s2,s3,s4 = st.columns(4)
            s1.metric(f"Current {name}", f"{ds[col].iloc[-1]:.2f}")
            s2.metric("30-day Avg",      f"{roll.iloc[-1]:.2f}")
            s3.metric("Min",             f"{ds[col].min():.2f}")
            s4.metric("Max",             f"{ds[col].max():.2f}")
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=ds["date"],y=ds[col],mode="lines",
                line=dict(color=clr,width=1.2),fill="tozeroy",
                fillcolor="rgba(34,211,238,0.05)",name=name,
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.2f}}<extra></extra>"))
            fig_p.add_trace(go.Scatter(x=ds["date"],y=roll,mode="lines",
                line=dict(color="#fbbf24",width=1.5,dash="dash"),name="30-day avg",
                hovertemplate=f"30d avg: %{{y:.2f}}<extra></extra>"))
            fig_p.update_layout(**plotly_layout(240),
                xaxis=dict(showgrid=False,title=""),
                yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title=f"{name} µg/m³"),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10),
                            orientation="h",y=1.12,xanchor="right",x=1))
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Summary Statistics</p>',
                unsafe_allow_html=True)
    a,b,c,d,e = st.columns(5)
    a.metric("Mean AQI",   f"{df_hist['AQI'].mean():.1f}")
    b.metric("Max AQI",    f"{df_hist['AQI'].max():.1f}")
    c.metric("Min AQI",    f"{df_hist['AQI'].min():.1f}")
    d.metric("Std Dev",    f"{df_hist['AQI'].std():.1f}")
    e.metric("Total Days", f"{len(df_hist):,}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — EXPLAINABILITY & CUSTOM PREDICTION               ║
# ╚══════════════════════════════════════════════════════════════╝
elif page == "🔬 Explainability & Custom Prediction":

    # ── Feature importance (explainability) ───────────────────
    st.markdown('<p class="section-header">🔬 Feature Importance — All Models</p>',
                unsafe_allow_html=True)
    exp_tabs = st.tabs(["Random Forest", "Ridge Regression", "Correlation with AQI"])

    with exp_tabs[0]:
        mdl = models.get("Random Forest")
        if mdl:
            imps = get_feature_importance(mdl, FCOLS)
            if imps is not None:
                fi_df = pd.DataFrame({"Feature":FCOLS[:len(imps)],
                                       "Importance":imps[:len(FCOLS)]})
                fi_df = fi_df.sort_values("Importance",ascending=True).tail(15)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                    marker=dict(color=fi_df["Importance"],
                                colorscale=[[0,"#0f2035"],[0.5,"#818cf8"],[1,"#22d3ee"]],
                                showscale=True,colorbar=dict(title="Importance",thickness=10)),
                    text=[f"{v:.4f}" for v in fi_df["Importance"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>"))
                fig_fi.update_layout(**plotly_layout(420),
                    title="Random Forest — Top 15 Features",
                    xaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Importance"),
                    yaxis=dict(showgrid=False,automargin=True),
                    margin=dict(t=40,b=30,l=150,r=80))
                st.plotly_chart(fig_fi, use_container_width=True, key="rf_fi")

    with exp_tabs[1]:
        mdl = models.get("Ridge Regression")
        if mdl:
            imps = get_feature_importance(mdl, FCOLS)
            if imps is not None:
                fi_df2 = pd.DataFrame({"Feature":FCOLS[:len(imps)],"Importance":imps[:len(FCOLS)]})
                fi_df2 = fi_df2.sort_values("Importance",ascending=True).tail(15)
                fig_ri = go.Figure(go.Bar(
                    x=fi_df2["Importance"], y=fi_df2["Feature"], orientation="h",
                    marker_color="#4ade80",
                    hovertemplate="<b>%{y}</b><br>|Coef|: %{x:.4f}<extra></extra>"))
                fig_ri.update_layout(**plotly_layout(420),
                    title="Ridge Regression — Top 15 |Coefficients|",
                    xaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="|Coefficient|"),
                    yaxis=dict(showgrid=False,automargin=True),
                    margin=dict(t=40,b=30,l=150,r=30))
                st.plotly_chart(fig_ri, use_container_width=True, key="ridge_fi")

    with exp_tabs[2]:
        ccols2 = [c for c in ["pm2_5","pm10","ozone","temperature","humidity",
                               "wind_speed","AQI_lag_1","AQI_3day_mean"] if c in df_hist.columns]
        corrs = df_hist[ccols2].corrwith(df_hist["AQI"]).sort_values()
        clrs2 = ["#ef4444" if v > 0 else "#22c55e" for v in corrs.values]
        fig_corr = go.Figure(go.Bar(
            x=corrs.values, y=corrs.index, orientation="h",
            marker_color=clrs2,
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"))
        fig_corr.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,.3)")
        fig_corr.update_layout(**plotly_layout(340),
            title="Feature Correlation with AQI",
            xaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",title="Correlation"),
            yaxis=dict(showgrid=False,automargin=True),
            margin=dict(t=40,b=30,l=150,r=20))
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_chart")
        st.caption("🔴 Positive = associated with higher AQI · 🟢 Negative = associated with lower AQI")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # SECTION A — MANUAL FEATURE INPUT
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(34,211,238,.08),rgba(129,140,248,.05));
    border:1px solid rgba(34,211,238,.2);border-left:4px solid #22d3ee;
    border-radius:14px;padding:14px 20px;margin:10px 0 18px;'>
        <p style='font-family:Poppins;font-size:1.05rem;font-weight:700;
        color:#22d3ee;margin:0 0 4px;'>🎛️ Section A — Manual AQI Prediction</p>
        <p style='font-size:.85rem;color:#64748b;margin:0;'>
        Enter feature values below and predict AQI using all three models.</p>
    </div>""", unsafe_allow_html=True)

    last = df_hist.iloc[-1]

    with st.form("manual_pred_form"):
        st.markdown("#### 🌤️ Weather Parameters")
        c1,c2,c3 = st.columns(3)
        temp     = c1.number_input("Temperature (°C)", value=float(last["temperature"]), step=0.1)
        humidity = c2.number_input("Humidity (%)",      value=float(last["humidity"]),    step=1.0)
        pressure = c3.number_input("Pressure (hPa)",    value=float(last["pressure"]),    step=0.1)
        c4,c5    = st.columns(2)
        wind     = c4.number_input("Wind Speed (km/h)", value=float(last["wind_speed"]), step=0.1)
        rain     = c5.number_input("Rainfall (mm)",     value=float(last["rain"]),       step=0.01)

        st.markdown("#### 🏭 Pollutant Parameters")
        p1,p2,p3 = st.columns(3)
        pm2_5 = p1.number_input("PM2.5 (µg/m³)", value=float(last["pm2_5"]),           step=0.1)
        pm10  = p2.number_input("PM10 (µg/m³)",  value=float(last["pm10"]),            step=0.1)
        ozone = p3.number_input("Ozone (µg/m³)", value=float(last["ozone"]),            step=0.1)
        p4,p5,p6 = st.columns(3)
        co  = p4.number_input("CO (µg/m³)",  value=float(last["carbon_monoxide"]),   step=1.0)
        no2 = p5.number_input("NO₂ (µg/m³)", value=float(last["nitrogen_dioxide"]),  step=0.1)
        so2 = p6.number_input("SO₂ (µg/m³)", value=float(last["sulphur_dioxide"]),   step=0.1)

        submitted = st.form_submit_button("🔮 Predict AQI with All Models",
                                          use_container_width=True)

    if submitted:
        row = last.copy()
        for k,v in [("temperature",temp),("humidity",humidity),("pressure",pressure),
                    ("wind_speed",wind),("rain",rain),("pm2_5",pm2_5),("pm10",pm10),
                    ("ozone",ozone),("carbon_monoxide",co),("nitrogen_dioxide",no2),
                    ("sulphur_dioxide",so2),
                    ("rain_flag",1 if rain>0 else 0),("temp_humidity",temp*humidity)]:
            row[k] = v
        X_in = pd.DataFrame([row[FCOLS]])

        results = []
        for name, mdl in models.items():
            try:
                pred = predict_single(mdl, X_in)
                results.append({"model":name, "pred":pred,
                                 "color":MODEL_COLORS.get(name,"#38bdf8")})
            except Exception as e:
                st.warning(f"{name}: {e}")

        if results:
            # Production result hero
            prod_res = next((r for r in results if "LSTM" in r["model"]), results[0])
            pc = aqi_color(prod_res["pred"])
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(34,211,238,.08),rgba(15,32,53,.8));
            border-radius:18px;padding:22px;border:1px solid rgba(34,211,238,.25);
            text-align:center;margin:14px 0;">
                <p style="font-size:9px;font-weight:700;letter-spacing:.18em;
                text-transform:uppercase;color:#475569;margin:0 0 4px;">
                Production Model — {prod_res['model']}</p>
                <span style="font-family:Poppins;font-size:72px;font-weight:800;
                color:{pc};line-height:1;">{prod_res['pred']:.0f}</span>
                <span style="font-size:14px;color:#475569;"> AQI</span>
                <div style="margin-top:8px;font-size:13px;color:#e2e8f0;font-weight:600;">
                {aqi_label(prod_res['pred'])}</div>
            </div>""", unsafe_allow_html=True)

            # All model results
            if len(results) > 1:
                st.markdown("#### All Model Predictions")
                res_cols = st.columns(len(results))
                for col, res in zip(res_cols, results):
                    rc = aqi_color(res["pred"])
                    col.markdown(f"""
                    <div class="card" style="text-align:center;border-top:3px solid {rc};">
                        <p style="font-size:9px;color:#475569;text-transform:uppercase;
                        letter-spacing:.1em;margin:0 0 3px;">{res['model']}</p>
                        <span style="font-family:Poppins;font-size:32px;font-weight:800;
                        color:{rc};">{res['pred']:.0f}</span>
                        <p style="font-size:10px;color:{rc};margin:4px 0 0;">
                        {aqi_label(res['pred'])}</p>
                    </div>""", unsafe_allow_html=True)

                # Gauge comparison
                fig_g = go.Figure()
                for i, res in enumerate(results):
                    fig_g.add_trace(go.Indicator(
                        mode="gauge+number", value=res["pred"],
                        title={"text":res["model"],"font":{"size":10,"color":"#94a3b8"}},
                        number={"font":{"size":20,"color":aqi_color(res["pred"])}},
                        gauge={"axis":{"range":[0,250]},
                               "bar":{"color":aqi_color(res["pred"]),"thickness":0.25},
                               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                               "steps":[{"range":[0,50],"color":"rgba(34,197,94,.07)"},
                                        {"range":[50,100],"color":"rgba(234,179,8,.07)"},
                                        {"range":[100,150],"color":"rgba(249,115,22,.07)"},
                                        {"range":[150,250],"color":"rgba(239,68,68,.07)"}]},
                        domain={"column":i,"row":0}))
                fig_g.update_layout(
                    grid={"rows":1,"columns":len(results)},
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter",color="#e2e8f0",size=10),
                    margin=dict(t=20,b=20,l=10,r=10), height=220)
                st.plotly_chart(fig_g, use_container_width=True)

                # Bar comparison
                fig_bar = go.Figure(go.Bar(
                    x=[r["model"] for r in results],
                    y=[r["pred"] for r in results],
                    marker_color=[r["color"] for r in results],
                    text=[f"{r['pred']:.1f}" for r in results],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Predicted AQI: %{y:.1f}<extra></extra>"))
                fig_bar.update_layout(**plotly_layout(280),
                    title="Model Prediction Comparison",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True,gridcolor="rgba(51,65,85,.4)",
                               title="Predicted AQI", range=[0, max(r["pred"] for r in results)*1.3]),
                    showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # SECTION B — KARACHI AREA SELECTOR
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(129,140,248,.08),rgba(74,222,128,.05));
    border:1px solid rgba(129,140,248,.2);border-left:4px solid #818cf8;
    border-radius:14px;padding:14px 20px;margin:10px 0 18px;'>
        <p style='font-family:Poppins;font-size:1.05rem;font-weight:700;
        color:#818cf8;margin:0 0 4px;'>📍 Section B — Karachi Area AQI Lookup</p>
        <p style='font-size:.85rem;color:#64748b;margin:0;'>
        Select a Karachi area to fetch its live AQI from OpenWeather API.</p>
    </div>""", unsafe_allow_html=True)

    area_sel = st.selectbox("Select Karachi Area", list(KARACHI_AREAS.keys()))
    alat, alon = KARACHI_AREAS[area_sel]

    if st.button("🌍 Fetch Live AQI for Selected Area", use_container_width=True):
        with st.spinner(f"Fetching live data for {area_sel}…"):
            area_poll = fetch_live_aqi(alat, alon)

        if area_poll:
            la = area_poll["aqi"]; lc = aqi_color(la)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(129,140,248,.10),rgba(15,32,53,.8));
            border-radius:16px;padding:20px;border:1px solid rgba(129,140,248,.3);
            text-align:center;margin:12px 0;">
                <p style="font-size:9px;font-weight:700;letter-spacing:.15em;
                text-transform:uppercase;color:#475569;margin:0 0 4px;">
                📍 {area_sel.upper()}</p>
                <span style="font-family:Poppins;font-size:60px;font-weight:800;
                color:{lc};line-height:1;">{la:.0f}</span>
                <span style="font-size:13px;color:#475569;"> AQI</span>
                <div style="margin-top:8px;font-size:13px;color:#e2e8f0;font-weight:600;">
                {aqi_label(la)}</div>
                <div style="font-size:11px;color:#475569;margin-top:4px;">{aqi_alert(la)}</div>
            </div>""", unsafe_allow_html=True)

            # Pollutant metrics
            p1,p2,p3 = st.columns(3)
            p1.metric("PM2.5", f"{area_poll['pm2_5']:.2f} µg/m³")
            p2.metric("PM10",  f"{area_poll['pm10']:.2f} µg/m³")
            p3.metric("Ozone", f"{area_poll['ozone']:.2f} µg/m³")
            p4,p5,p6 = st.columns(3)
            p4.metric("CO",  f"{area_poll['carbon_monoxide']:.2f} µg/m³")
            p5.metric("NO₂", f"{area_poll['nitrogen_dioxide']:.2f} µg/m³")
            p6.metric("SO₂", f"{area_poll['sulphur_dioxide']:.2f} µg/m³")

            # Gauge
            fig_ag = go.Figure(go.Indicator(
                mode="gauge+number", value=la,
                title={"text":f"Live AQI — {area_sel}","font":{"size":13,"color":"#94a3b8"}},
                number={"font":{"size":32,"color":lc}},
                gauge={"axis":{"range":[0,300]},
                       "bar":{"color":lc,"thickness":0.22},
                       "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                       "steps":[{"range":[0,50],"color":"rgba(34,197,94,.1)"},
                                 {"range":[50,100],"color":"rgba(234,179,8,.1)"},
                                 {"range":[100,150],"color":"rgba(249,115,22,.1)"},
                                 {"range":[150,200],"color":"rgba(239,68,68,.1)"},
                                 {"range":[200,300],"color":"rgba(168,85,247,.1)"}]}))
            fig_ag.update_layout(**plotly_layout(260, margin=dict(t=30,b=10,l=20,r=20)))
            st.plotly_chart(fig_ag, use_container_width=True)

            # Area map
            map_a = pd.DataFrame({"lat":[alat],"lon":[alon],
                                   "name":[area_sel],"AQI":[la]})
            fig_ml = px.scatter_mapbox(map_a,lat="lat",lon="lon",hover_name="name",
                hover_data={"AQI":True,"lat":False,"lon":False},
                color_discrete_sequence=[lc],zoom=12,height=300)
            fig_ml.update_traces(marker=dict(size=18))
            fig_ml.update_layout(mapbox_style="carto-darkmatter",
                mapbox_center={"lat":alat,"lon":alon},
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=0,b=0,l=0,r=0),height=300)
            st.plotly_chart(fig_ml, use_container_width=True)
        else:
            st.error("Could not fetch live data. Check your API key or internet connection.")

    # ── All Karachi areas map ─────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🗺️ Karachi — All Monitoring Areas</p>',
                unsafe_allow_html=True)
    kdf = pd.DataFrame([
        {"Location":n,"lat":lt,"lon":ln,
         "AQI (est.)":round(live_aqi+(lt-LAT)*40+(ln-LON)*25,1)}
        for n,(lt,ln) in KARACHI_AREAS.items()])
    fig_km = px.scatter_mapbox(kdf,lat="lat",lon="lon",hover_name="Location",
        hover_data={"AQI (est.)":True,"lat":False,"lon":False},
        color="AQI (est.)",
        color_continuous_scale=["#22c55e","#eab308","#f97316","#a855f7"],
        size_max=16, zoom=10, height=380)
    fig_km.update_traces(marker=dict(size=14))
    fig_km.update_layout(mapbox_style="carto-darkmatter",
        mapbox_center={"lat":24.86,"lon":67.07},
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0,b=0,l=0,r=0),height=380,
        coloraxis_colorbar=dict(title="AQI",thickness=12))
    st.plotly_chart(fig_km, use_container_width=True)
    st.caption("AQI estimates based on spatial interpolation from the live Defence Phase 7 reading.")

    # Footer
    st.markdown("---")
    st.markdown("""<div style="text-align:center;">
        <span class="footer-chip">⚡ FastAPI</span>
        <span class="footer-chip">🎈 Streamlit</span>
        <span class="footer-chip">🌲 Scikit-Learn</span>
        <span class="footer-chip">🔥 PyTorch</span>
        <span class="footer-chip">☁ OpenWeather</span>
        <span class="footer-chip">📊 Plotly</span>
        <span class="footer-chip">🏗 Hopsworks</span>
    </div>
    <p style="text-align:center;color:#334155;font-size:.78rem;margin-top:10px;">
    Built by <strong style="color:#fbbf24;">Maham Ahmed</strong> ·
    Bachelor of Data Science · 2026 · 📍 Karachi
    </p>""", unsafe_allow_html=True)
