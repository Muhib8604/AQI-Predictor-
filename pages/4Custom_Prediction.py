"""
pages/Custom_Prediction.py
UI-only multi-area lab. Does not change training / feature / predict pipelines.
"""

import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

from feature_snapshot import build_today_features
from predict import predict_all_horizons

load_dotenv()

st.set_page_config(page_title="Custom AQI Lab", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; color:#e7ecef; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #12201a 0%, #080a0d 45%, #080a0d 100%); }
.page-title { font-family:'Outfit',sans-serif; font-size:2.1rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em; }
.page-sub { color:#94a3b8; font-size:0.95rem; margin-bottom:1.2rem; }
.section-label { font-family:'Outfit',sans-serif; font-size:0.72rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#34d399; margin:0.4rem 0 0.6rem; }
.glass { background:rgba(16,22,28,0.72); border:1px solid rgba(52,211,153,0.18); border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 10px 40px rgba(0,0,0,0.35),0 0 28px rgba(52,211,153,0.06); }
.glow-card { background:linear-gradient(160deg,rgba(20,28,34,0.95),rgba(12,16,20,0.98)); border:1px solid rgba(52,211,153,0.22); border-radius:18px; padding:1.1rem; box-shadow:0 0 24px rgba(52,211,153,0.08); }
.pill { display:inline-block; background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.28); color:#6ee7b7; padding:0.2rem 0.7rem; border-radius:999px; font-size:0.75rem; font-weight:600; }
.pill-warn { background:rgba(251,191,36,0.1); border-color:rgba(251,191,36,0.3); color:#fbbf24; }
div[data-baseweb="slider"] [role="slider"] {
  background-color: #34d399 !important;
  border-color: #34d399 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
  background: #34d399 !important;
}
div[data-testid="stMetric"] { background:rgba(14,18,22,0.85); border:1px solid rgba(52,211,153,0.16); border-radius:14px; padding:0.85rem 1rem; }
.stButton > button { background:linear-gradient(90deg,#059669,#34d399)!important; color:#04140e!important; font-family:'Outfit',sans-serif!important; font-weight:700!important; border:none!important; border-radius:12px!important; box-shadow:0 8px 24px rgba(16,185,129,0.25); }
</style>
""", unsafe_allow_html=True)

# UI-only station map (best-effort Karachi pins). Offline → Unavailable.
LOCATIONS = {
    "Zafar Memon DHA": {"lat": 24.8050, "lon": 67.0450, "station_id": "162592", "area": "DHA Phase 6"},
    "Saddar": {"lat": 24.8607, "lon": 67.0011, "station_id": "A471613", "area": "Saddar Town"},
    "Clifton": {"lat": 24.8138, "lon": 67.0267, "station_id": "162592", "area": "Clifton (nearest DHA feed)"},
    "Gulshan-e-Iqbal": {"lat": 24.9056, "lon": 67.0822, "station_id": "A471613", "area": "Gulshan"},
    "North Nazimabad": {"lat": 24.9420, "lon": 67.0450, "station_id": "162592", "area": "North Nazimabad"},
    "Korangi": {"lat": 24.8500, "lon": 67.1500, "station_id": "A471613", "area": "Korangi"},
    "Malir": {"lat": 24.8930, "lon": 67.1950, "station_id": "162592", "area": "Malir"},
    "PECHS": {"lat": 24.8730, "lon": 67.0620, "station_id": "A471613", "area": "PECHS / Tariq Road"},
}


def fetch_station_aqi(station_id: str):
    """Page-local fetch only — does not modify project pipeline modules."""
    token = os.getenv("AQICN_API_KEY")
    if not token:
        return None, "Missing AQICN_API_KEY"
    sid = str(station_id).lstrip("@")
    url = f"https://api.waqi.info/feed/@{sid}/?token={token}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            return None, "Station error"
        payload = data.get("data") or {}
        aqi = payload.get("aqi")
        name = (payload.get("city") or {}).get("name", sid)
        if aqi is None:
            return None, "Station offline"
        return {"aqi": float(aqi), "name": name}, None
    except Exception as e:
        return None, str(e)


st.markdown('<div class="page-title">Custom AQI Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Pick an area for live station reading + map. '
    'Scenario prediction still uses your existing model baseline (unchanged).</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1])
with left:
    st.markdown('<div class="section-label">Location</div>', unsafe_allow_html=True)
    selected = st.selectbox("Area", list(LOCATIONS.keys()), index=0, label_visibility="collapsed")
    meta = LOCATIONS[selected]

    live, err = fetch_station_aqi(meta["station_id"])
    if live:
        aqi_block = f'<div style="font-size:2rem;font-family:Outfit;font-weight:800;color:#34d399;">{live["aqi"]:.0f}</div><div style="color:#94a3b8;font-size:0.85rem;">Live AQI · {live["name"]}</div>'
        pill = '<span class="pill">Station online</span>'
    else:
        aqi_block = f'<div style="font-size:1.1rem;color:#fbbf24;">Unavailable</div><div style="color:#64748b;font-size:0.8rem;">{err or "No reading"}</div>'
        pill = '<span class="pill pill-warn">Station offline / no data</span>'

    st.markdown(
        f'<div class="glass" style="margin-top:0.6rem;">'
        f'<div style="font-family:Outfit;font-weight:700;font-size:1.2rem;">{selected}</div>'
        f'<div style="color:#94a3b8;">{meta["area"]} · Karachi</div>'
        f'<div style="margin-top:0.9rem;">{aqi_block}</div>'
        f'<div style="margin-top:0.7rem;">{pill}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-label">Map</div>', unsafe_allow_html=True)
    st.map(pd.DataFrame({"lat": [meta["lat"]], "lon": [meta["lon"]]}), zoom=12)

st.markdown("---")
st.markdown('<div class="section-label">Scenario controls</div>', unsafe_allow_html=True)
st.caption("Sliders override weather/pollutants only. Lags stay from live baseline. No pipeline writes.")

with st.spinner("Loading baseline…"):
    baseline = build_today_features()
if baseline is None:
    st.error("Baseline features unavailable (weather/Hopsworks). Scenario blocked.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Weather**")
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, float(baseline.get("temperature", 25.0)), 0.5)
    humidity = st.slider("Humidity (%)", 0, 100, int(baseline.get("humidity", 50)))
    pressure = st.slider("Pressure (hPa)", 950, 1050, int(baseline.get("pressure", 1010)))
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Wind & rain**")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 30.0, float(baseline.get("wind_speed", 5.0)), 0.5)
    rain = st.slider("Rain (mm)", 0.0, 100.0, float(baseline.get("rain", 0.0)), 0.5)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Particulates**")
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, float(baseline.get("pm10", 80.0)), 1.0)
    pm2_5 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(baseline.get("pm2_5", 40.0)), 1.0)
    st.markdown("</div>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    carbon_monoxide = st.slider("CO (µg/m³)", 0.0, 3000.0, float(baseline.get("carbon_monoxide", 800.0)), 10.0)
with c5:
    nitrogen_dioxide = st.slider("NO₂ (µg/m³)", 0.0, 200.0, float(baseline.get("nitrogen_dioxide", 40.0)), 1.0)
with c6:
    sulphur_dioxide = st.slider("SO₂ (µg/m³)", 0.0, 200.0, float(baseline.get("sulphur_dioxide", 30.0)), 1.0)
    ozone = st.slider("Ozone (µg/m³)", 0.0, 300.0, float(baseline.get("ozone", 90.0)), 1.0)

if st.button("Run scenario prediction"):
    feats = dict(baseline)
    feats.update({
        "temperature": temperature, "humidity": humidity, "pressure": pressure,
        "wind_speed": wind_speed, "rain": rain, "pm10": pm10, "pm2_5": pm2_5,
        "carbon_monoxide": carbon_monoxide, "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide": sulphur_dioxide, "ozone": ozone,
    })
    try:
        with st.spinner("Running models…"):
            results = predict_all_horizons(feats)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    st.markdown(f'<div class="section-label">Results · {selected}</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Day 1", f"{results['day1']['predicted_aqi']:.1f}", results["day1"]["model_used"])
    r2.metric("Day 2", f"{results['day2']['predicted_aqi']:.1f}", results["day2"]["model_used"])
    r3.metric("Day 3", f"{results['day3']['predicted_aqi']:.1f}", results["day3"]["model_used"])
    r4.metric("3-day avg", f"{results['average_aqi']:.1f}")