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
/* Slider thumb: dark center, green ring (not a solid green dot) */
div[data-baseweb="slider"] [role="slider"] {
  background-color: #0b1310 !important;
  border: 2px solid #34d399 !important;
  box-shadow: 0 0 0 3px rgba(52,211,153,0.15) !important;
}
/* Slider filled track: outline only, transparent fill (no solid green bar) */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
  background: transparent !important;
  border: 1.5px solid #34d399 !important;
  border-radius: 999px !important;
}
/* Slider unfilled base track stays subtle/dark */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
  background: rgba(255,255,255,0.06) !important;
}
div[data-testid="stMetric"] { background:rgba(14,18,22,0.85); border:1px solid rgba(52,211,153,0.16); border-radius:14px; padding:0.85rem 1rem; }
.stButton > button { background:linear-gradient(90deg,#059669,#34d399)!important; color:#04140e!important; font-family:'Outfit',sans-serif!important; font-weight:700!important; border:none!important; border-radius:12px!important; box-shadow:0 8px 24px rgba(16,185,129,0.25); }
</style>
""", unsafe_allow_html=True)

# Karachi-area locations, resolved via their exact AQICN station IDs
# (not guessed lat/lon) — this is authoritative and avoids ever
# matching a station in the wrong city/country.
#
# IMPORTANT: these IDs keep their "A" prefix and are used as-is in the
# URL (no "@"), matching the working pattern in this project's aqi.py.
LOCATIONS = {
    "Zafar Memon DHA": {"station_id": "A545140", "area": "DHA Phase 6"},
    "Saddar": {"station_id": "A544708", "area": "Saddar Town"},
    "Clifton": {"station_id": "A547342", "area": "Clifton"},
    "Gulshan-e-Iqbal": {"station_id": "A545320", "area": "Gulshan"},
    "North Nazimabad": {"station_id": "A545017", "area": "North Nazimabad"},
    "Korangi": {"station_id": "A544699", "area": "Korangi"},
    "Malir": {"station_id": "A545422", "area": "Malir"},
    "PECHS": {"station_id": "A544681", "area": "PECHS / Tariq Road"},
}


def _iaqi_value(iaqi: dict, key: str):
    entry = iaqi.get(key)
    return entry.get("v") if isinstance(entry, dict) else None


def fetch_station_by_id(station_id: str):
    """Fetch a specific, known-correct AQICN station by its station ID.

    Page-local fetch only — does not modify project pipeline modules.
    Also returns the station's own reported lat/lon (from AQICN's
    response) so the map always matches the station, and individual
    pollutant/weather sub-readings (iaqi) so the scenario sliders below
    can default to this station's real numbers.

    IMPORTANT: these station IDs (e.g. "A545140") are used AS-IS in the
    URL, with no "@" prefix added — this matches the working format
    already used in this project's aqi.py. Adding "@" here breaks the
    lookup ("Unknown ID").
    """
    token = os.getenv("AQICN_API_KEY")
    if not token:
        return None, "Missing AQICN_API_KEY"
    sid = str(station_id).strip()
    url = f"https://api.waqi.info/feed/{sid}/"
    try:
        r = requests.get(url, params={"token": token}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            return None, "Station lookup failed"
        payload = data.get("data") or {}
        aqi = payload.get("aqi")
        if aqi is None or aqi == "-":
            return None, "No live reading from this station"
        iaqi = payload.get("iaqi") or {}
        city = payload.get("city") or {}
        name = city.get("name", "Station")
        geo = city.get("geo") or [None, None]
        return {
            "aqi": float(aqi),
            "name": name,
            "lat": geo[0],
            "lon": geo[1],
            "pm2_5": _iaqi_value(iaqi, "pm25"),
            "pm10": _iaqi_value(iaqi, "pm10"),
            "ozone": _iaqi_value(iaqi, "o3"),
            "carbon_monoxide": _iaqi_value(iaqi, "co"),
            "nitrogen_dioxide": _iaqi_value(iaqi, "no2"),
            "sulphur_dioxide": _iaqi_value(iaqi, "so2"),
            "temperature": _iaqi_value(iaqi, "t"),
            "humidity": _iaqi_value(iaqi, "h"),
            "pressure": _iaqi_value(iaqi, "p"),
            "wind_speed": _iaqi_value(iaqi, "w"),
        }, None
    except Exception as e:
        return None, str(e)


st.markdown('<div class="page-title">🧪 Custom AQI Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Pick an area for live station reading + map. '
    'Scenario prediction still uses your existing model baseline (unchanged).</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1])
with left:
    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    selected = st.selectbox("Area", list(LOCATIONS.keys()), index=0, label_visibility="collapsed")
    meta = LOCATIONS[selected]

    live, err = fetch_station_by_id(meta["station_id"])
    if live:
        aqi_block = f'<div style="font-size:2rem;font-family:Outfit;font-weight:800;color:#34d399;">{live["aqi"]:.0f}</div><div style="color:#94a3b8;font-size:0.85rem;">Live AQI · {live["name"]}</div>'
        pill = '<span class="pill">🟢 Station online</span>'
    else:
        aqi_block = f'<div style="font-size:1.1rem;color:#fbbf24;">Unavailable</div><div style="color:#64748b;font-size:0.8rem;">{err or "No reading"}</div>'
        pill = '<span class="pill pill-warn">⚠️ Station offline / no data</span>'

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
    st.markdown('<div class="section-label">🗺️ Map</div>', unsafe_allow_html=True)
    # Use the station's own reported coordinates (from AQICN) if we have
    # a live reading; otherwise fall back to a Karachi-wide view instead
    # of a guessed pin, since we no longer hardcode per-area lat/lon.
    if live and live.get("lat") is not None and live.get("lon") is not None:
        map_lat, map_lon, map_zoom = live["lat"], live["lon"], 12
    else:
        map_lat, map_lon, map_zoom = 24.8607, 67.0011, 10  # Karachi city center
    st.map(pd.DataFrame({"lat": [map_lat], "lon": [map_lon]}), zoom=map_zoom)

st.markdown("---")
st.markdown('<div class="section-label">🎛️ Scenario controls</div>', unsafe_allow_html=True)
st.caption(
    "Sliders default to the selected area's live station reading (where available), "
    "otherwise fall back to the model baseline. Lags stay from the baseline. No pipeline writes."
)

with st.spinner("Loading baseline…"):
    baseline = build_today_features()
if baseline is None:
    st.error("Baseline features unavailable (weather/Hopsworks). Scenario blocked.")
    st.stop()


def default_for(key: str, fallback: float = 0.0) -> float:
    """Prefer the selected area's live station reading; fall back to the model baseline."""
    if live and live.get(key) is not None:
        try:
            return float(live[key])
        except (TypeError, ValueError):
            pass
    try:
        return float(baseline.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


# Widgets are keyed by `selected` so switching the area actually resets
# each slider to that area's live numbers instead of keeping whatever
# value was left over from the previous area.
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Weather**")
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, default_for("temperature", 25.0), 0.5, key=f"temp_{selected}")
    humidity = st.slider("Humidity (%)", 0, 100, int(default_for("humidity", 50)), key=f"hum_{selected}")
    pressure = st.slider("Pressure (hPa)", 950, 1050, int(default_for("pressure", 1010)) or 1010, key=f"pres_{selected}")
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Wind & rain**")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 30.0, default_for("wind_speed", 5.0), 0.5, key=f"wind_{selected}")
    rain = st.slider("Rain (mm)", 0.0, 100.0, float(baseline.get("rain", 0.0)), 0.5, key=f"rain_{selected}")
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("**Particulates**")
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, default_for("pm10", 80.0), 1.0, key=f"pm10_{selected}")
    pm2_5 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, default_for("pm2_5", 40.0), 1.0, key=f"pm25_{selected}")
    st.markdown("</div>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    carbon_monoxide = st.slider("CO (µg/m³)", 0.0, 3000.0, default_for("carbon_monoxide", 800.0), 10.0, key=f"co_{selected}")
with c5:
    nitrogen_dioxide = st.slider("NO₂ (µg/m³)", 0.0, 200.0, default_for("nitrogen_dioxide", 40.0), 1.0, key=f"no2_{selected}")
with c6:
    sulphur_dioxide = st.slider("SO₂ (µg/m³)", 0.0, 200.0, default_for("sulphur_dioxide", 30.0), 1.0, key=f"so2_{selected}")
    ozone = st.slider("Ozone (µg/m³)", 0.0, 300.0, default_for("ozone", 90.0), 1.0, key=f"o3_{selected}")

if st.button("▶️ Run scenario prediction"):
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
    st.markdown(f'<div class="section-label">📊 Results · {selected}</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Day 1", f"{results['day1']['predicted_aqi']:.1f}", results["day1"]["model_used"])
    r2.metric("Day 2", f"{results['day2']['predicted_aqi']:.1f}", results["day2"]["model_used"])
    r3.metric("Day 3", f"{results['day3']['predicted_aqi']:.1f}", results["day3"]["model_used"])
    r4.metric("3-day avg", f"{results['average_aqi']:.1f}")
