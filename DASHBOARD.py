import streamlit as st
import json
import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="◈",
    layout="wide",
)

# ============================================================
# DESIGN SYSTEM — graphite + emerald
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    background: #080a0d;
    color: #e7ecef;
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: radial-gradient(1100px 520px at 12% -8%, #14241c 0%, #080a0d 42%, #080a0d 100%);
}

.main { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    line-height: 1.2;
    display: inline-block;
    max-width: 100%;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.35);
    color: #6ee7b7;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 12px;
    vertical-align: middle;
}

.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55);
    animation: pulse 1.6s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
    70% { box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

[data-testid="stMetric"] {
    background: rgba(14, 18, 22, 0.88);
    border: 1px solid rgba(52, 211, 153, 0.16);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
    transition: border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(52, 211, 153, 0.4);
    box-shadow: 0 0 28px rgba(52, 211, 153, 0.12);
}

.card {
    background: linear-gradient(160deg, rgba(20, 28, 34, 0.96), rgba(12, 16, 20, 0.98));
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(52, 211, 153, 0.18);
    box-shadow: 0 0 22px rgba(52, 211, 153, 0.06);
    transition: border-color .25s ease, box-shadow .25s ease;
}
.card:hover {
    border-color: rgba(52, 211, 153, 0.4);
    box-shadow: 0 0 32px rgba(52, 211, 153, 0.12);
}

.legend-wrap {
    display: flex;
    border-radius: 10px;
    overflow: hidden;
    height: 12px;
    margin-top: 10px;
}
.legend-seg { flex: 1; }
.legend-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 6px;
}

.pollutant-row { margin-bottom: 14px; }
.pollutant-name {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-bottom: 4px;
}
.bar-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 1s ease;
}

.model-badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.28);
    color: #a7f3d0;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    margin-top: 6px;
}

hr { border: none; border-top: 1px solid rgba(255, 255, 255, 0.06); }

section[data-testid="stSidebar"] {
    background: #07090c;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.stDownloadButton button {
    background: linear-gradient(90deg, #059669, #34d399) !important;
    color: #04140e !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}

.alert-banner {
    background: linear-gradient(90deg, rgba(251, 113, 133, 0.16), rgba(251, 113, 133, 0.06));
    border: 1px solid rgba(251, 113, 133, 0.45);
    border-left: 4px solid #fb7185;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 14px 0;
    color: #fecdd3;
    font-size: 0.98rem;
}

.monitor-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #a7f3d0;
    margin: 12px 0;
}
.monitor-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34d399;
    animation: pulse 1.6s infinite;
}

.live-compare-card {
    background: linear-gradient(160deg, rgba(20, 28, 34, 0.96), rgba(12, 16, 20, 0.98));
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(52, 211, 153, 0.18);
    margin: 14px 0;
    box-shadow: 0 0 24px rgba(52, 211, 153, 0.06);
}
.live-compare-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #6ee7b7;
    margin-bottom: 6px;
}

.footer-chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    margin: 3px;
    color: #94a3b8;
}

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #080a0d; }
::-webkit-scrollbar-thumb { background: #1f2a24; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #2d3d34; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "temperature", "humidity", "wind_speed", "pressure", "rain",
    "pm25", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
]


def load_manifest():
    if not os.path.exists("model_manifest.json"):
        return None
    with open("model_manifest.json", "r") as f:
        return json.load(f)


manifest = load_manifest()

avg_mae = avg_rmse = avg_r2 = None
if manifest:
    avg_mae = (manifest["day1"]["mae"] + manifest["day2"]["mae"] + manifest["day3"]["mae"]) / 3
    avg_rmse = (manifest["day1"]["rmse"] + manifest["day2"]["rmse"] + manifest["day3"]["rmse"]) / 3
    avg_r2 = (manifest["day1"]["r2"] + manifest["day2"]["r2"] + manifest["day3"]["r2"]) / 3

POLLUTANT_LIMITS = {"pm25": 60, "pm10": 100, "ozone": 100}


def aqi_status(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def aqi_color(aqi):
    if aqi <= 50:
        return "#34d399"
    elif aqi <= 100:
        return "#fbbf24"
    elif aqi <= 150:
        return "#fb923c"
    elif aqi <= 200:
        return "#fb7185"
    elif aqi <= 300:
        return "#c084fc"
    return "#64748b"


ALERT_LOG_PATH = "alert_log.csv"


def log_alert(forecast_date, predicted_aqi, threshold):
    predicted_aqi_rounded = round(float(predicted_aqi), 1)
    entry = pd.DataFrame([{
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_date": forecast_date,
        "predicted_aqi": predicted_aqi_rounded,
        "threshold": threshold,
        "status": aqi_status(predicted_aqi),
    }])
    if os.path.exists(ALERT_LOG_PATH):
        existing = pd.read_csv(ALERT_LOG_PATH)
        if "predicted_aqi" in existing.columns:
            existing["predicted_aqi"] = existing["predicted_aqi"].astype(float).round(1)
        matching = existing[
            (existing["forecast_date"] == forecast_date)
            & (existing["predicted_aqi"] == predicted_aqi_rounded)
        ]
        if not matching.empty:
            return
        entry = pd.concat([existing, entry], ignore_index=True)
    entry.to_csv(ALERT_LOG_PATH, index=False)


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        return pd.read_csv(ALERT_LOG_PATH)
    return pd.DataFrame(columns=["logged_at", "forecast_date", "predicted_aqi", "threshold", "status"])


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Karachi AQI")
st.sidebar.caption("3-day forecast · live station · hazard alerts")
st.sidebar.markdown("---")
st.sidebar.markdown("**Location**")
st.sidebar.success("Zafar Memon DHA, Karachi")
st.sidebar.markdown("---")
st.sidebar.markdown("**Champion models**")
if manifest:
    st.sidebar.write(f"Day 1 · {manifest['day1']['model_type']}")
    st.sidebar.write(f"Day 2 · {manifest['day2']['model_type']}")
    st.sidebar.write(f"Day 3 · {manifest['day3']['model_type']}")
else:
    st.sidebar.write("No model found")

st.sidebar.markdown("---")
st.sidebar.markdown("**Performance**")
if manifest:
    st.sidebar.metric("Average R²", f"{avg_r2:.3f}")
    st.sidebar.metric("Average MAE", f"{avg_mae:.2f}")
    st.sidebar.metric("Average RMSE", f"{avg_rmse:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Alert settings**")
alert_threshold = st.sidebar.slider(
    "Hazard threshold (AQI)",
    min_value=50, max_value=300, value=150, step=10,
    help="Banner + toast when any forecast day meets or exceeds this value.",
)
enable_toast_alert = st.sidebar.checkbox("Toast on alert", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Refresh forecast"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Updated {datetime.now().strftime('%d %b %Y %H:%M')}")

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div style="padding-top:0.25rem; margin-bottom:0.35rem;">
      <span class="hero-title">Karachi AQI Predictor</span>
      <span class="live-badge"><span class="pulse-dot"></span> LIVE</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("3-day AQI forecast · weather & pollutant context · machine learning")

st.markdown(
    f"""
<div class="monitor-badge">
    <span class="monitor-dot"></span>
    Hazard monitoring active — threshold <b>{alert_threshold} AQI</b>
</div>
""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Location", "Zafar Memon DHA", "Karachi")
if manifest:
    k2.metric(
        "Models",
        f"{manifest['day1']['model_type']} / {manifest['day2']['model_type']} / {manifest['day3']['model_type']}",
        "Day 1 / 2 / 3",
    )
    k3.metric("Accuracy", f"R² {avg_r2:.3f}", f"MAE {avg_mae:.2f}")
else:
    k2.metric("Models", "Not trained", "Run training_pipeline.py")
    k3.metric("Accuracy", "N/A", "")
k4.metric("Updated", datetime.now().strftime("%H:%M"), datetime.now().strftime("%d %b"))

st.markdown(
    """
<div class="legend-wrap">
    <div class="legend-seg" style="background:#34d399;"></div>
    <div class="legend-seg" style="background:#fbbf24;"></div>
    <div class="legend-seg" style="background:#fb923c;"></div>
    <div class="legend-seg" style="background:#fb7185;"></div>
    <div class="legend-seg" style="background:#c084fc;"></div>
    <div class="legend-seg" style="background:#64748b;"></div>
</div>
<div class="legend-labels">
    <span>0 Good</span><span>50 Moderate</span><span>100 USG</span>
    <span>150 Unhealthy</span><span>200 Very Unhealthy</span><span>300+ Hazardous</span>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast():
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    r = requests.get(f"{backend_url}/predict")
    return r.status_code, (r.json() if r.status_code == 200 else None)


with st.spinner("Fetching latest forecast…"):
    status_code, payload = fetch_forecast()

if status_code == 200 and payload and "error" in payload:
    st.error(f"Backend error: {payload['error']}")
    st.info(
        "Models may be missing. Run `python training_pipeline.py`, confirm "
        "`model_manifest.json` exists, restart FastAPI, then refresh."
    )

elif status_code == 200 and payload and "3_day_AQI_forecast" in payload:
    data = payload["3_day_AQI_forecast"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    latest = df.iloc[0]["predicted_aqi"]

    exceeded_days = df[df["predicted_aqi"] >= alert_threshold]
    if not exceeded_days.empty:
        worst = exceeded_days.loc[exceeded_days["predicted_aqi"].idxmax()]
        st.markdown(
            f"""
        <div class="alert-banner">
        <b>Hazard alert</b> — {worst['date'].strftime('%d %b')} forecast AQI
        <b>{worst['predicted_aqi']:.0f}</b> meets or exceeds threshold {alert_threshold}.
        Status: {aqi_status(worst['predicted_aqi'])}
        </div>
        """,
            unsafe_allow_html=True,
        )
        if enable_toast_alert:
            st.toast(f"Hazardous AQI forecast: {worst['predicted_aqi']:.0f}", icon="⚠️")
        for _, row in exceeded_days.iterrows():
            log_alert(row["date"].strftime("%Y-%m-%d"), row["predicted_aqi"], alert_threshold)

    live_aqi = payload.get("live_aqi")
    live_station_name = payload.get("live_station_name")

    st.markdown('<div class="live-compare-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="live-compare-title">Live station vs model prediction</div>',
        unsafe_allow_html=True,
    )
    lc1, lc2, lc3 = st.columns(3)
    if live_aqi is not None:
        lc1.metric(
            f"Live AQI ({live_station_name or 'station'})",
            f"{live_aqi:.0f}" if isinstance(live_aqi, (int, float)) else str(live_aqi),
        )
        lc2.metric("Predicted AQI (Day 1)", f"{latest:.1f}")
        try:
            diff = float(latest) - float(live_aqi)
            lc3.metric("Difference", f"{diff:+.1f}", help="Predicted minus live.")
        except (TypeError, ValueError):
            lc3.metric("Difference", "N/A")
    else:
        lc1.warning("Live station reading unavailable.")
        lc2.metric("Predicted AQI (Day 1)", f"{latest:.1f}")
        lc3.write("")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1, 2])

    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest,
                title={"text": "Day 1 forecast AQI"},
                gauge={
                    "axis": {"range": [0, 300]},
                    "bar": {"color": "#34d399"},
                    "steps": [
                        {"range": [0, 50], "color": "#22c55e"},
                        {"range": [50, 100], "color": "#eab308"},
                        {"range": [100, 150], "color": "#f97316"},
                        {"range": [150, 200], "color": "#ef4444"},
                        {"range": [200, 300], "color": "#a855f7"},
                    ],
                    "bar": {"color": "#34d399"},
                },
            )
        )
        gauge.update_layout(
            template="plotly_dark",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e7ecef"},
        )
        st.plotly_chart(gauge, use_container_width=True)
        st.success(aqi_status(latest))

    with right:
        fig = px.line(df, x="date", y="predicted_aqi", markers=True, title="3-day AQI forecast")
        fig.update_layout(
            template="plotly_dark",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e7ecef"},
        )
        fig.update_traces(
            line=dict(width=4, color="#34d399"),
            marker=dict(size=10, color="#6ee7b7"),
            fill="tozeroy",
            fillcolor="rgba(52, 211, 153, 0.08)",
        )
        if live_aqi is not None:
            try:
                fig.add_hline(
                    y=float(live_aqi),
                    line_dash="dash",
                    line_color="#fb7185",
                    annotation_text=f"Live {float(live_aqi):.0f}",
                    annotation_position="top left",
                )
            except (TypeError, ValueError):
                pass
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3-day forecast")
    cols = st.columns(3)
    for i, day in enumerate(data):
        with cols[i]:
            glow = aqi_color(day["predicted_aqi"])
            model_used = day.get("model_used")
            model_tag = f'<span class="model-badge">{model_used}</span>' if model_used else ""
            st.markdown(
                f"""
<div class="card" style="border-top:3px solid {glow};">
<h3 style="margin:0 0 8px 0;">{str(day["date"])[:10]}</h3>
<h1 style="color:{glow}; margin:0;">{day["predicted_aqi"]:.1f}</h1>
<b>{aqi_status(day["predicted_aqi"])}</b>
{model_tag}
<hr>
Temperature {day["temperature"]:.1f} °C<br>
Humidity {day["humidity"]}%<br>
Wind {day["wind_speed"]} m/s<br>
PM2.5 {day["pm25"]}<br>
PM10 {day["pm10"]}<br>
Ozone {day["ozone"]}<br>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df, x="date", y="pm25", title="PM2.5",
            color="pm25", color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            df, x="date", y="pm10", title="PM10",
            color="pm10", color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(df, x="date", y="temperature", markers=True, title="Temperature")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#fb923c", width=3))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.line(df, x="date", y="humidity", markers=True, title="Humidity")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#6ee7b7", width=3))
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.line(df, x="date", y="wind_speed", markers=True, title="Wind speed")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#a3e635", width=3))
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.line(df, x="date", y="pressure", markers=True, title="Pressure")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#c084fc", width=3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Pollutant breakdown")
    r1, r2 = st.columns([1.1, 1])
    with r1:
        latest_row = data[0]
        radar_categories = ["PM2.5", "PM10", "Ozone", "CO", "NO₂", "SO₂"]
        radar_values = [
            latest_row["pm25"], latest_row["pm10"], latest_row["ozone"],
            latest_row["carbon_monoxide"], latest_row["nitrogen_dioxide"],
            latest_row["sulphur_dioxide"],
        ]
        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                line=dict(color="#34d399"),
                fillcolor="rgba(52, 211, 153, 0.22)",
                name="Today",
            )
        )
        radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(0,0,0,0)"),
            showlegend=False,
            height=380,
            title="Pollutant radar",
        )
        st.plotly_chart(radar, use_container_width=True)
    with r2:
        for key, label in [("pm25", "PM2.5"), ("pm10", "PM10"), ("ozone", "Ozone")]:
            val = latest_row[key]
            limit = POLLUTANT_LIMITS[key]
            pct = min(100, (val / limit) * 100)
            bar_color = "#34d399" if pct < 60 else ("#fbbf24" if pct < 100 else "#fb7185")
            st.markdown(
                f"""
            <div class="pollutant-row">
                <div class="pollutant-name"><span>{label}</span><span>{val} µg/m³ (limit {limit})</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Summary")
    a, b, c, d = st.columns(4)
    a.metric("Average AQI", f"{df['predicted_aqi'].mean():.1f}")
    b.metric("Maximum AQI", f"{df['predicted_aqi'].max():.1f}")
    c.metric("Minimum AQI", f"{df['predicted_aqi'].min():.1f}")
    d.metric("Backend average", f"{payload.get('average_aqi', df['predicted_aqi'].mean()):.1f}")

    st.markdown("---")
    st.subheader("Location")
    st.map(pd.DataFrame({"lat": [24.8050], "lon": [67.0450]}))

    st.markdown("---")
    st.subheader("Forecast table")
    styled_df = df.style.background_gradient(subset=["predicted_aqi"], cmap="RdYlGn_r").format(precision=2)
    st.dataframe(styled_df, use_container_width=True)
    st.download_button(
        "Download forecast CSV",
        df.to_csv(index=False).encode(),
        "karachi_aqi_forecast.csv",
        "text/csv",
    )

    st.markdown("---")
    st.subheader("Health guidance")
    if latest <= 50:
        st.success("Air quality is good. Outdoor activity is fine for most people.")
    elif latest <= 100:
        st.info("Acceptable air quality. Sensitive individuals should watch symptoms.")
    elif latest <= 150:
        st.warning("Sensitive groups should limit prolonged outdoor exposure.")
    else:
        st.error("Limit outdoor activity. Consider a mask if you must go outside.")

    st.markdown("---")
    st.subheader("Alert history")
    alert_log = load_alert_log()
    if alert_log.empty:
        st.info("No hazard alerts logged yet.")
    else:
        st.dataframe(alert_log.sort_values("logged_at", ascending=False), use_container_width=True, hide_index=True)
        h1, _ = st.columns([1, 3])
        h1.metric("Total alerts", len(alert_log))
        if h1.button("Clear log"):
            os.remove(ALERT_LOG_PATH)
            st.rerun()

    st.markdown("---")
    st.subheader("Model information")
    if manifest:
        st.write(
            f"""
**Day 1:** {manifest["day1"]["model_type"]} (MAE {manifest["day1"]["mae"]:.2f})  
**Day 2:** {manifest["day2"]["model_type"]} (MAE {manifest["day2"]["mae"]:.2f})  
**Day 3:** {manifest["day3"]["model_type"]} (MAE {manifest["day3"]["mae"]:.2f})
"""
        )
    else:
        st.info("No trained models found. Run `python training_pipeline.py`.")

    st.caption(
        "Target: AQI · Horizon: 3 days · Candidates: Random Forest, Ridge, PyTorch · "
        "Features: weather, pollutants, lags, rolling stats"
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center;">
        <span class="footer-chip">FastAPI</span>
        <span class="footer-chip">Streamlit</span>
        <span class="footer-chip">Scikit-Learn</span>
        <span class="footer-chip">PyTorch</span>
        <span class="footer-chip">OpenWeather</span>
        <span class="footer-chip">Plotly</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.caption("Karachi AQI · production-style monitoring dashboard")

else:
    st.error("Unable to fetch prediction. Is the FastAPI server running on port 8000?")