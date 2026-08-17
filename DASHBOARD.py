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
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# 🎨 STYLING — glassmorphism + animated gradient background
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]{
    background:#07111F;
    color:white;
    font-family:'Inter','Poppins',sans-serif;
}

/* animated gradient backdrop */
.stApp{
    background:linear-gradient(-45deg,#07111F,#0b1a2e,#091422,#0d2338);
    background-size:400% 400%;
    animation:gradientShift 18s ease infinite;
}

@keyframes gradientShift{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.main{
    background:transparent;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1,h2,h3,h4{
    color:white;
    font-family:'Poppins',sans-serif;
}

/* ---------- Hero header ---------- */
.hero-title{
    font-size:2.6rem;
    font-weight:800;
    background:linear-gradient(90deg,#22d3ee,#38bdf8,#818cf8,#22d3ee);
    background-size:300% 300%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shine 6s ease infinite;
    margin-bottom:0;
}
@keyframes shine{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.live-badge{
    display:inline-flex;
    align-items:center;
    gap:8px;
    background:rgba(34,197,94,.12);
    border:1px solid rgba(34,197,94,.4);
    color:#4ade80;
    padding:4px 14px;
    border-radius:999px;
    font-size:.8rem;
    font-weight:600;
    margin-left:14px;
    vertical-align:middle;
}
.pulse-dot{
    width:9px;height:9px;border-radius:50%;
    background:#22c55e;
    box-shadow:0 0 0 0 rgba(34,197,94,.7);
    animation:pulse 1.6s infinite;
}
@keyframes pulse{
    0%{box-shadow:0 0 0 0 rgba(34,197,94,.6);}
    70%{box-shadow:0 0 0 10px rgba(34,197,94,0);}
    100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}
}

/* ---------- Metric / glass cards ---------- */
.metric-card{
    background:rgba(30,41,59,.65);
    backdrop-filter:blur(18px);
    border-radius:18px;
    padding:20px;
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 0 25px rgba(0,255,255,.08);
    transition:.3s;
}
.metric-card:hover{
    transform:translateY(-6px);
    box-shadow:0 0 35px rgba(0,255,255,.20);
}

[data-testid="stMetric"]{
    background:rgba(30,41,59,.55);
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.08);
    border-radius:16px;
    padding:16px 18px 10px 18px;
    transition:.3s ease;
    box-shadow:0 4px 18px rgba(0,0,0,.25);
}
[data-testid="stMetric"]:hover{
    transform:translateY(-4px);
    border-color:rgba(56,189,248,.4);
    box-shadow:0 8px 26px rgba(56,189,248,.15);
}

.card{
    background:linear-gradient(160deg,#182840,#101c2e);
    border-radius:18px;
    padding:20px;
    border:1px solid #2B3A4D;
    transition:.35s ease;
    animation:fadeInUp .6s ease;
}
.card:hover{
    transform:translateY(-5px) scale(1.01);
    border-color:rgba(56,189,248,.5);
    box-shadow:0 10px 30px rgba(56,189,248,.15);
}

@keyframes fadeInUp{
    from{opacity:0; transform:translateY(14px);}
    to{opacity:1; transform:translateY(0);}
}

/* ---------- AQI scale legend ---------- */
.legend-wrap{
    display:flex;
    border-radius:12px;
    overflow:hidden;
    height:14px;
    margin-top:10px;
    box-shadow:0 0 15px rgba(0,0,0,.4);
}
.legend-seg{ flex:1; }
.legend-labels{
    display:flex;
    justify-content:space-between;
    font-size:.72rem;
    color:#9CA3AF;
    margin-top:6px;
}

/* ---------- Pollutant progress bars ---------- */
.pollutant-row{ margin-bottom:14px; }
.pollutant-name{
    display:flex;
    justify-content:space-between;
    font-size:.85rem;
    color:#cbd5e1;
    margin-bottom:4px;
}
.bar-track{
    background:rgba(255,255,255,.08);
    border-radius:8px;
    height:10px;
    overflow:hidden;
}
.bar-fill{
    height:100%;
    border-radius:8px;
    transition:width 1s ease;
}

/* ---------- Model-used badge on forecast cards ---------- */
.model-badge{
    display:inline-block;
    background:rgba(129,140,248,.15);
    border:1px solid rgba(129,140,248,.4);
    color:#c7d2fe;
    padding:2px 10px;
    border-radius:999px;
    font-size:.72rem;
    margin-top:6px;
}

/* ---------- misc ---------- */
hr{
    border:none;
    border-top:1px solid #2B3A4D;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1524,#070f1c);
    border-right:1px solid rgba(255,255,255,.06);
}

.stDownloadButton button{
    background:linear-gradient(90deg,#0ea5e9,#22d3ee);
    color:#04121b;
    font-weight:700;
    border:none;
    border-radius:10px;
    transition:.3s;
}
.stDownloadButton button:hover{
    transform:translateY(-2px);
    box-shadow:0 6px 18px rgba(34,211,238,.35);
}

/* ---------- Hazard alert banner (fires only when threshold crossed) ---------- */
.alert-banner{
    background:linear-gradient(90deg,rgba(239,68,68,.18),rgba(239,68,68,.08));
    border:1px solid rgba(239,68,68,.55);
    border-left:5px solid #ef4444;
    border-radius:14px;
    padding:16px 20px;
    margin:16px 0;
    font-size:1.02rem;
    color:#fecaca;
    animation:alertPulse 2s ease-in-out infinite;
}
@keyframes alertPulse{
    0%{box-shadow:0 0 0 0 rgba(239,68,68,.35);}
    70%{box-shadow:0 0 0 14px rgba(239,68,68,0);}
    100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}
}

/* ---------- Persistent "monitoring active" indicator (always visible) ---------- */
.monitor-badge{
    display:inline-flex;
    align-items:center;
    gap:10px;
    background:rgba(56,189,248,.08);
    border:1px solid rgba(56,189,248,.3);
    border-radius:12px;
    padding:10px 16px;
    font-size:.85rem;
    color:#93c5fd;
    margin:12px 0;
}
.monitor-dot{
    width:8px;height:8px;border-radius:50%;
    background:#38bdf8;
    box-shadow:0 0 0 0 rgba(56,189,248,.6);
    animation:pulse 1.6s infinite;
}

/* ---------- Live vs Predicted comparison card ---------- */
.live-compare-card{
    background:linear-gradient(160deg,#182840,#101c2e);
    border-radius:18px;
    padding:22px;
    border:1px solid #2B3A4D;
    margin:16px 0;
}
.live-compare-title{
    font-size:1.05rem;
    font-weight:700;
    color:#93c5fd;
    margin-bottom:6px;
}

.footer-chip{
    display:inline-block;
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.1);
    padding:4px 12px;
    border-radius:999px;
    font-size:.78rem;
    margin:3px;
    color:#93c5fd;
}

::-webkit-scrollbar{ width:10px; }
::-webkit-scrollbar-track{ background:#0a1626; }
::-webkit-scrollbar-thumb{ background:#22405e; border-radius:6px; }
::-webkit-scrollbar-thumb:hover{ background:#2f5a82; }

</style>
""", unsafe_allow_html=True)

FEATURES=[

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

"sulphur_dioxide"

]

def load_manifest():

    if not os.path.exists("model_manifest.json"):
        return None

    with open("model_manifest.json", "r") as f:
        return json.load(f)

manifest = load_manifest()

# Compute the 3-model average metrics ONCE, up top, so both the sidebar
# and the hero KPI cards can use them without recalculating.
avg_mae = avg_rmse = avg_r2 = None
if manifest:
    avg_mae = (manifest["day1"]["mae"] + manifest["day2"]["mae"] + manifest["day3"]["mae"]) / 3
    avg_rmse = (manifest["day1"]["rmse"] + manifest["day2"]["rmse"] + manifest["day3"]["rmse"]) / 3
    avg_r2 = (manifest["day1"]["r2"] + manifest["day2"]["r2"] + manifest["day3"]["r2"]) / 3

# WHO-ish reference limits used only for the visual progress bars below
POLLUTANT_LIMITS = {
    "pm25": 60,
    "pm10": 100,
    "ozone": 100,
}

def aqi_status(aqi):

    if aqi <= 50:
        return "🟢 Good"

    elif aqi <= 100:
        return "🟡 Moderate"

    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "🔴 Unhealthy"

    elif aqi <= 300:
        return "🟣 Very Unhealthy"

    return "⚫ Hazardous"

def aqi_color(aqi):
    if aqi <= 50: return "#22c55e"
    elif aqi <= 100: return "#eab308"
    elif aqi <= 150: return "#f97316"
    elif aqi <= 200: return "#ef4444"
    elif aqi <= 300: return "#a855f7"
    return "#4b5563"

# ============================================================
# 🚨 ALERT LOGGING — persists hazard events to a local CSV log
# ============================================================
ALERT_LOG_PATH = "alert_log.csv"

def log_alert(forecast_date, predicted_aqi, threshold):
    """Append a hazard alert to the local log, skipping exact duplicates
    (same forecast_date + same predicted_aqi already logged)."""
    predicted_aqi_rounded = round(float(predicted_aqi), 1)

    entry = pd.DataFrame([{
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_date": forecast_date,
        "predicted_aqi": predicted_aqi_rounded,
        "threshold": threshold,
        "status": aqi_status(predicted_aqi)
    }])

    if os.path.exists(ALERT_LOG_PATH):
        existing = pd.read_csv(ALERT_LOG_PATH)

        # Force consistent types
        if "predicted_aqi" in existing.columns:
            existing["predicted_aqi"] = existing["predicted_aqi"].astype(float).round(1)

        # Type-checker friendly way
        matching = existing[
            (existing["forecast_date"] == forecast_date) &
            (existing["predicted_aqi"] == predicted_aqi_rounded)
        ]
        if not matching.empty:
            return

        entry = pd.concat([existing, entry], ignore_index=True)

    entry.to_csv(ALERT_LOG_PATH, index=False)

def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        return pd.read_csv(ALERT_LOG_PATH)
    return pd.DataFrame(columns=["logged_at","forecast_date","predicted_aqi","threshold","status"])

# ============================================================
# 🧭 SIDEBAR
# ============================================================
st.sidebar.title("🌍 Karachi AQI Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("📍 Location")

st.sidebar.success("Saddar, Karachi")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Champion Models")

if manifest:

    st.sidebar.write(f"Day 1 : {manifest['day1']['model_type']}")
    st.sidebar.write(f"Day 2 : {manifest['day2']['model_type']}")
    st.sidebar.write(f"Day 3 : {manifest['day3']['model_type']}")

else:

    st.sidebar.write("No model found")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Performance")

if manifest:

    st.sidebar.metric("Average R²", f"{avg_r2:.3f}")
    st.sidebar.metric("Average MAE", f"{avg_mae:.2f}")
    st.sidebar.metric("Average RMSE", f"{avg_rmse:.2f}")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙ Features")

st.sidebar.write(len(FEATURES),"Input Features")

st.sidebar.markdown("---")

st.sidebar.markdown("---")

st.sidebar.subheader("🚨 Alert Settings")

alert_threshold = st.sidebar.slider(
    "Hazard alert threshold (AQI)",
    min_value=50, max_value=300, value=150, step=10,
    help="A banner + toast fire whenever any of the 3 forecast days meets or exceeds this value."
)

enable_toast_alert = st.sidebar.checkbox("🔔 Toast notification on alert", value=True)

st.sidebar.markdown("---")

# NEW: manual refresh control
if st.sidebar.button("🔄 Refresh Forecast"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(

f"Updated\n{datetime.now().strftime('%d %b %Y %H:%M')}"

)

# ============================================================
# 🏷️ HERO HEADER
# ============================================================
st.markdown(
    """
    <span class="hero-title">🌍 Karachi Air Quality Prediction Dashboard</span>
    <span class="live-badge"><span class="pulse-dot"></span> LIVE</span>
    """,
    unsafe_allow_html=True
)

st.caption(

"AI-powered 3-Day AQI Forecast using Weather Forecast + Air Pollution Forecast + Machine Learning"

)

# NEW: persistent indicator — always visible, independent of whether an
# alert is currently firing, so it's clear the hazard-monitoring system
# is active even on a calm/Good-AQI day.
st.markdown(f"""
<div class="monitor-badge">
    <span class="monitor-dot"></span>
    🛡️ Hazard monitoring active — alert threshold set to <b>{alert_threshold} AQI</b>
    (a banner + notification fire automatically if any forecast day reaches this level)
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4=st.columns(4)

k1.metric(

"📍 Location",

"Saddar",

"Karachi"

)

if manifest:
    k2.metric(
        "🤖 Champion Models",
        f"{manifest['day1']['model_type']} / {manifest['day2']['model_type']} / {manifest['day3']['model_type']}",
        "Day 1 / Day 2 / Day 3"
    )
    k3.metric(
        "🎯 Accuracy (avg)",
        f"R² = {avg_r2:.3f}",
        f"MAE {avg_mae:.2f}"
    )
else:
    k2.metric("🤖 Model", "Not trained yet", "Run training_pipeline.py")
    k3.metric("🎯 Accuracy", "N/A", "")

k4.metric(

"🕒 Updated",

datetime.now().strftime("%H:%M"),

datetime.now().strftime("%d %b")

)

# NEW: AQI color scale legend, always visible for reference
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast():
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    r = requests.get(f"{backend_url}/predict")
    return r.status_code, (r.json() if r.status_code == 200 else None)

with st.spinner("Fetching latest AQI forecast..."):
    status_code, payload = fetch_forecast()

if status_code == 200 and payload and "error" in payload:

    st.error(f"⚠️ Backend error: {payload['error']}")
    st.info(
        "This usually means the per-day models haven't been generated yet in this folder. "
        "Run `python training_pipeline.py` in your project root (the same folder as "
        "`DASHBOARD.py` and `main.py`), confirm `model_manifest.json` appears, then restart "
        "the FastAPI backend and click 🔄 Refresh Forecast in the sidebar."
    )

elif status_code == 200 and payload and "3_day_AQI_forecast" in payload:

    data = payload["3_day_AQI_forecast"]

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])

    latest = df.iloc[0]["predicted_aqi"]

    # ---------- 🚨 Hazardous AQI alert check ----------
    exceeded_days = df[df["predicted_aqi"] >= alert_threshold]

    if not exceeded_days.empty:

        worst = exceeded_days.loc[exceeded_days["predicted_aqi"].idxmax()]

        st.markdown(f"""
        <div class="alert-banner">
        🚨 <b>HAZARD ALERT</b> — {worst['date'].strftime('%d %b')} forecast AQI of
        <b>{worst['predicted_aqi']:.0f}</b> meets or exceeds your threshold of {alert_threshold}.
        Status: {aqi_status(worst['predicted_aqi'])}
        </div>
        """, unsafe_allow_html=True)

        if enable_toast_alert:
            st.toast(f"Hazardous AQI forecast: {worst['predicted_aqi']:.0f}", icon="🚨")

        for _, row in exceeded_days.iterrows():
            log_alert(row["date"].strftime("%Y-%m-%d"), row["predicted_aqi"], alert_threshold)

    # ============================================================
    # 🔴 LIVE vs 🔮 PREDICTED — compares the station's current live AQI
    # reading against today's Day 1 model prediction. live_aqi comes
    # from main.py's /predict response (via aqi.py / AQICN), independent
    # of the forecast models — if the station is down this just shows
    # "unavailable" instead of breaking the page.
    # ============================================================
    live_aqi = payload.get("live_aqi")
    live_station_name = payload.get("live_station_name")

    st.markdown('<div class="live-compare-card">', unsafe_allow_html=True)
    st.markdown('<div class="live-compare-title">📡 Live Station Reading vs Model Prediction</div>', unsafe_allow_html=True)

    lc1, lc2, lc3 = st.columns(3)

    if live_aqi is not None:
        lc1.metric(
            f"🔴 Live AQI ({live_station_name or 'station'})",
            f"{live_aqi:.0f}" if isinstance(live_aqi, (int, float)) else str(live_aqi),
        )
        lc2.metric("🔮 Predicted AQI (Day 1)", f"{latest:.1f}")
        try:
            diff = float(latest) - float(live_aqi)
            lc3.metric("Δ Difference", f"{diff:+.1f}", help="Predicted minus live. Positive = model forecasting worse air than right now.")
        except (TypeError, ValueError):
            lc3.metric("Δ Difference", "N/A")
    else:
        lc1.warning("⚠️ Live station reading unavailable right now — the configured AQICN station may be temporarily offline.")
        lc2.metric("🔮 Predicted AQI (Day 1)", f"{latest:.1f}")
        lc3.write("")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    left,right=st.columns([1,2])

    with left:

        gauge=go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=latest,

                title={"text":"Current Forecast AQI"},

                gauge={

                    "axis":{"range":[0,300]},

                    "bar":{"color":"cyan"},

                    "steps":[

                        {"range":[0,50],"color":"green"},

                        {"range":[50,100],"color":"yellow"},

                        {"range":[100,150],"color":"orange"},

                        {"range":[150,200],"color":"red"},

                        {"range":[200,300],"color":"purple"}

                    ]
                }
            )
        )

        gauge.update_layout(

            template="plotly_dark",

            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(gauge,use_container_width=True)

        st.success(aqi_status(latest))

    with right:

        fig=px.line(

            df,

            x="date",

            y="predicted_aqi",

            markers=True,

            title="3-Day AQI Forecast"

        )

        fig.update_layout(

            template="plotly_dark",

            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"

        )

        fig.update_traces(

            line=dict(width=5, color="#22d3ee"),

            marker=dict(size=11, color="#818cf8"),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.08)"

        )

        # Overlay the live reading as a horizontal reference line, if available
        if live_aqi is not None:
            try:
                fig.add_hline(
                    y=float(live_aqi),
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"Live: {float(live_aqi):.0f}",
                    annotation_position="top left",
                )
            except (TypeError, ValueError):
                pass

        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")

    st.subheader("📅 3-Day Forecast")

    cols=st.columns(3)

    for i,day in enumerate(data):

        with cols[i]:

            glow = aqi_color(day["predicted_aqi"])
            model_used = day.get("model_used")
            model_tag = f'<span class="model-badge">🤖 {model_used}</span>' if model_used else ""

            st.markdown(f"""

<div class="card" style="border-top:3px solid {glow};">

<h3>{day["date"][:10]}</h3>

<h1 style="color:{glow};">{day["predicted_aqi"]:.1f}</h1>

<b>{aqi_status(day["predicted_aqi"])}</b>

{model_tag}

<hr>

🌡 Temperature: {day["temperature"]:.1f} °C<br>

💧 Humidity: {day["humidity"]}%<br>

🌬 Wind: {day["wind_speed"]} m/s<br>

🌫 PM2.5: {day["pm25"]}<br>

🌫 PM10: {day["pm10"]}<br>

🟣 Ozone: {day["ozone"]}<br>

</div>

""",unsafe_allow_html=True)

    st.markdown("---")

    c1,c2=st.columns(2)

    with c1:

        fig=px.bar(

            df,

            x="date",

            y="pm25",

            title="PM2.5",
            color="pm25",
            color_continuous_scale=["#22c55e","#eab308","#ef4444"]

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig,use_container_width=True)

    with c2:

        fig=px.bar(

            df,

            x="date",

            y="pm10",

            title="PM10",
            color="pm10",
            color_continuous_scale=["#22c55e","#eab308","#ef4444"]

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig,use_container_width=True)

    c3,c4=st.columns(2)

    with c3:

        fig=px.line(

            df,

            x="date",

            y="temperature",

            markers=True,

            title="Temperature"

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#fb923c", width=4))

        st.plotly_chart(fig,use_container_width=True)

    with c4:

        fig=px.line(

            df,

            x="date",

            y="humidity",

            markers=True,

            title="Humidity"

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#38bdf8", width=4))

        st.plotly_chart(fig,use_container_width=True)

    c5,c6=st.columns(2)

    with c5:

        fig=px.line(

            df,

            x="date",

            y="wind_speed",

            markers=True,

            title="Wind Speed"

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#a3e635", width=4))

        st.plotly_chart(fig,use_container_width=True)

    with c6:

        fig=px.line(

            df,

            x="date",

            y="pressure",

            markers=True,

            title="Pressure"

        )

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line=dict(color="#c084fc", width=4))

        st.plotly_chart(fig,use_container_width=True)

    # NEW: Pollutant radar chart + progress bars for the latest day
    st.markdown("---")

    st.subheader("🧪 Pollutant Breakdown (Today)")

    r1, r2 = st.columns([1.1, 1])

    with r1:
        latest_row = data[0]
        radar_categories = ["PM2.5","PM10","Ozone","CO","NO₂","SO₂"]
        radar_values = [
            latest_row["pm25"], latest_row["pm10"], latest_row["ozone"],
            latest_row["carbon_monoxide"], latest_row["nitrogen_dioxide"],
            latest_row["sulphur_dioxide"]
        ]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            line=dict(color="#22d3ee"),
            fillcolor="rgba(34,211,238,.25)",
            name="Today"
        ))
        radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(0,0,0,0)"),
            showlegend=False,
            height=380,
            title="Pollutant Radar"
        )
        st.plotly_chart(radar, use_container_width=True)

    with r2:
        st.write("")
        for key, label in [("pm25","PM2.5"), ("pm10","PM10"), ("ozone","Ozone")]:
            val = latest_row[key]
            limit = POLLUTANT_LIMITS[key]
            pct = min(100, (val/limit)*100)
            bar_color = "#22c55e" if pct < 60 else ("#eab308" if pct < 100 else "#ef4444")
            st.markdown(f"""
            <div class="pollutant-row">
                <div class="pollutant-name"><span>{label}</span><span>{val} µg/m³ (limit {limit})</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Summary Statistics")

    a,b,c,d=st.columns(4)

    a.metric("Average AQI",f"{df['predicted_aqi'].mean():.1f}")

    b.metric("Maximum AQI",f"{df['predicted_aqi'].max():.1f}")

    c.metric("Minimum AQI",f"{df['predicted_aqi'].min():.1f}")

    d.metric("Backend Average AQI", f"{payload.get('average_aqi', df['predicted_aqi'].mean()):.1f}")

    st.markdown("---")

    st.subheader("🗺 Forecast Location")

    map_df=pd.DataFrame({

        "lat":[24.8598],

        "lon":[67.0099]

    })

    st.map(map_df)

    st.markdown("---")

    st.subheader("📋 Forecast Table")

    styled_df = df.style.background_gradient(
        subset=["predicted_aqi"], cmap="RdYlGn_r"
    ).format(precision=2)

    st.dataframe(

        styled_df,

        use_container_width=True

    )

    csv=df.to_csv(index=False).encode()

    st.download_button(

        "⬇ Download Forecast CSV",

        csv,

        "karachi_aqi_forecast.csv",

        "text/csv"

    )

    st.markdown("---")

    st.subheader("❤️ Health Recommendation")

    if latest<=50:

        st.success("Excellent air quality. Outdoor activities are encouraged.")

    elif latest<=100:

        st.info("Air quality is acceptable. Sensitive individuals should monitor symptoms.")

    elif latest<=150:

        st.warning("People with asthma or respiratory illness should reduce prolonged outdoor exposure.")

    else:

        st.error("Avoid unnecessary outdoor activities and wear a protective mask if outside.")

    st.markdown("---")

    st.subheader("🚨 Alert History")

    alert_log = load_alert_log()

    if alert_log.empty:
        st.info("No hazard alerts have been triggered yet — nothing has crossed your threshold.")
    else:
        st.dataframe(
            alert_log.sort_values("logged_at", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        h1, h2 = st.columns([1,3])
        h1.metric("Total Alerts Logged", len(alert_log))
        if h1.button("🗑 Clear Log"):
            os.remove(ALERT_LOG_PATH)
            st.rerun()

    st.markdown("---")

    st.subheader("🤖 Model Information")

    if manifest:
        st.write(f"""
**Day 1 Model:** {manifest["day1"]["model_type"]}  (MAE {manifest["day1"]["mae"]:.2f})

**Day 2 Model:** {manifest["day2"]["model_type"]}  (MAE {manifest["day2"]["mae"]:.2f})

**Day 3 Model:** {manifest["day3"]["model_type"]}  (MAE {manifest["day3"]["mae"]:.2f})
""")
    else:
        st.info("No trained models found yet — run `python training_pipeline.py` to generate `model_manifest.json`.")

    st.write("""

**Target:** AQI Prediction

**Forecast Horizon:** 3 Days (one independently-trained model per day)

**Candidate Algorithms:** Random Forest, Ridge Regression, PyTorch (best of the 3 kept per day)

**Features Used**

- Temperature

- Humidity

- Pressure

- Wind Speed

- Rain

- PM2.5

- PM10

- Ozone

- Carbon Monoxide

- Nitrogen Dioxide

- Sulphur Dioxide

- Historical lag features

- Rolling AQI averages

- AQI change rate

""")

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;">
        <span class="footer-chip">⚡ FastAPI</span>
        <span class="footer-chip">🎈 Streamlit</span>
        <span class="footer-chip">🌲 Scikit-Learn</span>
        <span class="footer-chip">🔥 PyTorch</span>
        <span class="footer-chip">☁ OpenWeather API</span>
        <span class="footer-chip">📊 Plotly</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption(

        "Built with ❤️ using FastAPI • Streamlit • Scikit-Learn • PyTorch • OpenWeather API"

    )

else:

    st.error("Unable to fetch prediction.")
