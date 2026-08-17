"""
pages/Custom_Prediction.py
---------------------------
Lets the user override pollutant/weather values by hand and see what the
models would predict under that hypothetical scenario. Lag/time-based
features (aqi_lag_*, rolling stats, month/day, etc.) come from TODAY's
real snapshot via feature_snapshot.build_today_features() — the user only
controls the "controllable" raw inputs (temperature, pollutants, etc.),
since the lag/rolling features can't be meaningfully hand-picked (they're
derived from real recent history).

NOTE ON CITY: only Saddar, Karachi is currently supported. The backend
(feature_pipeline.py / aqi.py / weather.py) is hard-coded to one location
right now — adding real multi-city support means a bigger change (separate
feature groups or a location column throughout the pipeline), not just a
dropdown here. The selector below is honest about that rather than faking
a switch that doesn't actually do anything.
"""

import streamlit as st

from feature_snapshot import build_today_features
from predict import predict_all_horizons

st.set_page_config(page_title="Custom AQI Prediction", page_icon="🧪", layout="wide")

st.title("🧪 Custom AQI Prediction")
st.caption("Adjust pollutant and weather values to see how the forecast changes.")

# ------------------------------------------------------------------
# Area selector — honest placeholder until multi-city is actually built
# ------------------------------------------------------------------
st.selectbox(
    "📍 Area",
    ["Saddar, Karachi"],
    disabled=True,
    help="Only one location is supported right now — the data pipeline "
         "(feature store, weather/AQICN fetch) is built for a single city. "
         "Multi-city support needs pipeline changes, not just this dropdown.",
)

st.markdown("---")

with st.spinner("Loading today's real baseline (lag/rolling features)..."):
    baseline = build_today_features()

if baseline is None:
    st.error(
        "Couldn't load today's baseline features (live weather fetch failed, or "
        "there isn't enough history in the feature store yet). Custom prediction "
        "needs this baseline for the lag/rolling inputs it can't let you set by hand."
    )
    st.stop()

st.subheader("🎛️ Adjust the controllable inputs")
st.caption("Everything else (lag history, rolling averages, date-based features) uses today's real values.")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**🌤️ Weather**")
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, float(baseline.get("temperature", 25.0)), 0.5)
    humidity = st.slider("Humidity (%)", 0, 100, int(baseline.get("humidity", 50)))
    pressure = st.slider("Pressure (hPa)", 950, 1050, int(baseline.get("pressure", 1010)))

with c2:
    st.markdown("**💨 Wind & Rain**")
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, float(baseline.get("wind_speed", 5.0)), 0.5)
    rain = st.slider("Rain (mm)", 0.0, 100.0, float(baseline.get("rain", 0.0)), 0.5)

with c3:
    st.markdown("**🌫️ Particulates**")
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, float(baseline.get("pm10", 80.0)), 1.0)
    pm2_5 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(baseline.get("pm2_5", 40.0)), 1.0)

c4, c5, c6 = st.columns(3)

with c4:
    st.markdown("**🏭 Carbon Monoxide**")
    carbon_monoxide = st.slider("CO (µg/m³)", 0.0, 3000.0, float(baseline.get("carbon_monoxide", 800.0)), 10.0)

with c5:
    st.markdown("**🚗 Nitrogen Dioxide**")
    nitrogen_dioxide = st.slider("NO₂ (µg/m³)", 0.0, 200.0, float(baseline.get("nitrogen_dioxide", 40.0)), 1.0)

with c6:
    st.markdown("**🌋 Sulphur Dioxide / Ozone**")
    sulphur_dioxide = st.slider("SO₂ (µg/m³)", 0.0, 200.0, float(baseline.get("sulphur_dioxide", 30.0)), 1.0)
    ozone = st.slider("Ozone (µg/m³)", 0.0, 300.0, float(baseline.get("ozone", 90.0)), 1.0)

st.markdown("---")

if st.button("🔮 Predict AQI for this scenario", type="primary"):

    custom_features = dict(baseline)  # start from today's real lag/rolling features
    custom_features.update({
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "rain": rain,
        "pm10": pm10,
        "pm2_5": pm2_5,
        "carbon_monoxide": carbon_monoxide,
        "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide": sulphur_dioxide,
        "ozone": ozone,
    })
    # "aqi" isn't a FEATURE_COLS entry, but predict.py reads features.get("aqi", 0.0)
    # as a leftover default for day1's now-unused prev_aqi slot — harmless either way.

    try:
        with st.spinner("Running RandomForest / Ridge / PyTorch..."):
            results = predict_all_horizons(custom_features)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Prediction complete.")

    r1, r2, r3 = st.columns(3)
    for col, horizon in zip([r1, r2, r3], ["day1", "day2", "day3"]):
        res = results[horizon]
        col.metric(
            f"{horizon.replace('day', 'Day ')}",
            f"{res['predicted_aqi']:.1f}",
            res["model_used"],
        )

    st.metric("Average (3-day)", f"{results['average_aqi']:.1f}")

    st.caption(
        "This scenario uses TODAY's real lag/rolling AQI history combined with "
        "the pollutant/weather values you set above — it does not change what "
        "actually gets saved to the feature store or the live forecast on the "
        "home page."
    )
