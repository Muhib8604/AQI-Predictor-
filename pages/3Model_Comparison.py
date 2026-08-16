import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Karachi AQI — Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# 🎨 STYLING — same theme as the main dashboard
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"]{ background:#07111F; color:white; font-family:'Inter','Poppins',sans-serif; }
.stApp{
    background:linear-gradient(-45deg,#07111F,#0b1a2e,#091422,#0d2338);
    background-size:400% 400%; animation:gradientShift 18s ease infinite;
}
@keyframes gradientShift{ 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
.block-container{ padding-top:2rem; padding-bottom:2rem; }
h1,h2,h3,h4{ color:white; font-family:'Poppins',sans-serif; }
.hero-title{
    font-size:2.3rem; font-weight:800;
    background:linear-gradient(90deg,#22d3ee,#38bdf8,#818cf8,#22d3ee);
    background-size:300% 300%; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:shine 6s ease infinite;
}
@keyframes shine{ 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
[data-testid="stMetric"]{
    background:rgba(30,41,59,.55); backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.08); border-radius:16px;
    padding:16px 18px 10px 18px; transition:.3s ease; box-shadow:0 4px 18px rgba(0,0,0,.25);
}
[data-testid="stMetric"]:hover{
    transform:translateY(-4px); border-color:rgba(56,189,248,.4); box-shadow:0 8px 26px rgba(56,189,248,.15);
}
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b1524,#070f1c); border-right:1px solid rgba(255,255,255,.06); }
hr{ border:none; border-top:1px solid #2B3A4D; }
.badge{
    display:inline-block; background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.4);
    color:#38bdf8; padding:3px 12px; border-radius:999px; font-size:.78rem; margin-right:6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="hero-title">🤖 Multi-Model Forecast Comparison</span>', unsafe_allow_html=True)
st.caption("Statistical → Machine Learning → Deep Learning, benchmarked on held-out historical data")
st.markdown("""
<span class="badge">📉 ARIMA (Statistical)</span>
<span class="badge">🌲 Random Forest (ML)</span>
<span class="badge">🧠 LSTM (Deep Learning)</span>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 📂 LOAD DATA
# ============================================================
DEFAULT_PATH = "training_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data(path):
    d = pd.read_csv(path)
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)
    return d

df = None
try:
    df = load_data(DEFAULT_PATH)
except FileNotFoundError:
    st.warning(f"Couldn't find `{DEFAULT_PATH}` next to the app. Upload it below.")
    uploaded = st.file_uploader("Upload training_dataset.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

if df is None:
    st.stop()

if "aqi" not in df.columns:
    st.error("This page needs an `aqi` column in the dataset to backtest forecasting models.")
    st.stop()

if "date" not in df.columns:
    st.warning("No `date` column found — using row order as the time axis instead.")
    df["date"] = pd.RangeIndex(len(df))

# ============================================================
# ⚙️ SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("🤖 Model Comparison Settings")
st.sidebar.markdown("---")

test_size = st.sidebar.slider("Backtest window (days held out)", min_value=7, max_value=60, value=14, step=1)
lstm_window = st.sidebar.slider("LSTM lookback window (days)", min_value=3, max_value=21, value=7, step=1)
lstm_epochs = st.sidebar.slider("LSTM training epochs", min_value=5, max_value=100, value=25, step=5)

feature_candidates = [c for c in [
    "temperature", "humidity", "wind_speed", "pressure", "rain",
    "pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"
] if c in df.columns]

st.sidebar.markdown("---")
ml_features = st.sidebar.multiselect(
    "Features for the ML model",
    options=[c for c in df.select_dtypes(include="number").columns if c != "aqi"],
    default=feature_candidates
)

if len(df) <= test_size + 10:
    st.error("Not enough rows in the dataset for this backtest window — reduce it in the sidebar.")
    st.stop()

train_df = df.iloc[:-test_size].reset_index(drop=True)
test_df = df.iloc[-test_size:].reset_index(drop=True)

results = {}   # model_name -> {"pred": array, "mae":..., "rmse":...}

st.markdown("---")
st.subheader("📊 Backtest: Actual vs Predicted (last {} days)".format(test_size))

# ============================================================
# 📉 STATISTICAL MODEL — ARIMA
# ============================================================
with st.spinner("Fitting ARIMA (statistical baseline)..."):
    try:
        from statsmodels.tsa.arima.model import ARIMA

        arima_model = ARIMA(train_df["aqi"].values, order=(2, 1, 2)).fit()
        arima_pred = arima_model.forecast(steps=test_size)

        results["ARIMA (Statistical)"] = {
            "pred": np.array(arima_pred),
            "mae": mean_absolute_error(test_df["aqi"], arima_pred),
            "rmse": np.sqrt(mean_squared_error(test_df["aqi"], arima_pred))
        }
    except Exception as e:
        st.warning(f"ARIMA failed to fit on this data: {e}")

# ============================================================
# 🌲 MACHINE LEARNING MODEL — Random Forest
# ============================================================
if len(ml_features) > 0:
    with st.spinner("Training Random Forest (ML baseline)..."):
        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(train_df[ml_features], train_df["aqi"])
            rf_pred = rf.predict(test_df[ml_features])

            results["Random Forest (ML)"] = {
                "pred": rf_pred,
                "mae": mean_absolute_error(test_df["aqi"], rf_pred),
                "rmse": np.sqrt(mean_squared_error(test_df["aqi"], rf_pred))
            }
        except Exception as e:
            st.warning(f"Random Forest failed: {e}")
else:
    st.info("Select at least one feature in the sidebar to include the Random Forest model.")

# ============================================================
# 🧠 DEEP LEARNING MODEL — LSTM
# ============================================================
tf_available = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf_available = False

if tf_available:
    with st.spinner("Training LSTM (deep learning baseline)..."):
        try:
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_df[["aqi"]].values)

            def make_sequences(series, window):
                Xs, ys = [], []
                for i in range(len(series) - window):
                    Xs.append(series[i:i+window])
                    ys.append(series[i+window])
                return np.array(Xs), np.array(ys)

            X_seq, y_seq = make_sequences(train_scaled, lstm_window)

            if len(X_seq) < 10:
                st.warning("Not enough training rows for the chosen LSTM window — reduce the lookback window.")
            else:
                lstm = keras.Sequential([
                    layers.Input(shape=(lstm_window, 1)),
                    layers.LSTM(32),
                    layers.Dense(16, activation="relu"),
                    layers.Dense(1)
                ])
                lstm.compile(optimizer="adam", loss="mse")
                lstm.fit(X_seq, y_seq, epochs=lstm_epochs, verbose=0, batch_size=16)

                # Recursive forecast for the test window
                history = list(train_scaled[-lstm_window:].flatten())
                lstm_preds_scaled = []
                for _ in range(test_size):
                    window_input = np.array(history[-lstm_window:]).reshape(1, lstm_window, 1)
                    next_val = lstm.predict(window_input, verbose=0)[0, 0]
                    lstm_preds_scaled.append(next_val)
                    history.append(next_val)

                lstm_pred = scaler.inverse_transform(
                    np.array(lstm_preds_scaled).reshape(-1, 1)
                ).flatten()

                results["LSTM (Deep Learning)"] = {
                    "pred": lstm_pred,
                    "mae": mean_absolute_error(test_df["aqi"], lstm_pred),
                    "rmse": np.sqrt(mean_squared_error(test_df["aqi"], lstm_pred))
                }
        except Exception as e:
            st.warning(f"LSTM training failed: {e}")
else:
    st.info(
        "TensorFlow isn't installed, so the deep learning model is skipped. "
        "Run `pip install tensorflow` and reload this page to enable the LSTM baseline."
    )

# ============================================================
# 📈 RESULTS
# ============================================================
if not results:
    st.error("No models could be fit on this data. Check your dataset and feature selection.")
    st.stop()

metrics_df = pd.DataFrame([
    {"Model": name, "MAE": r["mae"], "RMSE": r["rmse"]}
    for name, r in results.items()
]).sort_values("MAE")

m_cols = st.columns(len(results) + 1)
m_cols[0].metric("Best Model", metrics_df.iloc[0]["Model"], f"MAE {metrics_df.iloc[0]['MAE']:.2f}")
for i, (_, row) in enumerate(metrics_df.iterrows()):
    m_cols[i+1].metric(row["Model"], f"MAE {row['MAE']:.2f}", f"RMSE {row['RMSE']:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test_df["date"], y=test_df["aqi"], mode="lines+markers",
    name="Actual", line=dict(color="white", width=3, dash="dot")
))

palette = ["#22d3ee", "#f97316", "#a78bfa"]
for i, (name, r) in enumerate(results.items()):
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=r["pred"], mode="lines+markers",
        name=name, line=dict(color=palette[i % len(palette)], width=3)
    ))

fig.update_layout(
    template="plotly_dark", height=450,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title="Backtest — Actual vs Predicted AQI",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}), use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# 🔮 FORWARD 3-DAY FORECAST COMPARISON
# ============================================================
st.subheader("🔮 Next 3-Day Forecast — Model by Model")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_forecast():
    r = requests.get("http://127.0.0.1:8000/predict")
    return r.status_code, (r.json() if r.status_code == 200 else None)

forward_results = {}

# ARIMA forward forecast (refit on full series)
try:
    from statsmodels.tsa.arima.model import ARIMA
    arima_full = ARIMA(df["aqi"].values, order=(2, 1, 2)).fit()
    forward_results["ARIMA (Statistical)"] = arima_full.forecast(steps=3)
except Exception:
    pass

# Random Forest forward forecast (uses live API's forecast weather/pollutant features)
status_code, payload = fetch_live_forecast()
if status_code == 200 and len(ml_features) > 0:
    try:
        future_days = payload["3_day_AQI_forecast"]
        future_X = pd.DataFrame(future_days)
        if "pm25" in future_X.columns and "pm2_5" not in future_X.columns:
            future_X = future_X.rename(columns={"pm25": "pm2_5"})
        missing = [c for c in ml_features if c not in future_X.columns]
        for c in missing:
            future_X[c] = df[c].mean()
        rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_full.fit(df[ml_features], df["aqi"])
        forward_results["Random Forest (ML)"] = rf_full.predict(future_X[ml_features])
    except Exception:
        pass
elif status_code != 200:
    st.info("Live `/predict` endpoint unreachable — Random Forest forward forecast skipped (backtest results above are unaffected).")

# LSTM forward forecast
if tf_available and "LSTM (Deep Learning)" in results:
    try:
        scaler_full = MinMaxScaler()
        full_scaled = scaler_full.fit_transform(df[["aqi"]].values)
        history = list(full_scaled[-lstm_window:].flatten())
        preds_scaled = []
        for _ in range(3):
            window_input = np.array(history[-lstm_window:]).reshape(1, lstm_window, 1)
            next_val = lstm.predict(window_input, verbose=0)[0, 0]
            preds_scaled.append(next_val)
            history.append(next_val)
        forward_results["LSTM (Deep Learning)"] = scaler_full.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
    except Exception:
        pass

if forward_results:
    horizon_labels = ["Day 1", "Day 2", "Day 3"]
    fig2 = go.Figure()
    for i, (name, preds) in enumerate(forward_results.items()):
        fig2.add_trace(go.Bar(x=horizon_labels, y=preds, name=name, marker_color=palette[i % len(palette)]))
    fig2.update_layout(
        template="plotly_dark", barmode="group", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Next 3-Day AQI Forecast by Model"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No forward forecasts could be generated — check the live API connection and feature selection above.")

st.markdown("---")
st.caption(
    "📌 Note: Random Forest and LSTM here are trained live in this page for fair comparison — "
    "they're independent benchmarks, not necessarily identical to your production model. "
    "ARIMA uses only historical AQI values (no weather/pollutant inputs)."
)
