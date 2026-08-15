import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
import matplotlib.pyplot as plt

# Make sibling project modules importable (weather.py, history.py, feature_snapshot.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

st.set_page_config(
    page_title="Karachi AQI — Explainability",
    page_icon="🧠",
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
.info-banner{
    background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.35);
    border-radius:12px; padding:12px 16px; font-size:.85rem; color:#bae6fd;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="hero-title">🧠 Model Explainability (SHAP)</span>', unsafe_allow_html=True)
st.caption("Which features drive each day's AQI prediction — pick a horizon below")

st.markdown("""
<div class="info-banner">
ℹ️ You now have <b>3 separate models</b> (one per forecast day), and the champion algorithm can be
different for each — pick a day below to explain that specific model. This page reconstructs today's
exact 20-feature input the same way <code>main.py</code> does (via <code>feature_snapshot.py</code>,
shared by both), so it can never drift out of sync with what's actually being served again.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 📦 LOAD MANIFEST + FEATURE SCHEMA
# ============================================================
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "model_manifest.json")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "model_features.pkl")

if not os.path.exists(MANIFEST_PATH) or not os.path.exists(FEATURES_PATH):
    st.error(
        "No trained per-day models found yet. Run `python training_pipeline.py` in your "
        "project root, then reload this page."
    )
    st.stop()

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

feature_names = joblib.load(FEATURES_PATH)

with st.expander("📋 Exact feature schema used by every model"):
    st.code(", ".join(feature_names))

# ============================================================
# 🗓️ HORIZON SELECTOR
# ============================================================
horizon_labels = {"day1": "Day 1 (tomorrow)", "day2": "Day 2", "day3": "Day 3"}
selected_horizon = st.selectbox(
    "Which day's model do you want to explain?",
    options=["day1", "day2", "day3"],
    format_func=lambda h: f"{horizon_labels[h]} — {manifest[h]['model_type']} (MAE {manifest[h]['mae']:.2f})"
)

info = manifest[selected_horizon]
st.success(f"Explaining **{horizon_labels[selected_horizon]}**: `{info['model_type']}` model (MAE={info['mae']:.2f}, R²={info['r2']:.3f})")

# ============================================================
# 🔧 LOAD THIS HORIZON'S MODEL + BUILD A UNIFIED predict_fn
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_predict_fn(info):
    model_path = os.path.join(PROJECT_ROOT, info["file"])

    if info["kind"] == "sklearn":
        model = joblib.load(model_path)

        def predict_fn(X):
            return model.predict(pd.DataFrame(X, columns=feature_names))

        return model, predict_fn

    else:  # PyTorch

     import torch
    from model_definition import AQINet

    scaler = joblib.load(
        os.path.join(PROJECT_ROOT, info["scaler_file"])
    )

    model = AQINet(len(feature_names))

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    def predict_fn(X):

        X_df = pd.DataFrame(X, columns=feature_names)

        scaled = scaler.transform(X_df)

        with torch.no_grad():

            prediction = model(
                torch.tensor(
                    scaled,
                    dtype=torch.float32
                )
            ).numpy().flatten()

        return (
            prediction * info["target_std"]
        ) + info["target_mean"]

    return model, predict_fn

model, predict_fn = load_model_and_predict_fn(info)

# ============================================================
# 📂 BACKGROUND DATA — rebuild the SAME engineered features
# training_pipeline.py used, straight from training_dataset.csv
# ============================================================
@st.cache_data(show_spinner=False)
def load_background():

    sys.path.insert(0, PROJECT_ROOT)

    from training_pipeline import (
        connect_feature_store,
        prepare_clean_dataset,
        FEATURE_COLS,
    )

    # Read directly from Hopsworks
    fs = connect_feature_store()

    feature_group = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    raw = feature_group.read()

    daily = prepare_clean_dataset(raw)

    return daily[FEATURE_COLS]

try:
    X_bg = load_background()
except Exception as e:

    st.error(f"Couldn't load background data from Hopsworks.\n\n{e}")

    st.stop()

sample_size = min(150, len(X_bg))
X_sample = X_bg.sample(sample_size, random_state=42) if len(X_bg) > sample_size else X_bg
background_for_explainer = X_sample.sample(min(50, len(X_sample)), random_state=1)

st.markdown("---")
st.subheader("🌐 Global Feature Importance")

try:
    import shap
except ImportError:
    st.error("The `shap` package isn't installed. Run `pip install shap` and reload this page.")
    st.stop()

@st.cache_resource(show_spinner="Computing SHAP values (this can take ~10-30s for non-tree models)...")
def compute_shap_values( model_kind, X_sample_df, background_df):
    # Random Forest gets the fast, exact TreeExplainer. Ridge (wrapped in a
    # scaling Pipeline) and PyTorch aren't tree models, so they use the
    # generic permutation-based Explainer via the predict_fn wrapper above —
    # slower, but works correctly for any model type.
    from sklearn.ensemble import RandomForestRegressor

    is_random_forest = model_kind == "sklearn" and isinstance(model, RandomForestRegressor)

    if is_random_forest:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(predict_fn, background_df)

    return explainer(X_sample_df)

sv = compute_shap_values (info["kind"], X_sample, background_for_explainer)

col_a, col_b = st.columns(2)
with col_a:
    shap.summary_plot(sv.values, X_sample, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")
with col_b:
    shap.summary_plot(sv.values, X_sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")

mean_abs_shap = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean |SHAP value|": np.abs(sv.values).mean(axis=0)
}).sort_values("Mean |SHAP value|", ascending=False)
st.dataframe(mean_abs_shap, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# 🎯 LOCAL EXPLANATION — today's live forecast for this horizon
# ============================================================
st.subheader(f"🎯 Why This Prediction? (Today's Live {horizon_labels[selected_horizon]} Forecast)")

try:
    from feature_snapshot import build_today_features
    today_features = build_today_features()
except Exception as e:
    today_features = None
    st.warning(f"Couldn't fetch live weather data: {e}")

if today_features is not None:
    try:
        input_row = pd.DataFrame([today_features])[feature_names]
        predicted_today = float(predict_fn(input_row)[0])
        st.metric(f"Model's Predicted AQI ({horizon_labels[selected_horizon]})", f"{predicted_today:.1f}")

        with st.spinner("Computing local explanation..."):
            local_sv = compute_shap_values( info["kind"], input_row, background_for_explainer)

        fig = plt.figure(figsize=(9, 5))
        try:
            import shap as _shap
            exp = _shap.Explanation(
                values=local_sv.values[0],
                base_values=local_sv.base_values[0] if hasattr(local_sv.base_values, "__len__") else local_sv.base_values,
                data=input_row.iloc[0].values,
                feature_names=feature_names,
            )
            shap.plots.waterfall(exp, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
        except Exception as e:
            st.warning(f"Waterfall plot failed: {e}")
        plt.close("all")

    except Exception as e:
        st.warning(f"Couldn't reconstruct today's live input row: {e}")
else:
    st.info("No live explanation available right now — the global SHAP analysis above still reflects the model faithfully.")

st.markdown("---")
st.caption(
    "🧠 Explainability powered by SHAP · Random Forest uses TreeExplainer (fast, exact) · "
    "Ridge/PyTorch use a permutation-based Explainer via the model's predict function · "
    "Global importance from training_dataset.csv · Local explanation from feature_snapshot.py (shared with main.py)"
)
