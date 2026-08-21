import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#080a0d",
    "axes.facecolor": "#0f1419",
    "text.color": "#e7ecef",
    "axes.labelcolor": "#e7ecef",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

st.set_page_config(
    page_title="Karachi AQI — Explainability",
    page_icon="◈",
    layout="wide",
)

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
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}
.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: clamp(1.5rem, 3.2vw, 2.05rem) !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
    line-height: 1.5 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    display: inline-block !important;
    max-width: 100%;
    padding-top: 6px !important;
    padding-bottom: 8px !important;
    /* agar gradient text use ho raha ho to clip avoid */
    -webkit-background-clip: unset !important;
    background-clip: unset !important;
    -webkit-text-fill-color: #f8fafc !important;
}

[data-testid="stMetric"] {
    background: rgba(14, 18, 22, 0.88);
    border: 1px solid rgba(52, 211, 153, 0.16);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(52, 211, 153, 0.4);
    box-shadow: 0 0 28px rgba(52, 211, 153, 0.12);
}
section[data-testid="stSidebar"] {
    background: #07090c;
    border-right: 1px solid rgba(255,255,255,0.05);
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); }
.info-banner {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.28);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #a7f3d0;
}
.section-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #34d399;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔬 Model explainability</div>', unsafe_allow_html=True)
st.caption("SHAP · which features drive each day's AQI prediction")

st.markdown("""
<div class="info-banner">
You have <b>3 separate models</b> (one per forecast day). Pick a day to explain that model.
This page rebuilds today's input via <code>feature_snapshot.py</code> — the same path as the live API —
so explanations stay aligned with what is served.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "model_manifest.json")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "model_features.pkl")

if not os.path.exists(MANIFEST_PATH) or not os.path.exists(FEATURES_PATH):
    st.error(
        "No trained per-day models found. Run `python training_pipeline.py` in the project root, then reload."
    )
    st.stop()

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

feature_names = joblib.load(FEATURES_PATH)

with st.expander("Feature schema used by every model"):
    st.code(", ".join(feature_names))

st.markdown('<div class="section-label">Horizon</div>', unsafe_allow_html=True)
horizon_labels = {"day1": "Day 1 (tomorrow)", "day2": "Day 2", "day3": "Day 3"}
selected_horizon = st.selectbox(
    "Which day's model do you want to explain?",
    options=["day1", "day2", "day3"],
    format_func=lambda h: f"{horizon_labels[h]} — {manifest[h]['model_type']} (MAE {manifest[h]['mae']:.2f})",
)

info = manifest[selected_horizon]
st.success(
    f"Explaining **{horizon_labels[selected_horizon]}**: `{info['model_type']}` "
    f"(MAE={info['mae']:.2f}, R²={info['r2']:.3f})"
)


@st.cache_resource(show_spinner=False)
def load_model_and_predict_fn(info):
    model_path = os.path.join(PROJECT_ROOT, info["file"])

    if info["kind"] == "sklearn":
        model = joblib.load(model_path)

        def predict_fn(X):
            return model.predict(pd.DataFrame(X, columns=feature_names))

        return model, predict_fn

    else:
        import torch
        from model_definition import AQINet

        scaler = joblib.load(os.path.join(PROJECT_ROOT, info["scaler_file"]))
        model = AQINet(len(feature_names))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        def predict_fn(X):
            X_df = pd.DataFrame(X, columns=feature_names)
            scaled = scaler.transform(X_df)
            with torch.no_grad():
                prediction = model(
                    torch.tensor(scaled, dtype=torch.float32)
                ).numpy().flatten()
            return (prediction * info["target_std"]) + info["target_mean"]

        return model, predict_fn


model, predict_fn = load_model_and_predict_fn(info)


@st.cache_data(show_spinner=False)
def load_background():
    sys.path.insert(0, PROJECT_ROOT)
    from training_pipeline import prepare_clean_dataset, FEATURE_COLS
    from feature_store import connect_feature_store

    fs = connect_feature_store()
    feature_group = fs.get_feature_group(name="aqi_features", version=2)
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
st.markdown('<div class="section-label">Global importance</div>', unsafe_allow_html=True)
st.subheader("🌐 Global feature importance")

try:
    import shap
except ImportError:
    st.error("`shap` is not installed. Run `pip install shap` and reload.")
    st.stop()


@st.cache_resource(show_spinner="Computing SHAP values…")
def compute_shap_values(model_kind, X_sample_df, background_df):
    from sklearn.ensemble import RandomForestRegressor

    is_random_forest = model_kind == "sklearn" and isinstance(model, RandomForestRegressor)

    if is_random_forest:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(predict_fn, background_df)

    return explainer(X_sample_df)


sv = compute_shap_values(info["kind"], X_sample, background_for_explainer)

col_a, col_b = st.columns(2)
with col_a:
    shap.summary_plot(sv.values, X_sample, plot_type="bar", show=False)
    for ax in plt.gcf().axes:
        for patch in ax.patches:
            patch.set_facecolor("#34d399")
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")
with col_b:
    shap.summary_plot(sv.values, X_sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")

mean_abs_shap = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean |SHAP value|": np.abs(sv.values).mean(axis=0),
}).sort_values("Mean |SHAP value|", ascending=False)

st.dataframe(mean_abs_shap, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<div class="section-label">Local explanation</div>', unsafe_allow_html=True)
st.subheader(f"🧭 Why this prediction? · {horizon_labels[selected_horizon]}")

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
        st.metric(
            f"Predicted AQI ({horizon_labels[selected_horizon]})",
            f"{predicted_today:.1f}",
        )

        with st.spinner("Computing local explanation…"):
            local_sv = compute_shap_values(info["kind"], input_row, background_for_explainer)

        plt.figure(figsize=(9, 5))
        try:
            exp = shap.Explanation(
                values=local_sv.values[0],
                base_values=(
                    local_sv.base_values[0]
                    if hasattr(local_sv.base_values, "__len__")
                    else local_sv.base_values
                ),
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
    st.info("No live explanation available — global SHAP above still reflects the model.")

st.markdown("---")
st.caption(
    "SHAP · Random Forest uses TreeExplainer · Ridge/PyTorch use permutation Explainer · "
    "Global from feature store · Local from feature_snapshot (shared with API)"
)
