"""
pages/Model_Comparison.py
UI polish only — same data contract with model_comparison.json / model_manifest.json.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="◈", layout="wide")

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
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

/* TOP SPACE — title uper se cut na ho */
.block-container {
    padding-top: 5.5rem !important;
    overflow: visible !important;
}

.page-title {
    font-family: 'Outfit', sans-serif;
    margin-top: 1.5rem !important;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    line-height: 1.45;
    padding-top: 0.35rem;
    padding-bottom: 0.25rem;
    overflow: visible !important;
    display: block;
}
.page-sub { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1rem; }
.section-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #34d399;
    margin: 0.2rem 0 0.5rem;
}
[data-testid="stMetric"] {
    background: rgba(14, 18, 22, 0.88);
    border: 1px solid rgba(52, 211, 153, 0.16);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(52, 211, 153, 0.4);
    box-shadow: 0 0 28px rgba(52, 211, 153, 0.12);
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] {
    background: #07090c;
    border-right: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📈 Model comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">RandomForest vs Ridge vs PyTorch — cross-validated metrics per horizon</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">📈 Model comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">RandomForest vs Ridge vs PyTorch — cross-validated metrics per horizon</div>',
    unsafe_allow_html=True,
)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


comparison = load_json("model_comparison.json")
manifest = load_json("model_manifest.json")

if comparison is None:
    st.warning(
        "`model_comparison.json` not found. Rerun `python training_pipeline.py` once to generate it "
        "(saved alongside `model_manifest.json`)."
    )
    st.stop()

rows = []
for horizon_name, horizon_data in comparison.items():
    champion = horizon_data.get("champion")
    for model_name, metrics in horizon_data.get("candidates", {}).items():
        rows.append({
            "Horizon": horizon_name.replace("day", "Day "),
            "Model": model_name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
            "Champion": model_name == champion,
        })

df = pd.DataFrame(rows)

st.markdown('<div class="section-label">Champions</div>', unsafe_allow_html=True)
st.subheader("🏆 Champions")
cols = st.columns(3)
for col, horizon_name in zip(cols, ["day1", "day2", "day3"]):
    horizon_data = comparison.get(horizon_name, {})
    champ = horizon_data.get("champion", "—")
    champ_metrics = horizon_data.get("candidates", {}).get(champ, {})
    col.metric(
        horizon_name.replace("day", "Day "),
        champ,
        f"MAE {champ_metrics.get('mae', float('nan')):.2f}" if champ_metrics else "",
    )

st.markdown("---")
st.markdown('<div class="section-label">MAE</div>', unsafe_allow_html=True)
st.subheader("📉 MAE by model (lower is better)")
fig_mae = px.bar(
    df, x="Horizon", y="MAE", color="Model", barmode="group", text_auto=".2f",
    color_discrete_sequence=["#34d399", "#6ee7b7", "#a7f3d0"],
)
fig_mae.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e7ecef"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_mae, use_container_width=True)

st.markdown('<div class="section-label">R²</div>', unsafe_allow_html=True)
st.subheader("📈 R² by model (higher is better)")
fig_r2 = px.bar(
    df, x="Horizon", y="R2", color="Model", barmode="group", text_auto=".3f",
    color_discrete_sequence=["#34d399", "#6ee7b7", "#a7f3d0"],
)
fig_r2.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e7ecef"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_r2, use_container_width=True)

st.markdown("---")
st.markdown('<div class="section-label">Table</div>', unsafe_allow_html=True)
st.subheader("📋 Full comparison table")


def highlight_champion(row):
    return [
        "background-color: rgba(52, 211, 153, 0.15)" if row["Champion"] else ""
        for _ in row
    ]


display_df = df[["Horizon", "Model", "MAE", "RMSE", "R2", "Champion"]].sort_values(["Horizon", "MAE"])
st.dataframe(
    display_df.style.apply(highlight_champion, axis=1).format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Pooled cross-validated MAE / RMSE / R² on reconstructed absolute AQI. "
    "Day 2/3 use upstream out-of-fold predictions for reconstruction — not ground truth."
)
