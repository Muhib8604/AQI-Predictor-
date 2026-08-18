"""
pages/Model_Comparison.py
--------------------------
Shows MAE / RMSE / R2 for ALL THREE candidate models (RandomForest, Ridge,
PyTorch) across all 3 forecast horizons — not just whichever one won.

Reads model_comparison.json, which training_pipeline.py now saves
alongside model_manifest.json (added in this session — rerun
training_pipeline.py once after updating it for this file to have data).

If you had an older Model Comparison page showing ARIMA — that model
isn't part of this project's actual tournament (RandomForest/Ridge/
PyTorch only), so it isn't reproduced here. If ARIMA numbers matter to
you, share whatever generated them and it can be added back properly.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Comparison")
st.caption("RandomForest vs Ridge vs PyTorch — cross-validated performance per forecast horizon.")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


comparison = load_json("model_comparison.json")
manifest = load_json("model_manifest.json")

if comparison is None:
    st.warning(
        "`model_comparison.json` not found yet. This file is created by the "
        "updated `training_pipeline.py` — rerun `python training_pipeline.py` "
        "once to generate it (it's saved alongside `model_manifest.json`)."
    )
    st.stop()

# ------------------------------------------------------------------
# Build one long-format DataFrame across all horizons/models for plotting
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Champion summary strip
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# MAE comparison — lower is better
# ------------------------------------------------------------------
st.subheader("📉 MAE by Model (lower is better)")
fig_mae = px.bar(
    df, x="Horizon", y="MAE", color="Model", barmode="group",
    text_auto=".2f",
)
fig_mae.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_mae, use_container_width=True)

# ------------------------------------------------------------------
# R2 comparison — higher is better
# ------------------------------------------------------------------
st.subheader("📈 R² by Model (higher is better)")
fig_r2 = px.bar(
    df, x="Horizon", y="R2", color="Model", barmode="group",
    text_auto=".3f",
)
fig_r2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_r2, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------
# Full table, champion row highlighted
# ------------------------------------------------------------------
st.subheader("📋 Full Comparison Table")


def highlight_champion(row):
    return ["background-color: rgba(34,197,94,.15)" if row["Champion"] else "" for _ in row]


display_df = df[["Horizon", "Model", "MAE", "RMSE", "R2", "Champion"]].sort_values(["Horizon", "MAE"])
st.dataframe(
    display_df.style.apply(highlight_champion, axis=1).format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Metrics are pooled cross-validated MAE/RMSE/R² on reconstructed absolute AQI "
    "(Day 2/3 use the upstream model's own out-of-fold predictions to reconstruct "
    "the level — never ground truth — so these numbers reflect real deployed accuracy)."
)
