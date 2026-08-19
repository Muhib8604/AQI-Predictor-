import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(page_title="Karachi AQI — EDA", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { background:#080a0d; color:#e7ecef; font-family:'IBM Plex Sans',sans-serif; }
.stApp { background: radial-gradient(1100px 520px at 12% -8%, #14241c 0%, #080a0d 42%, #080a0d 100%); }
h1,h2,h3,h4 { font-family:'Outfit',sans-serif !important; color:#f8fafc !important; letter-spacing:-0.02em; }
.hero-title { font-family:'Outfit',sans-serif; font-size:2.15rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; }
[data-testid="stMetric"] {
  background:rgba(14,18,22,.88); border:1px solid rgba(52,211,153,.16);
  border-radius:14px; padding:14px 16px; box-shadow:0 8px 28px rgba(0,0,0,.28);
}
section[data-testid="stSidebar"] { background:#07090c; border-right:1px solid rgba(255,255,255,.05); }
hr { border:none; border-top:1px solid rgba(255,255,255,.06); }
.stTabs [data-baseweb="tab"] { background:rgba(20,28,34,.7); border-radius:10px 10px 0 0; color:#94a3b8; }
.stTabs [aria-selected="true"] { background:rgba(52,211,153,.15) !important; color:#6ee7b7 !important; }
.section-label {
  font-family:'Outfit',sans-serif; font-size:.72rem; font-weight:600;
  letter-spacing:.12em; text-transform:uppercase; color:#34d399; margin-bottom:.35rem;
}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,25,0.9)",
    font=dict(color="#e7ecef", family="IBM Plex Sans"),
    margin=dict(l=40, r=20, t=50, b=40),
)
EMERALD = ["#34d399", "#6ee7b7", "#a7f3d0", "#10b981", "#059669", "#fbbf24", "#fb7185"]

st.markdown('<div class="hero-title">Exploratory data analysis</div>', unsafe_allow_html=True)
st.caption("Training data deep-dive · interactive charts · Karachi AQI")

st.markdown("---")

DEFAULT_PATH = "training_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

df = None
try:
    df = load_data(DEFAULT_PATH)
    st.success(f"Loaded `{DEFAULT_PATH}` — {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    st.warning(f"`{DEFAULT_PATH}` not found. Upload manually.")
    uploaded = st.file_uploader("Upload training_dataset.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.stop()

st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
st.subheader("Dataset overview")
o1, o2, o3, o4 = st.columns(4)
o1.metric("Rows", f"{df.shape[0]:,}")
o2.metric("Columns", f"{df.shape[1]}")
o3.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
o4.metric("Numeric columns", f"{df.select_dtypes(include='number').shape[1]}")

with st.expander("Columns & dtypes"):
    st.dataframe(
        pd.DataFrame({"Column": df.dtypes.index, "Dtype": df.dtypes.values.astype(str)}),
        use_container_width=True, hide_index=True,
    )
with st.expander("Missing values"):
    missing = df.isnull().sum()
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing count"]
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    if missing.sum() == 0:
        st.info("No missing values.")
with st.expander("Preview"):
    st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Target", "Features", "Correlation", "AQI vs features", "Time & outliers",
])

with tab1:
    st.subheader("AQI distribution")
    t1, t2 = st.columns(2)
    with t1:
        fig = px.histogram(df, x="aqi", nbins=30, color_discrete_sequence=["#34d399"])
        fig.update_layout(**PLOT_LAYOUT, title="AQI histogram")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = px.box(df, y="aqi", color_discrete_sequence=["#6ee7b7"])
        fig.update_layout(**PLOT_LAYOUT, title="AQI boxplot")
        st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{df['aqi'].mean():.1f}")
    c2.metric("Median", f"{df['aqi'].median():.1f}")
    c3.metric("Std", f"{df['aqi'].std():.1f}")
    c4.metric("Range", f"{df['aqi'].min():.0f} – {df['aqi'].max():.0f}")

with tab2:
    st.subheader("Feature distributions")
    features = [f for f in ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "ozone"] if f in df.columns]
    grid = st.columns(2)
    for i, feature in enumerate(features):
        with grid[i % 2]:
            fig = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=[EMERALD[i % len(EMERALD)]])
            fig.update_layout(**PLOT_LAYOUT, title=feature)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr, color_continuous_scale=["#080a0d", "#064e3b", "#34d399", "#fbbf24"],
        aspect="auto", labels=dict(color="ρ"),
    )
    fig.update_layout(**PLOT_LAYOUT, height=640, title="Correlation matrix")
    st.plotly_chart(fig, use_container_width=True)
    if "aqi" in corr.columns:
        st.markdown("**Top correlations with AQI**")
        corr_t = corr["aqi"].drop("aqi").reindex(corr["aqi"].drop("aqi").abs().sort_values(ascending=False).index)
        st.dataframe(
            corr_t.reset_index().rename(columns={"index": "Feature", "aqi": "Correlation with AQI"}),
            use_container_width=True, hide_index=True,
        )

with tab4:
    st.subheader("AQI vs features")
    compare = [f for f in ["temperature", "humidity", "pm2_5", "pm10", "ozone"] if f in df.columns]
    grid = st.columns(2)
    for i, feature in enumerate(compare):
        with grid[i % 2]:
            fig = px.scatter(
                df, x=feature, y="aqi", opacity=0.65,
                color_discrete_sequence=[EMERALD[i % len(EMERALD)]],
            )
            fig.update_layout(**PLOT_LAYOUT, title=f"{feature} vs AQI")
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    if "date" in df.columns:
        st.subheader("AQI over time")
        df_time = df.copy()
        df_time["date"] = pd.to_datetime(df_time["date"])
        fig = px.line(df_time, x="date", y="aqi", color_discrete_sequence=["#34d399"])
        fig.update_layout(**PLOT_LAYOUT, title="AQI over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `date` column — time series skipped.")

    st.subheader("Outlier view")
    outlier_cols = [c for c in ["temperature", "humidity", "pm2_5", "pm10", "aqi"] if c in df.columns]
    melted = df[outlier_cols].melt(var_name="Feature", value_name="Value")
    fig = px.box(melted, x="Feature", y="Value", color="Feature", color_discrete_sequence=EMERALD)
    fig.update_layout(**PLOT_LAYOUT, title="Outlier detection", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("EDA · interactive Plotly · training dataset")