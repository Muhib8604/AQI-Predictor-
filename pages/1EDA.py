import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Karachi AQI — EDA",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 🎨 STYLING — same theme as the main dashboard
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]{
    background:#07111F;
    color:white;
    font-family:'Inter','Poppins',sans-serif;
}

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

.block-container{ padding-top:2rem; padding-bottom:2rem; }

h1,h2,h3,h4{ color:white; font-family:'Poppins',sans-serif; }

.hero-title{
    font-size:2.3rem;
    font-weight:800;
    background:linear-gradient(90deg,#22d3ee,#38bdf8,#818cf8,#22d3ee);
    background-size:300% 300%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shine 6s ease infinite;
}
@keyframes shine{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
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
    padding:18px;
    border:1px solid #2B3A4D;
    margin-bottom:10px;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1524,#070f1c);
    border-right:1px solid rgba(255,255,255,.06);
}

hr{ border:none; border-top:1px solid #2B3A4D; }

.stTabs [data-baseweb="tab-list"]{
    gap:6px;
}
.stTabs [data-baseweb="tab"]{
    background:rgba(30,41,59,.5);
    border-radius:10px 10px 0 0;
    padding:8px 16px;
    color:#cbd5e1;
}
.stTabs [aria-selected="true"]{
    background:rgba(56,189,248,.18) !important;
    color:#38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 🌙 Matplotlib / Seaborn dark theme to match dashboard
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "#0b1524",
    "axes.facecolor": "#0b1524",
    "savefig.facecolor": "#0b1524",
    "axes.edgecolor": "#2B3A4D",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "#9CA3AF",
    "ytick.color": "#9CA3AF",
    "grid.color": "#1f2a3d",
    "axes.titlecolor": "white",
    "font.size": 11,
})
sns.set_style("darkgrid", {"axes.facecolor": "#0b1524", "grid.color": "#1f2a3d"})
ACCENT_PALETTE = ["#22d3ee", "#818cf8", "#f97316", "#4ade80", "#f43f5e", "#eab308", "#38bdf8"]
sns.set_palette(ACCENT_PALETTE)

# ============================================================
# 🏷️ HEADER
# ============================================================
st.markdown('<span class="hero-title">📊 Exploratory Data Analysis</span>', unsafe_allow_html=True)
st.caption("Deep dive into the training dataset behind the AQI prediction model")

st.markdown("---")

# ============================================================
# 📂 LOAD DATA (with graceful fallback to a manual upload)
# ============================================================
DEFAULT_PATH = "training_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

df = None
try:
    df = load_data(DEFAULT_PATH)
    st.success(f"Loaded `{DEFAULT_PATH}` — {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    st.warning(f"Couldn't find `{DEFAULT_PATH}` next to the app. Upload it manually below 👇")
    uploaded = st.file_uploader("Upload training_dataset.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.stop()

# ============================================================
# 🧾 DATASET OVERVIEW
# ============================================================
st.subheader("🧾 Dataset Overview")

o1, o2, o3, o4 = st.columns(4)
o1.metric("Rows", f"{df.shape[0]:,}")
o2.metric("Columns", f"{df.shape[1]}")
o3.metric("Missing Cells", f"{int(df.isnull().sum().sum()):,}")
o4.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")

with st.expander("📋 Columns & Data Types"):
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Dtype": df.dtypes.values.astype(str)
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with st.expander("🕳 Missing Values"):
    missing = df.isnull().sum()
    missing_df = missing[missing >= 0].reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    if missing.sum() == 0:
        st.info("No missing values found in the dataset — nice and clean! ✅")

with st.expander("🔍 Preview Data"):
    st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

# ============================================================
# 🗂 TABBED EDA SECTIONS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Target Distribution",
    "📈 Feature Distributions",
    "🔗 Correlation",
    "🧬 AQI vs Features",
    "🕒 Time & Outliers"
])

# ---------- TAB 1: Target distribution ----------
with tab1:
    st.subheader("AQI Distribution")
    t1, t2 = st.columns(2)

    with t1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["aqi"], bins=30, kde=True, color="#22d3ee", ax=ax)
        ax.set_title("AQI Distribution")
        st.pyplot(fig, use_container_width=True)

    with t2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=df["aqi"], color="#818cf8", ax=ax)
        ax.set_title("AQI Boxplot")
        st.pyplot(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean AQI", f"{df['aqi'].mean():.1f}")
    c2.metric("Median AQI", f"{df['aqi'].median():.1f}")
    c3.metric("Std Dev", f"{df['aqi'].std():.1f}")
    c4.metric("Range", f"{df['aqi'].min():.0f} – {df['aqi'].max():.0f}")

# ---------- TAB 2: Feature distributions ----------
with tab2:
    st.subheader("Feature Distributions")

    features = [
        "temperature",
        "humidity",
        "wind_speed",
        "pm2_5",
        "pm10",
        "ozone"
    ]
    features = [f for f in features if f in df.columns]

    grid = st.columns(2)
    for i, feature in enumerate(features):
        with grid[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df[feature], kde=True, color=ACCENT_PALETTE[i % len(ACCENT_PALETTE)], ax=ax)
            ax.set_title(feature)
            st.pyplot(fig, use_container_width=True)

# ---------- TAB 3: Correlation heatmap ----------
with tab3:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False,
        linewidths=.4,
        linecolor="#0b1524",
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

    if "aqi" in df.columns:
        st.markdown("**Top correlations with AQI**")
        corr_with_target = df.corr(numeric_only=True)["aqi"].drop("aqi").sort_values(key=abs, ascending=False)
        st.dataframe(corr_with_target.reset_index().rename(
            columns={"index": "Feature", "aqi": "Correlation with AQI"}
        ), use_container_width=True, hide_index=True)

# ---------- TAB 4: AQI vs features ----------
with tab4:
    st.subheader("AQI vs Features")

    compare = [
        "temperature",
        "humidity",
        "pm2_5",
        "pm10",
        "ozone"
    ]
    compare = [f for f in compare if f in df.columns]

    grid = st.columns(2)
    for i, feature in enumerate(compare):
        with grid[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(
                x=df[feature],
                y=df["aqi"],
                color=ACCENT_PALETTE[i % len(ACCENT_PALETTE)],
                alpha=.7,
                ax=ax
            )
            ax.set_title(f"{feature} vs AQI")
            st.pyplot(fig, use_container_width=True)

# ---------- TAB 5: Time series + outliers ----------
with tab5:
    if "date" in df.columns:
        st.subheader("AQI Over Time")
        df_time = df.copy()
        df_time["date"] = pd.to_datetime(df_time["date"])

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_time["date"], df_time["aqi"], color="#22d3ee", linewidth=1.5)
        ax.set_title("AQI Over Time")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No `date` column found — skipping the time series plot.")

    st.subheader("Outlier Detection")
    outlier_cols = [c for c in ["temperature", "humidity", "pm2_5", "pm10", "aqi"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df[outlier_cols], ax=ax, palette=ACCENT_PALETTE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_title("Outlier Detection")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.caption("📊 EDA page — auto-generated from the training dataset • Built with Streamlit, Matplotlib & Seaborn")
