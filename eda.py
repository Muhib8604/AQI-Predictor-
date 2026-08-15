import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("training_dataset.csv")

# ----------------------------
# DATASET OVERVIEW
# ----------------------------

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n")

print("=" * 50)
print("COLUMNS")
print("=" * 50)
print(df.columns)

print("\n")

print("=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n")

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ----------------------------
# TARGET DISTRIBUTION
# ----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["aqi"], bins=30, kde=True)
plt.title("AQI Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(y=df["aqi"])
plt.title("AQI Boxplot")
plt.show()

# ----------------------------
# CORRELATION HEATMAP
# ----------------------------

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# FEATURE DISTRIBUTIONS
# ----------------------------

features = [

    "temperature",
    "humidity",
    "wind_speed",
    "pm2_5",
    "pm10",
    "ozone"

]

for feature in features:

    plt.figure(figsize=(7,4))

    sns.histplot(df[feature], kde=True)

    plt.title(feature)

    plt.show()

# ----------------------------
# AQI VS FEATURES
# ----------------------------

compare = [

    "temperature",
    "humidity",
    "pm2_5",
    "pm10",
    "ozone"

]

for feature in compare:

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        x=df[feature],
        y=df["aqi"]
    )

    plt.title(f"{feature} vs AQI")

    plt.show()

# ----------------------------
# AQI OVER TIME
# ----------------------------

if "date" in df.columns:

    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(14,5))

    plt.plot(
        df["date"],
        df["aqi"]
    )

    plt.title("AQI Over Time")

    plt.show()

# ----------------------------
# OUTLIERS
# ----------------------------

plt.figure(figsize=(10,5))

sns.boxplot(
    data=df[
        [
            "temperature",
            "humidity",
            "pm2_5",
            "pm10",
            "aqi"
        ]
    ]
)

plt.xticks(rotation=30)

plt.title("Outlier Detection")

plt.show()

print("\nEDA COMPLETE.")