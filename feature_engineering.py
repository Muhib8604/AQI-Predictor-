import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("training_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["day_of_week"] = df["date"].dt.day_name()

print(df[["date", "year", "month", "day", "day_of_week"]].head(10))



df["aqi_lag_1"] = df["aqi"].shift(1)

print(
    df[
        [
            "date",
            "aqi",
            "aqi_lag_1"
        ]
    ].head(10)
)

df = df.dropna()

print(df[["date", "aqi", "aqi_lag_1"]].head())



df["aqi_rolling_mean_3"] = (
    df["aqi"]
    .rolling(window=3)
    .mean()
)

print(
    df[
        [
            "date",
            "aqi",
            "aqi_lag_1",
            "aqi_rolling_mean_3"
        ]
    ].head(10)
)



df["aqi_change"] = df["aqi"].diff()

print(
    df[
        [
            "date",
            "aqi",
            "aqi_lag_1",
            "aqi_rolling_mean_3",
            "aqi_change"
        ]
    ].head(10)
)




encoder = LabelEncoder()

df["day_of_week"] = encoder.fit_transform(df["day_of_week"])

print(df[["date", "day_of_week"]].head(10))



df = df.dropna()

X = df[
    [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "year",
        "month",
        "day",
        "day_of_week",
        "aqi_lag_1",
        "aqi_rolling_mean_3"
    ]
]

y = df["aqi"]

print("Features")
pd.set_option("display.max_columns", None)
print(X.head())

print()

print("Target")
print(y.head())

X.to_csv("training_features.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "aqi_prediction_model.pkl")

print("Model saved successfully!")

joblib.dump(
    X.columns.tolist(),
    "model_features.pkl"
)

print("Feature names saved successfully!")

predictions = model.predict(X_test)

print("Predictions:")
print(predictions[:10])

mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print("RMSE:", rmse)


r2 = r2_score(
    y_test,
    predictions
)

print("R² Score:", r2)