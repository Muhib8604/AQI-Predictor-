import pandas as pd


weather_df = pd.read_csv("historical_weather.csv")  

aqi_df = pd.read_csv("historical_aqi_clean.csv")



print("Weather Dataset")
print(weather_df.head())

print()

print("AQI Dataset")
print(aqi_df.head())



weather_df["date"] = pd.to_datetime(weather_df["date"])

aqi_df["date"] = pd.to_datetime(aqi_df["date"])



training_df = pd.merge(
    weather_df,
    aqi_df,
    on="date",
    how="inner"
)


print()

print("Merged Dataset")
print(training_df.head())

print()

print("Merged Shape:")
print(training_df.shape)

print()

print("Missing Values:")
print(training_df.isnull().sum())



training_df.to_csv(
    "training_dataset.csv",
    index=False
)

print()

print("Training dataset created successfully!")