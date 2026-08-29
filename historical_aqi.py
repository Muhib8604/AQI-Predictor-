import pandas as pd



aqi_df = pd.read_csv(
    "air_quality_historical.csv",
    sep="\t"
)



print("Original Dataset:")
print(aqi_df.head())

print("\nColumns:")
print(aqi_df.columns.tolist())

print("\nOriginal Shape:")
print(aqi_df.shape)



aqi_df = aqi_df.dropna(subset=["us_aqi"])



aqi_df = aqi_df[
    [
        "date",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi"
    ]
]


aqi_df = aqi_df.rename(
    columns={
        "us_aqi": "aqi"
    }
)



print("\nCleaned Dataset:")
print(aqi_df.head())

print("\nCleaned Shape:")
print(aqi_df.shape)



aqi_df.to_csv(
    "historical_aqi_clean.csv",
    index=False
)

print("\nHistorical AQI dataset cleaned successfully!")