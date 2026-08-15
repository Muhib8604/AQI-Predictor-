import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from weather import get_openweather_data
from aqi import get_aqicn_data

openweather_clean_data = get_openweather_data()
print("OpenWeather Data:")
print("Timestamp:", openweather_clean_data["timestamp"])

print("Temperature:", openweather_clean_data["weather_temperature"], "°C")

print("Humidity:", openweather_clean_data["weather_humidity"], "%")

print("Pressure:", openweather_clean_data["pressure"], "hPa")

print("Wind Speed:", openweather_clean_data["wind_speed"], "m/s")

print("Wind Direction:", openweather_clean_data["wind_direction"], "°")

print("Rain:", openweather_clean_data["rain"], "mm")

print("Cloud Coverage:", openweather_clean_data["clouds"], "%")


aqicn_clean_data = get_aqicn_data()
print("AQICN DATA:")
print("Station ID:", aqicn_clean_data["station_id"])

print("Station Name:", aqicn_clean_data["station_name"])

print("AQI:", aqicn_clean_data["aqi"])

print("PM1:", aqicn_clean_data["pm1"])

print("PM2.5:", aqicn_clean_data["pm25"])

print("PM10:", aqicn_clean_data["pm10"])

print("Temperature:", aqicn_clean_data["aqi_temperature"], "°C")

print("Humidity:", aqicn_clean_data["aqi_humidity"], "%")

print("Timestamp:", aqicn_clean_data["timestamp"])

weather_df = pd.DataFrame([openweather_clean_data])

weather_df["timestamp"] = pd.to_datetime(
    weather_df["timestamp"],
    utc=True
)
weather_df["hour"] = weather_df["timestamp"].dt.floor("h")

aqi_df = pd.DataFrame([aqicn_clean_data])

aqi_df["timestamp"] = pd.to_datetime(
    aqi_df["timestamp"],
    utc=True
)

aqi_df["hour"] = aqi_df["timestamp"].dt.floor("h")

combined_df = pd.merge(
    weather_df,
    aqi_df,
    on="hour",
    how="inner"
)

print("\nWeather DataFrame:")
print(weather_df)

print("\nAQI DataFrame:")
print(aqi_df)

print("\nCombined DataFrame:")
print(combined_df)