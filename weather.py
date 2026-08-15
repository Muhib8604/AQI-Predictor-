import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()


def get_openweather_data():

    api_key = os.getenv("OPENWEATHER_API_KEY")

    lat = 24.8576
    lon = 67.0302

    # Weather forecast
    weather_url = "https://api.openweathermap.org/data/2.5/forecast"

    weather_response = requests.get(
        weather_url,
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key
        }
    )

    if weather_response.status_code != 200:
        print("Error fetching weather forecast")
        return None

    weather_data = weather_response.json()

    # Air pollution forecast
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

    air_response = requests.get(
        air_url,
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key
        }
    )

    if air_response.status_code != 200:
        print("Error fetching air pollution forecast")
        return None

    air_data = air_response.json()

    forecast = weather_data["list"]
    air_forecast = air_data["list"]

    print("Forecast entries:", len(forecast))
    print("Air forecast entries:", len(air_forecast))

    weather_indexes = [7, 15, 23]
    air_indexes = [7, 15, 23]

    result = []

    for w, a in zip(weather_indexes, air_indexes):

        weather = forecast[w]
        pollution = air_forecast[a]["components"]

        result.append({
            "date": weather["dt_txt"],
            "temperature": weather["main"]["temp"] - 273.15,
            "humidity": weather["main"]["humidity"],
            "pressure": weather["main"]["pressure"],
            "wind_speed": weather["wind"]["speed"],
            "rain": weather.get("rain", {}).get("3h", 0),
            "clouds": weather["clouds"]["all"],

            "pm10": pollution["pm10"],
            "pm25": pollution["pm2_5"],
            "carbon_monoxide": pollution["co"],
            "nitrogen_dioxide": pollution["no2"],
            "sulphur_dioxide": pollution["so2"],
            "ozone": pollution["o3"]
        })

    return result


if __name__ == "__main__":
    print(get_openweather_data())


def get_current_weather():
    """Fetch the CURRENT (right-now) weather + pollutant reading, as opposed
    to get_openweather_data() above which returns 3-day FORECAST slices.
    Used by feature_pipeline.py for hourly feature-store ingestion, where we
    want to log what conditions actually were at this hour, not a forecast."""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    lat = 24.8576
    lon = 67.0302

    weather_response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key}
    )
    if weather_response.status_code != 200:
        print("Error fetching current weather")
        return None
    weather_data = weather_response.json()

    air_response = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params={"lat": lat, "lon": lon, "appid": api_key}
    )
    if air_response.status_code != 200:
        print("Error fetching current air pollution")
        return None
    pollution = air_response.json()["list"][0]["components"]

    return {
        "date": datetime.fromtimestamp(weather_data["dt"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": weather_data["main"]["temp"] - 273.15,
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "rain": weather_data.get("rain", {}).get("1h", 0),

        "pm10": pollution["pm10"],
        "pm25": pollution["pm2_5"],
        "carbon_monoxide": pollution["co"],
        "nitrogen_dioxide": pollution["no2"],
        "sulphur_dioxide": pollution["so2"],
        "ozone": pollution["o3"]
    }