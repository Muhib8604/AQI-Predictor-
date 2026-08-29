import time
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()


REQUEST_TIMEOUT = 10          
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2     


def _get_with_retry(url, params, what: str):
    """GET with a timeout and a few retries. Returns the parsed JSON on
    success, or None (after logging) if every attempt fails."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"[weather] {what} attempt {attempt}/{MAX_RETRIES} failed — {last_error}")
        except requests.exceptions.Timeout:
            last_error = f"timed out after {REQUEST_TIMEOUT}s"
            print(f"[weather] {what} attempt {attempt}/{MAX_RETRIES} — {last_error}")
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"[weather] {what} attempt {attempt}/{MAX_RETRIES} — request error: {last_error}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"[weather] {what} FAILED after {MAX_RETRIES} attempts — last error: {last_error}")
    return None


def get_openweather_data():

    api_key = os.getenv("OPENWEATHER_API_KEY")

    lat = 24.8576
    lon = 67.0302

    weather_data = _get_with_retry(
        "https://api.openweathermap.org/data/2.5/forecast",
        {"lat": lat, "lon": lon, "appid": api_key},
        "weather forecast",
    )
    if weather_data is None:
        return None

    air_data = _get_with_retry(
        "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
        {"lat": lat, "lon": lon, "appid": api_key},
        "air pollution forecast",
    )
    if air_data is None:
        return None

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


def get_current_weather():
    """Fetch the CURRENT (right-now) weather + pollutant reading, as opposed
    to get_openweather_data() above which returns 3-day FORECAST slices.
    Used by feature_pipeline.py for hourly feature-store ingestion, where we
    want to log what conditions actually were at this hour, not a forecast."""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    lat = 24.8576
    lon = 67.0302

    weather_data = _get_with_retry(
        "https://api.openweathermap.org/data/2.5/weather",
        {"lat": lat, "lon": lon, "appid": api_key},
        "current weather",
    )
    if weather_data is None:
        return None

    air_data = _get_with_retry(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        {"lat": lat, "lon": lon, "appid": api_key},
        "current air pollution",
    )
    if air_data is None:
        return None

    pollution = air_data["list"][0]["components"]

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


if __name__ == "__main__":
    print(get_openweather_data())
