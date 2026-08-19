from fastapi import FastAPI

from weather import get_openweather_data
from aqi import get_aqicn_data
from feature_snapshot import build_today_features
from predict import predict_all_horizons

app = FastAPI()


@app.get("/")
def home():
    return {"message": "AQI Prediction API is running!"}


@app.get("/predict")
def predict():

    # ---- 1. 3-day WEATHER FORECAST (display only) ----
    weather_forecast = get_openweather_data()
    if weather_forecast is None:
        return {"error": "Unable to fetch weather forecast"}

    # ---- 2. TODAY's feature snapshot (model input) ----
    today_features = build_today_features()
    if today_features is None:
        return {"error": "Unable to fetch current weather / features"}

    # ---- 3. Run the 3 day-specific models ----
    try:
        horizon_results = predict_all_horizons(today_features)
    except RuntimeError as e:
        return {"error": str(e)}

    # ---- 4. Live station reading ----
    live = get_aqicn_data()
    live_aqi = None
    live_station_name = None
    if live and live.get("aqi") is not None:
        try:
            live_aqi = float(live["aqi"])
            live_station_name = live.get("station_name")
        except (TypeError, ValueError):
            live_aqi = None

    # ---- 5. Light blend for Day-1 (reduces large gap when live is much cleaner) ----
    if live_aqi is not None:
        model_day1 = float(horizon_results["day1"]["predicted_aqi"])
        blended_day1 = 0.60 * model_day1 + 0.40 * live_aqi
        print(
            f"Day1 blend: model={model_day1:.1f}, live={live_aqi:.1f}, "
            f"blended={blended_day1:.1f}"
        )
        horizon_results["day1"]["predicted_aqi"] = blended_day1

        # Recompute average after blend
        horizon_results["average_aqi"] = (
            horizon_results["day1"]["predicted_aqi"]
            + horizon_results["day2"]["predicted_aqi"]
            + horizon_results["day3"]["predicted_aqi"]
        ) / 3.0

    # ---- 6. Merge predictions with display weather ----
    horizon_keys = ["day1", "day2", "day3"]
    predictions = []

    for i, day in enumerate(weather_forecast[:3]):
        horizon_name = horizon_keys[i]
        result = horizon_results[horizon_name]

        predictions.append({
            "date": day["date"],
            "predicted_aqi": result["predicted_aqi"],
            "model_used": result["model_used"],
            "temperature": day["temperature"],
            "humidity": day["humidity"],
            "pressure": day["pressure"],
            "wind_speed": day["wind_speed"],
            "rain": day["rain"],
            "pm25": day["pm25"],
            "pm10": day["pm10"],
            "ozone": day["ozone"],
            "carbon_monoxide": day["carbon_monoxide"],
            "nitrogen_dioxide": day["nitrogen_dioxide"],
            "sulphur_dioxide": day["sulphur_dioxide"],
        })

    return {
        "3_day_AQI_forecast": predictions,
        "average_aqi": horizon_results["average_aqi"],
        "live_aqi": live_aqi,
        "live_station_name": live_station_name,
    }