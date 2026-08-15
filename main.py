from fastapi import FastAPI

from weather import get_openweather_data
from aqi import get_aqicn_data  # noqa: F401 (available for future use)

from history import update_history
from feature_snapshot import build_today_features
from predict import predict_all_horizons

app = FastAPI()


@app.get("/")
def home():
    return {"message": "AQI Prediction API is running!"}


@app.get("/predict")
def predict():

    # ---- 1. 3-day WEATHER FORECAST — used only to show each day's raw
    # weather/pollutant readings in the dashboard cards, not fed to the models ----
    weather_forecast = get_openweather_data()
    if weather_forecast is None:
        return {"error": "Unable to fetch weather forecast"}

    # ---- 2. TODAY's actual snapshot — this is what all 3 day-specific
    # models are actually trained on and predict from. Built by
    # feature_snapshot.py, the SAME function the Explainability dashboard
    # page uses, so the two can never drift out of sync again. ----
    today_features = build_today_features()
    if today_features is None:
        return {"error": "Unable to fetch current weather"}

    # ---- 3. Run the 3 day-specific models on this SAME snapshot ----
    try:
        horizon_results = predict_all_horizons(today_features)
    except RuntimeError as e:
        return {"error": str(e)}

    # ---- 4. Merge each day's prediction with that day's DISPLAY weather ----
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

    # Keep the local lag/rolling history in sync with today's day1 prediction
    update_history(horizon_results["day1"]["predicted_aqi"])

    return {
        "3_day_AQI_forecast": predictions,
        "average_aqi": horizon_results["average_aqi"],
    }
