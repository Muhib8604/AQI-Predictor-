from fastapi import FastAPI

from weather import get_openweather_data
from aqi import get_aqicn_data
from feature_snapshot import build_today_features
from predict import predict_all_horizons


app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "AQI Prediction API is running!"
    }


@app.get("/predict")
def predict():

    # ========================================================
    # 1. 3-DAY WEATHER FORECAST
    #    Display information only
    # ========================================================

    try:
        weather_forecast = get_openweather_data()

    except Exception as e:
        return {
            "error": f"Weather API failed: {str(e)}"
        }

    if weather_forecast is None:
        return {
            "error": "Unable to fetch weather forecast"
        }

    # ========================================================
    # 2. TODAY'S MODEL FEATURE SNAPSHOT
    # ========================================================

    try:
        today_features = build_today_features()

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": f"Feature pipeline failed: {str(e)}"
        }

    if today_features is None:
        return {
            "error": "Unable to fetch current weather / features",
            "details": str(e),
            "type": type(e).__name__
        } 

    # ========================================================
    # 3. LIVE AQI
    # ========================================================

    try:
        live = get_aqicn_data()
    except Exception as e:
        print(f"Live AQI fetch failed: {e}")
        live = None

    live_aqi = None
    live_station_name = None

    if live and live.get("aqi") is not None:

        try:
            live_aqi = float(live["aqi"])
            live_station_name = live.get("station_name")

        except (TypeError, ValueError):
            live_aqi = None

    # ========================================================
    # 4. PUT LIVE AQI INTO THE FEATURE SNAPSHOT
    # ========================================================
    #
    # This is important.
    #
    # The model should know today's actual AQI.
    # The feature_snapshot module is responsible for creating
    # the historical lag/rolling features.
    #
    # We only override today's observed AQI here if AQICN
    # returned a valid live value.
    # ========================================================

    if live_aqi is not None:
        today_features["aqi"] = live_aqi

    # ========================================================
    # 5. RUN THE 3 HORIZON MODELS
    # ========================================================

    try:

        horizon_results = predict_all_horizons(
            today_features
        )

    except (RuntimeError, ValueError) as e:

        return {
            "error": str(e)
        }

    # ========================================================
    # 6. BUILD DISPLAY PREDICTIONS
    # ========================================================

    horizon_keys = [
        "day1",
        "day2",
        "day3"
    ]

    predictions = []

    for i, day in enumerate(weather_forecast[:3]):

        horizon_name = horizon_keys[i]

        result = horizon_results[horizon_name]

        predictions.append({

            "date": day["date"],

            "predicted_aqi": result[
                "predicted_aqi"
            ],

            "model_used": result[
                "model_used"
            ],

            "temperature": day["temperature"],
            "humidity": day["humidity"],
            "pressure": day["pressure"],
            "wind_speed": day["wind_speed"],
            "rain": day["rain"],

            "pm25": day["pm25"],
            "pm10": day["pm10"],
            "ozone": day["ozone"],

            "carbon_monoxide":
                day["carbon_monoxide"],

            "nitrogen_dioxide":
                day["nitrogen_dioxide"],

            "sulphur_dioxide":
                day["sulphur_dioxide"],
        })

    # ========================================================
    # 7. FINAL RESPONSE
    # ========================================================

    return {

        "3_day_AQI_forecast": predictions,

        "average_aqi":
            horizon_results["average_aqi"],

        "live_aqi":
            live_aqi,

        "live_station_name":
            live_station_name,
    }