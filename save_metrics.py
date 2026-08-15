import joblib

metrics = {

    "MAE": 5.60,

    "RMSE": 8.62,

    "R2": 0.939

}

joblib.dump(
    metrics,
    "model_metrics.pkl"
)

print("Metrics saved.")