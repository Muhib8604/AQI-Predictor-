import json
import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch

from model_definition import AQINet


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Karachi_AQI_Prediction")


with open("model_manifest.json") as f:
    manifest = json.load(f)


for horizon in ["day1", "day2", "day3"]:

    info = manifest[horizon]

    with mlflow.start_run(run_name=f"{horizon}_{info['model_type']}"):

        

        mlflow.log_param("Forecast Horizon", horizon)
        mlflow.log_param("Model Type", info["model_type"])
        mlflow.log_param("Model Kind", info["kind"])
        mlflow.log_param("Residual Model", info["is_delta"])

        

        mlflow.log_metric("MAE", info["mae"])
        mlflow.log_metric("RMSE", info["rmse"])
        mlflow.log_metric("R2", info["r2"])

        model_name = f"Karachi_AQI_{horizon}"

        

        if info["kind"] == "sklearn":

            model = joblib.load(info["file"])

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )

        
        else:

            feature_count = len(joblib.load("model_features.pkl"))

            model = AQINet(feature_count)

            model.load_state_dict(
                torch.load(info["file"], map_location="cpu")
            )

            model.eval()

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )

print("\nAll champion models successfully registered.")