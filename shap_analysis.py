import json
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("training_features.csv")

with open("model_manifest.json") as f:
    manifest = json.load(f)

for horizon in ["day1", "day2", "day3"]:

    info = manifest[horizon]

    if info["kind"] != "sklearn":

        print(f"{horizon} uses PyTorch.")
        print("Skipping SHAP TreeExplainer.")
        continue

    model = joblib.load(info["file"])

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.title(f"{horizon} Feature Importance")

    plt.show()