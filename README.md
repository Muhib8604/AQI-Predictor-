# 🌫️ Air Quality Index (AQI) Predictor & Forecasting System — Karachi, Pakistan

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-00C853?style=for-the-badge)](https://www.hopsworks.ai/)

An end-to-end Machine Learning pipeline and interactive analytical application predicting and forecasting Air Quality Index (AQI) metrics specifically calibrated for **Karachi, Pakistan**. Built with modern MLOps architecture, feature engineering pipelines, SHAP explainability, and multi-model performance benchmarks.

---

## 👤 Author & Links

- **Author:** Muhib Rashid
- **LinkedIn:** [linkedin.com/in/muhib-rashid-027881362](https://www.linkedin.com/in/muhib-rashid-027881362)
- **GitHub Repository:** [github.com/Muhib8604/AQI-Predictor-](https://github.com/Muhib8604/AQI-Predictor-)

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Dataset & Features](#-dataset--features)
5. [Model Benchmarks & Performance](#-model-benchmarks--performance)
6. [Interactive Web Dashboard](#-interactive-web-dashboard)
7. [Installation & Setup](#-installation--setup)
8. [Usage Guide](#-usage-guide)
9. [Project Structure](#-project-structure)
10. [Future Enhancements](#-future-enhancements)

---

## 🚀 Project Overview

Air pollution in major urban centers like Karachi presents severe public health risks. This project delivers a machine learning solution to analyze historical meteorological and pollutant data, train multiple ML/DL regression models, track experiment runs with MLflow, and deploy an interactive Streamlit application for real-time risk assessment, feature importance explainability (SHAP), and multi-day forecasting.

---

## ✨ Key Features

- 🔄 **Feature Store Integration:** Automated feature pipelines leveraging Hopsworks Feature Store for scalable data ingestion.
- 🧪 **Experiment Tracking:** Comprehensive MLflow integration tracking hyperparameter runs, metric logs (MAE, RMSE, R²), and model artifacts.
- 📊 **Exploratory Data Analysis (EDA):** Tabbed visual analysis of pollutant distributions, temporal trends, correlation matrices, and outlier detections using Plotly.
- 🔍 **Model Explainability (SHAP):** Global feature importance and individual prediction breakdown using SHAP summary plots, waterfall charts, and force plots.
- ⚡ **Interactive Prediction Lab:** Real-time scenario simulator allowing users to modify atmospheric parameters and assess predicted AQI levels instantly.
- 📈 **Multi-Model Comparison:** Evaluates Linear Models (Ridge/Lasso), Tree Ensembles (Random Forest, XGBoost, LightGBM), and Neural Networks (PyTorch).

---

## 🏗 System Architecture

```
[ OpenWeather / AQICN APIs ]
           │
           ▼
  [ Feature Pipeline ] ─────► [ Hopsworks Feature Store ]
                                         │
                                         ▼
                                [ Model Training ]
                                (Scikit-Learn, PyTorch)
                                         │
                                         ▼
                                [ MLflow Registry ]
                                         │
                                         ▼
                           [ Streamlit Web Application ]
                                (Dashboard / SHAP / Simulator)
```

---

## 📊 Dataset & Features

The system utilizes atmospheric measurements and engineered meteorological indicators, including:

| Category | Feature Variables |
| :--- | :--- |
| **Pollutants** | PM2.5, PM10, NO2, SO2, CO, O3 |
| **Weather** | Temperature, Relative Humidity, Wind Speed, Atmospheric Pressure, Visibility |
| **Engineered** | Rolling averages, lag values, diurnal cycle indicators, interactions |
| **Target Variable** | Air Quality Index (AQI) / US-AQI Standard |

---

## 🏆 Model Benchmarks & Performance

Multiple algorithms were evaluated under standardized cross-validation and test holdouts:

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **Lowest** | **Lowest** | **Highest (~0.90+)** |
| XGBoost / LightGBM | Competitive | Competitive | High |
| PyTorch Neural Net | Moderate | Moderate | Fair |
| Ridge Regression | Baseline | Baseline | Baseline |

---

## 💻 Interactive Web Dashboard

The Streamlit dashboard consists of the following visual tabs:
1. **Overview & Live Status:** Current AQI gauge, severity band indicator, and atmospheric summary cards.
2. **Exploratory Analysis:** Distribution plots, correlation heatmaps, lag relationships, and anomaly inspection.
3. **Model Explainability:** SHAP summary charts highlighting main drivers (e.g., PM2.5, Humidity, Temperature).
4. **Model Comparison:** Comparative bar charts for MAE/RMSE across evaluated algorithms.
5. **Custom Prediction Lab:** Interactive sliders allowing users to simulate custom climate scenarios.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Muhib8604/AQI-Predictor-.git
cd AQI-Predictor-
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory with your API keys:
```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
AQICN_API_KEY=your_aqicn_api_key
```

---

## 🚀 Usage Guide

### Run Feature & Training Pipeline
```bash
python src/feature_pipeline.py
python src/train.py
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
AQI-Predictor-/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory notebooks & SHAP analysis
├── src/
│   ├── feature_pipeline.py# Data ingestion & Hopsworks sync
│   ├── train.py           # Model training & MLflow logging
│   ├── evaluate.py        # SHAP calculation & metric evaluations
│   └── utils.py           # Data processing utilities
├── app.py                 # Streamlit web application
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 🤝 Connect & Feedback

Feel free to explore the repository, open issues, or reach out directly for collaborations!

- **LinkedIn:** [Muhib Rashid](https://www.linkedin.com/in/muhib-rashid-027881362)
- **GitHub:** [@Muhib8604](https://github.com/Muhib8604)
