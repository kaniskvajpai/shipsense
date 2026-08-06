"""
ShipSense — Inference & Delay Risk Module
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DELAY_RISK_THRESHOLD_MINUTES, CLEANED_DATA_PATH, PREDICTIONS_PATH, MODELS_DIR,
)
from src.train_model import FEATURE_ORDER


def load_model(models_dir=MODELS_DIR):
    model_path = os.path.join(models_dir, "eta_model.pkl")
    importances_path = os.path.join(models_dir, "feature_importances.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train_model.py first.")

    model = joblib.load(model_path)

    importances = {}
    if os.path.exists(importances_path):
        with open(importances_path) as f:
            importances = json.load(f)

    return model, importances


def predict_eta(df, model):
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"predict_eta: missing expected feature columns: {missing}")

    X = df[FEATURE_ORDER].copy()
    X["is_weekend"] = X["is_weekend"].astype(str)

    preds = model.predict(X)

    if np.isnan(preds).any():
        raise ValueError("predict_eta: model produced NaN predictions.")

    return pd.Series(preds, index=df.index, name="predicted_eta_minutes")


def compute_promised_eta(df):
    region_sla = df.groupby("region_id")["delivery_duration_minutes"].quantile(0.75)
    global_fallback = df["delivery_duration_minutes"].quantile(0.75)

    promised = df["region_id"].map(region_sla)
    if promised.isnull().any():
        n_missing = promised.isnull().sum()
        print(f"[compute_promised_eta] WARNING: {n_missing} rows had no regional SLA data; using fallback.")
        promised = promised.fillna(global_fallback)

    promised.name = "promised_eta_minutes"
    return promised


def flag_delay_risk(predicted_eta, promised_eta, threshold_minutes=DELAY_RISK_THRESHOLD_MINUTES):
    if len(predicted_eta) != len(promised_eta):
        raise ValueError("flag_delay_risk: length mismatch.")

    delay = predicted_eta.values - promised_eta.values
    status = np.where(delay > threshold_minutes, "At Risk", "On Track")
    return pd.Series(status, index=predicted_eta.index, name="risk_status")


def get_top_delay_factor(importances, risk_status):
    if not importances:
        return pd.Series([None] * len(risk_status), index=risk_status.index, name="top_delay_factor")

    top_feature = max(importances, key=importances.get)
    readable = top_feature.replace("cat__", "").split("_")[0] if "cat__" in top_feature else top_feature
    label_map = {
        "distance_km": "Distance",
        "courier_id": "Courier",
        "region_id": "Region",
        "aoi_id": "Delivery Area (AOI)",
        "hour_of_day": "Hour of Day",
        "day_of_week": "Day of Week",
    }
    readable = label_map.get(readable, readable)

    result = np.where(risk_status.values == "At Risk", readable, None)
    return pd.Series(result, index=risk_status.index, name="top_delay_factor")


def run_pipeline(output_path=PREDICTIONS_PATH):
    print("=== ShipSense Inference Pipeline ===")

    print("Step 1/4: Loading model and cleaned data...")
    model, importances = load_model()
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded model and {len(df)} rows.")

    print("Step 2/4: Predicting ETA...")
    df["predicted_eta_minutes"] = predict_eta(df, model)
    print(f"Predicted ETA range: {df['predicted_eta_minutes'].min():.1f} - {df['predicted_eta_minutes'].max():.1f} minutes")

    print("Step 3/4: Computing promised ETA and risk flag...")
    df["promised_eta_minutes"] = compute_promised_eta(df)
    df["delay_minutes"] = df["predicted_eta_minutes"] - df["promised_eta_minutes"]
    df["risk_status"] = flag_delay_risk(df["predicted_eta_minutes"], df["promised_eta_minutes"])

    risk_counts = df["risk_status"].value_counts()
    risk_pct = (risk_counts.get("At Risk", 0) / len(df)) * 100
    print(f"Risk distribution: {risk_counts.to_dict()} ({risk_pct:.1f}% at risk)")

    print("Step 4/4: Assigning top delay factor...")
    df["top_delay_factor"] = get_top_delay_factor(importances, df["risk_status"])
    print("Done.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_pipeline()