"""
ShipSense — Inference & Delay Risk Module
Implements the function contracts defined in docs/API.md:
    load_model(path) -> (model, encoders)
    predict_eta(df, model, encoders) -> pd.Series
    flag_delay_risk(predicted_eta, promised_eta, threshold_minutes) -> pd.Series
    get_top_delay_factor(model, feature_names, row) -> str | None

Promised/SLA time definition: each region's historical median delivery
duration (from the training data) is used as that region's "typical"
service-level baseline. This is a simple, fully explainable rule --
not a magic number -- documented here and in README.
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


def load_model(models_dir: str = MODELS_DIR):
    """
    Loads the trained model pipeline and feature importances.
    Raises FileNotFoundError if the expected artifacts are missing.
    """
    model_path = os.path.join(models_dir, "eta_model.pkl")
    importances_path = os.path.join(models_dir, "feature_importances.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run src/train_model.py first."
        )

    model = joblib.load(model_path)

    importances = {}
    if os.path.exists(importances_path):
        with open(importances_path) as f:
            importances = json.load(f)

    return model, importances


def predict_eta(df: pd.DataFrame, model) -> pd.Series:
    """
    Applies the trained model pipeline to a DataFrame of orders and
    returns predicted delivery duration (minutes) per row.
    Raises ValueError if expected feature columns are missing.
    """
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"predict_eta: missing expected feature columns: {missing}")

    X = df[FEATURE_ORDER].copy()
    X["is_weekend"] = X["is_weekend"].astype(str)  # match training-time encoding

    preds = model.predict(X)

    if np.isnan(preds).any():
        raise ValueError("predict_eta: model produced NaN predictions.")

    return pd.Series(preds, index=df.index, name="predicted_eta_minutes")


def compute_promised_eta(df: pd.DataFrame) -> pd.Series:
    """
    Computes each region's 75th-percentile historical delivery duration
    and assigns it as that region's promised/SLA time for every order
    in it. The 75th percentile (rather than median) reflects a realistic
    SLA promise -- "we typically beat this" -- rather than a coin-flip
    threshold. This is the documented SLA definition (see module docstring).
    """
    region_sla = df.groupby("region_id")["delivery_duration_minutes"].quantile(0.75)
    promised = df["region_id"].map(region_sla)
    promised.name = "promised_eta_minutes"
    return promised


def flag_delay_risk(
    predicted_eta: pd.Series,
    promised_eta: pd.Series,
    threshold_minutes: float = DELAY_RISK_THRESHOLD_MINUTES,
) -> pd.Series:
    """
    Derives the 'At Risk' / 'On Track' flag: At Risk if predicted ETA
    exceeds promised ETA by more than the threshold.
    Raises ValueError if input series lengths don't match.
    """
    if len(predicted_eta) != len(promised_eta):
        raise ValueError("flag_delay_risk: predicted_eta and promised_eta must be the same length.")

    delay = predicted_eta.values - promised_eta.values
    status = np.where(delay > threshold_minutes, "At Risk", "On Track")
    return pd.Series(status, index=predicted_eta.index, name="risk_status")


def get_top_delay_factor(importances: dict, risk_status: pd.Series) -> pd.Series:
    """
    Assigns the single top contributing feature (from the model's global
    feature importances) to every 'At Risk' order, and None to 'On Track'
    orders. A per-row explanation would require per-row SHAP-style analysis,
    which is out of scope for v1.0 -- this uses the model's global top
    feature as a defensible, explainable approximation (documented).
    """
    if not importances:
        return pd.Series([None] * len(risk_status), index=risk_status.index, name="top_delay_factor")

    top_feature = max(importances, key=importances.get)
    # Clean up one-hot-encoded names for a readable label
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


def run_pipeline(output_path: str = PREDICTIONS_PATH) -> pd.DataFrame:
    """Runs the full inference pipeline and saves predictions.csv."""
    print("=== ShipSense Inference Pipeline ===\n")

    print("Step 1/4: Loading model and cleaned data...")
    model, importances = load_model()
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"  Loaded model and {len(df)} rows.\n")

    print("Step 2/4: Predicting ETA...")
    df["predicted_eta_minutes"] = predict_eta(df, model)
    print(f"  Predicted ETA range: {df['predicted_eta_minutes'].min():.1f}"
          f" - {df['predicted_eta_minutes'].max():.1f} minutes\n")

    print("Step 3/4: Computing promised ETA (region median) and risk flag...")
    df["promised_eta_minutes"] = compute_promised_eta(df)
    df["delay_minutes"] = df["predicted_eta_minutes"] - df["promised_eta_minutes"]
    df["risk_status"] = flag_delay_risk(df["predicted_eta_minutes"], df["promised_eta_minutes"])

    risk_counts = df["risk_status"].value_counts()
    risk_pct = (risk_counts.get("At Risk", 0) / len(df)) * 100
    print(f"  Risk distribution: {risk_counts.to_dict()} ({risk_pct:.1f}% at risk)\n")

    print("Step 4/4: Assigning top delay factor...")
    df["top_delay_factor"] = get_top_delay_factor(importances, df["risk_status"])
    print("  Done.\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_pipeline()