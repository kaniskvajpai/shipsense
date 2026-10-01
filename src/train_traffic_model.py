"""
ShipSense — Traffic Prediction Module (post-v1.0 addition)

No live traffic API is used (correctly out of v1.0 scope, per
docs/ARCHITECTURE.md's no-live-data decision). Instead, this trains
a model to predict a CONGESTION PROXY: minutes of travel time per
kilometer, by region_id x hour_of_day x day_of_week. A higher value
means deliveries in that region/time are taking longer per km than
average -- a reasonable stand-in for "traffic conditions" derived
entirely from historical delivery data already in hand.

This is NOT real traffic data. It is an honest, clearly-labeled proxy.
Matches the existing offline-training, batch-scored architecture.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLEANED_DATA_PATH, MODELS_DIR, RANDOM_STATE

TRAFFIC_FEATURES = ["region_id", "hour_of_day", "day_of_week", "is_weekend"]


def build_traffic_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes minutes-per-km per order (dropping zero/near-zero distance
    rows to avoid division blow-ups), then aggregates to the mean
    congestion proxy per region x hour x day-of-week x weekend.
    """
    d = df.copy()
    d = d[d["distance_km"] > 0.05]
    d["min_per_km"] = d["delivery_duration_minutes"] / d["distance_km"]

    upper = d["min_per_km"].quantile(0.99)
    d = d[d["min_per_km"] <= upper]

    agg = (
        d.groupby(["region_id", "hour_of_day", "day_of_week", "is_weekend"])["min_per_km"]
        .mean()
        .reset_index(name="min_per_km")
    )
    return agg


def train_traffic_model(agg_df: pd.DataFrame):
    """Trains a Random Forest to predict the min_per_km congestion proxy."""
    X = agg_df[TRAFFIC_FEATURES].copy()
    X["is_weekend"] = X["is_weekend"].astype(str)
    y = agg_df["min_per_km"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["is_weekend"])],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=12,
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    metrics = {
        "mae_min_per_km": round(float(mae), 3),
        "mean_actual_min_per_km": round(float(y_test.mean()), 3),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return pipeline, metrics


def save_traffic_model(model, metrics: dict, models_dir: str = MODELS_DIR) -> None:
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "traffic_model.pkl")
    metrics_path = os.path.join(models_dir, "traffic_metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved traffic model to: {model_path}")
    print(f"Saved traffic metrics to: {metrics_path}")


def run_pipeline():
    print("=== ShipSense Traffic Prediction — Training ===\n")

    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {len(df)} rows.\n")

    print("Building congestion-proxy dataset (minutes per km)...")
    agg_df = build_traffic_dataset(df)
    print(f"Built {len(agg_df)} region/hour/day-of-week combinations.\n")

    print("Training traffic congestion model...")
    model, metrics = train_traffic_model(agg_df)
    print(f"MAE: {metrics['mae_min_per_km']} min/km "
          f"(mean actual: {metrics['mean_actual_min_per_km']} min/km)\n")

    save_traffic_model(model, metrics)

    all_regions = df["region_id"].unique()
    grid = pd.MultiIndex.from_product(
        [all_regions, range(24), range(7), [True, False]],
        names=["region_id", "hour_of_day", "day_of_week", "is_weekend"]
    ).to_frame(index=False)

    X_grid = grid[TRAFFIC_FEATURES].copy()
    X_grid["is_weekend"] = X_grid["is_weekend"].astype(str)
    grid["predicted_min_per_km"] = model.predict(X_grid).round(3)

    q33, q66 = grid["predicted_min_per_km"].quantile([0.33, 0.66])
    def label(v):
        if v <= q33:
            return "Low"
        elif v <= q66:
            return "Moderate"
        return "High"
    grid["congestion_level"] = grid["predicted_min_per_km"].apply(label)

    forecast_path = os.path.join("data", "processed", "traffic_forecast.csv")
    grid.to_csv(forecast_path, index=False)
    print(f"Saved full traffic forecast grid to: {forecast_path}")
    print(f"Grid shape: {grid.shape}")

    return model, metrics


if __name__ == "__main__":
    run_pipeline()