"""
ShipSense — Demand Forecasting Module (post-v1.0 addition)
Trains a model to predict expected order VOLUME (count) per
region_id x hour_of_day x day_of_week, using historical patterns.

This is distinct from the existing Demand Heatmap (which shows raw
historical counts) -- this produces a genuine trained forecast that
can be queried for any region/hour/day-of-week combination, including
ones with sparse historical data, via the model's learned pattern
rather than a raw lookup.

Architecture note: matches the existing offline-training, batch-scored
pattern (see docs/ARCHITECTURE.md) -- no live data, no new external
dependencies. Trained from data/processed/cleaned_data.csv, same as
the ETA model.
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

DEMAND_FEATURES = ["region_id", "hour_of_day", "day_of_week", "is_weekend"]


def build_demand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw order-level data into order COUNTS per
    region_id x hour_of_day x day_of_week x is_weekend.
    This is the training set for the forecasting model --
    one row = one (region, hour, day-of-week) combination,
    target = how many orders historically occurred in it.
    """
    counts = (
        df.groupby(["region_id", "hour_of_day", "day_of_week", "is_weekend"])
        .size()
        .reset_index(name="order_count")
    )
    return counts


def train_demand_model(counts_df: pd.DataFrame):
    """
    Trains a Random Forest to predict order_count from
    region_id, hour_of_day, day_of_week, is_weekend.
    Returns (model_pipeline, metrics_dict).
    """
    X = counts_df[DEMAND_FEATURES].copy()
    X["is_weekend"] = X["is_weekend"].astype(str)
    y = counts_df["order_count"]

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
        "mae_orders": round(float(mae), 2),
        "mean_actual_orders": round(float(y_test.mean()), 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return pipeline, metrics


def save_demand_model(model, metrics: dict, models_dir: str = MODELS_DIR) -> None:
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "demand_model.pkl")
    metrics_path = os.path.join(models_dir, "demand_metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved demand model to: {model_path}")
    print(f"Saved demand metrics to: {metrics_path}")


def run_pipeline():
    print("=== ShipSense Demand Forecasting — Training ===\n")

    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {len(df)} rows.\n")

    print("Building demand (region x hour x day-of-week) dataset...")
    counts_df = build_demand_dataset(df)
    print(f"Built {len(counts_df)} region/hour/day-of-week combinations.\n")

    print("Training demand forecasting model...")
    model, metrics = train_demand_model(counts_df)
    print(f"MAE: {metrics['mae_orders']} orders "
          f"(mean actual: {metrics['mean_actual_orders']} orders)\n")

    save_demand_model(model, metrics)

    # Also save the full forecast grid (every region x hour x day-of-week
    # combination, predicted) for the dashboard to read directly --
    # matches the existing "precompute, don't infer live" pattern.
    all_regions = df["region_id"].unique()
    grid = pd.MultiIndex.from_product(
        [all_regions, range(24), range(7), [True, False]],
        names=["region_id", "hour_of_day", "day_of_week", "is_weekend"]
    ).to_frame(index=False)

    X_grid = grid[DEMAND_FEATURES].copy()
    X_grid["is_weekend"] = X_grid["is_weekend"].astype(str)
    grid["predicted_orders"] = model.predict(X_grid).round(1)

    forecast_path = os.path.join("data", "processed", "demand_forecast.csv")
    grid.to_csv(forecast_path, index=False)
    print(f"Saved full forecast grid to: {forecast_path}")
    print(f"Grid shape: {grid.shape}")

    return model, metrics


if __name__ == "__main__":
    run_pipeline()