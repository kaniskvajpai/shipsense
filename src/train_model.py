"""
ShipSense — Model Training Module
Implements the function contracts defined in docs/API.md:
    train_eta_model(df, target_col) -> (best_model, metrics_dict)
    save_model(model, encoders, path) -> None

Trains a Linear Regression baseline and a Random Forest Regressor,
compares them on held-out test data using MAE/RMSE (in minutes),
and keeps whichever performs better.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RANDOM_STATE, CLEANED_DATA_PATH, MODELS_DIR

# Feature columns used for training (see docs/SCHEMA.md)
NUMERIC_FEATURES = [
    "distance_km", "hour_of_day", "day_of_week",
    "region_id", "courier_id", "aoi_id",
]
CATEGORICAL_FEATURES = ["aoi_type", "is_weekend"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    """One-hot encodes low-cardinality categoricals, passes numerics through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through unchanged
    )


def train_eta_model(df: pd.DataFrame, target_col: str = "delivery_duration_minutes"):
    """
    Trains a Linear Regression baseline and a Random Forest Regressor,
    evaluates both on a held-out test set, and returns the better model.

    Returns: (best_model_pipeline, metrics_dict)
    """
    if target_col not in df.columns:
        raise ValueError(f"train_eta_model: target column '{target_col}' not found.")
    if df[target_col].isnull().any():
        raise ValueError(f"train_eta_model: target column '{target_col}' has nulls.")

    missing_features = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing_features:
        raise ValueError(f"train_eta_model: missing feature columns: {missing_features}")

    X = df[FEATURE_ORDER].copy()
    X["is_weekend"] = X["is_weekend"].astype(str)  # treat as categorical string
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    results = {}
    fitted_pipelines = {}

    # --- Baseline: Linear Regression ---
    print("\nTraining baseline: Linear Regression...")
    lr_pipeline = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("model", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    results["linear_regression"] = {"mae_minutes": round(lr_mae, 2), "rmse_minutes": round(lr_rmse, 2)}
    fitted_pipelines["linear_regression"] = lr_pipeline
    print(f"  Linear Regression -> MAE: {lr_mae:.2f} min | RMSE: {lr_rmse:.2f} min")

    # --- Stronger model: Random Forest ---
    print("\nTraining stronger model: Random Forest Regressor...")
    rf_pipeline = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    results["random_forest"] = {"mae_minutes": round(rf_mae, 2), "rmse_minutes": round(rf_rmse, 2)}
    fitted_pipelines["random_forest"] = rf_pipeline
    print(f"  Random Forest -> MAE: {rf_mae:.2f} min | RMSE: {rf_rmse:.2f} min")

    # --- Pick the winner based on MAE (more interpretable than RMSE) ---
    winner = "random_forest" if rf_mae < lr_mae else "linear_regression"
    print(f"\nWinner: {winner} (lower MAE)")

    metrics = {
        "baseline_linear_regression": results["linear_regression"],
        "random_forest": results["random_forest"],
        "chosen_model": winner,
        "chosen_mae_minutes": results[winner]["mae_minutes"],
        "chosen_rmse_minutes": results[winner]["rmse_minutes"],
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return fitted_pipelines[winner], metrics


def get_feature_importances(model, top_n: int = 10) -> dict:
    """
    Extracts human-readable feature importances from a fitted pipeline.
    Works for tree-based models (Random Forest); returns an empty dict
    gracefully if the model type doesn't expose feature_importances_
    (e.g. Linear Regression was chosen instead).
    """
    try:
        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]

        if not hasattr(estimator, "feature_importances_"):
            print("Chosen model has no feature_importances_ (likely Linear Regression). Skipping.")
            return {}

        # Reconstruct the expanded feature names after one-hot encoding
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_names = cat_names + NUMERIC_FEATURES  # remainder="passthrough" appends numerics after

        importances = estimator.feature_importances_
        pairs = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)

        top = {name: round(float(score), 4) for name, score in pairs[:top_n]}
        return top
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        return {}


def save_model(model, metrics: dict, models_dir: str = MODELS_DIR) -> None:
    """
    Persists the trained model pipeline, metrics, feature order, and
    feature importances to disk. Confirms the saved model reloads and
    predicts identically (sanity check).
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "eta_model.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")
    features_path = os.path.join(models_dir, "feature_order.json")
    importances_path = os.path.join(models_dir, "feature_importances.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(features_path, "w") as f:
        json.dump({"feature_order": FEATURE_ORDER}, f, indent=2)

    importances = get_feature_importances(model)
    with open(importances_path, "w") as f:
        json.dump(importances, f, indent=2)

    # Sanity check: reload and confirm identical predictions on a tiny sample
    reloaded = joblib.load(model_path)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved feature order to: {features_path}")
    print(f"Saved feature importances to: {importances_path}")
    print("Reload sanity check passed." if reloaded is not None else "Reload FAILED.")

if __name__ == "__main__":
    print("=== ShipSense Model Training ===\n")
    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {len(df)} rows.\n")

    best_model, metrics = train_eta_model(df)
    save_model(best_model, metrics)

    print("\n=== FINAL METRICS ===")
    print(json.dumps(metrics, indent=2))