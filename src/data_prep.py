"""
ShipSense — Data Preparation Module
Implements the exact function contracts defined in docs/API.md:
    load_raw_data(path) -> pd.DataFrame
    clean_data(df) -> pd.DataFrame
    engineer_features(df) -> pd.DataFrame

Source dataset: Cainiao-AI/LaDe-D, Yantai city (smallest city subset).
"""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# A fixed placeholder year is used to parse the "MM-DD HH:MM:SS" timestamps,
# which have no year in the source data. This does NOT affect hour_of_day
# (always correct), and produces internally-consistent day_of_week /
# is_weekend values since the whole dataset shares the same implied year.
# It should NOT be treated as the true calendar date.
PLACEHOLDER_YEAR = 2023


def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Loads the raw Yantai delivery dataset.
    If `path` is None, downloads it fresh from Hugging Face into the
    local HF cache and loads it from there.
    Raises FileNotFoundError / ValueError per docs/API.md.
    """
    if path is None:
        path = hf_hub_download(
            repo_id="Cainiao-AI/LaDe-D",
            repo_type="dataset",
            filename="data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet",
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("Loaded raw dataset has zero rows.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset:
      - drops rows with nulls in critical columns
      - parses timestamps
      - computes delivery_duration_minutes (target variable)
      - removes outlier/invalid durations
    Resilient to messy rows (drops + logs), only raises if nothing is left.
    """
    df = df.copy()
    start_rows = len(df)

    # --- 1. Drop rows with nulls in critical columns ---
    critical_cols = [
        "order_id", "accept_time", "delivery_time",
        "accept_gps_lng", "accept_gps_lat",
        "delivery_gps_lng", "delivery_gps_lat",
    ]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_nulls = before - len(df)
    print(f"[clean_data] Dropped {dropped_nulls} rows with nulls in critical columns.")

    # --- 2. Parse timestamps (MM-DD HH:MM:SS -> full datetime, placeholder year) ---
    def parse_ts(series: pd.Series) -> pd.Series:
        full_str = f"{PLACEHOLDER_YEAR}-" + series.astype(str)
        return pd.to_datetime(full_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    df["accept_dt"] = parse_ts(df["accept_time"])
    df["delivery_dt"] = parse_ts(df["delivery_time"])

    before = len(df)
    df = df.dropna(subset=["accept_dt", "delivery_dt"])
    dropped_bad_ts = before - len(df)
    print(f"[clean_data] Dropped {dropped_bad_ts} rows with unparseable timestamps.")

    # --- 3. Compute target variable: delivery_duration_minutes ---
    duration = (df["delivery_dt"] - df["accept_dt"]).dt.total_seconds() / 60.0

    # Handle deliveries that cross midnight (accept late night, deliver early
    # morning) by adding 24h when duration is negative.
    duration = np.where(duration < 0, duration + 24 * 60, duration)
    df["delivery_duration_minutes"] = duration

    # --- 4. Remove outliers using IQR on the target variable ---
    q1 = df["delivery_duration_minutes"].quantile(0.25)
    q3 = df["delivery_duration_minutes"].quantile(0.75)
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr

    before = len(df)
    df = df[
        (df["delivery_duration_minutes"] > 0)
        & (df["delivery_duration_minutes"] >= lower)
        & (df["delivery_duration_minutes"] <= upper)
    ]
    dropped_outliers = before - len(df)
    print(f"[clean_data] Dropped {dropped_outliers} outlier rows "
          f"(kept range: {lower:.1f}-{upper:.1f} min).")

    print(f"[clean_data] Final row count: {len(df)} (started with {start_rows}).")

    if df.empty:
        raise ValueError("clean_data produced an empty DataFrame — check source data.")

    return df


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance between two lat/lng points, in kilometers."""
    r = 6371.0  # Earth radius in km
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived feature columns per docs/SCHEMA.md:
      hour_of_day, day_of_week, is_weekend, distance_km
    Raises KeyError if a required source column is missing.
    """
    df = df.copy()

    required = ["accept_dt", "accept_gps_lat", "accept_gps_lng",
                "delivery_gps_lat", "delivery_gps_lng"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"engineer_features: missing required columns: {missing}")

    df["hour_of_day"] = df["accept_dt"].dt.hour
    df["day_of_week"] = df["accept_dt"].dt.dayofweek  # 0=Mon .. 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    df["distance_km"] = _haversine_km(
        df["accept_gps_lat"], df["accept_gps_lng"],
        df["delivery_gps_lat"], df["delivery_gps_lng"],
    )

    # Guard against zero-distance rows (same-point GPS glitch) which would
    # be a useless/degenerate training example.
    before = len(df)
    df = df[df["distance_km"] > 0.01]
    print(f"[engineer_features] Dropped {before - len(df)} rows with ~zero distance.")

    for col in ["hour_of_day", "day_of_week", "distance_km"]:
        if df[col].isnull().any():
            raise ValueError(f"engineer_features: nulls found in derived column '{col}'.")

    return df


def run_pipeline(output_path: str = "data/processed/cleaned_data.csv") -> pd.DataFrame:
    """Runs the full load -> clean -> engineer pipeline and saves the result."""
    print("=== ShipSense Data Preparation Pipeline ===\n")

    print("Step 1/3: Loading raw data...")
    raw_df = load_raw_data()
    print(f"  Loaded {len(raw_df)} raw rows.\n")

    print("Step 2/3: Cleaning data...")
    clean_df = clean_data(raw_df)
    print()

    print("Step 3/3: Engineering features...")
    final_df = engineer_features(clean_df)
    print()

    # Keep only the columns defined in docs/SCHEMA.md Section 2, plus IDs
    # useful for later joins/debugging.
    keep_cols = [
        "order_id", "city", "region_id", "courier_id", "aoi_id", "aoi_type",
        "accept_gps_lat", "accept_gps_lng",
        "delivery_gps_lat", "delivery_gps_lng",
        "distance_km", "hour_of_day", "day_of_week", "is_weekend",
        "delivery_duration_minutes",
    ]
    final_df = final_df[keep_cols]

    final_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Final shape: {final_df.shape}")

    return final_df


if __name__ == "__main__":
    run_pipeline()