"""
ShipSense - Module 6 (fixed): compact live ETA model.

Changes vs v1:
  1. Trains on log1p(duration) -> expm1 at predict time. Stops the model
     collapsing every prediction toward the regional mean (~3-4 hours).
  2. region_id / aoi_type / is_weekend are fitted as true categories.
     NOTE: aoi_id (6,731 unique) and courier_id exceed sklearn's 255-cardinality
     limit for native categorical support in HistGradientBoostingRegressor, so
     they stay numeric (the booster can still split on their values).
  3. Trains p10 / p50 / p90 models so the app shows an ETA window
     instead of one misleading point estimate.

Run with the app environment (venv, scikit-learn 1.5.2 pinned):
    python src/train_live_eta.py

Writes:
    models/eta_live_model.pkl         (bundle: dict with models + cat_maps)
    models/eta_live_metrics.json
    data/processed/eta_defaults.json
Does NOT touch eta_model.pkl (the main batch model).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DELAY_RISK_THRESHOLD_MINUTES, RANDOM_STATE  # noqa: E402
from src.train_model import FEATURE_ORDER  # noqa: E402

DATA = ROOT / "data" / "processed" / "cleaned_data.csv"
MODELS = ROOT / "models"
OUT_MODEL = MODELS / "eta_live_model.pkl"
OUT_METRICS = MODELS / "eta_live_metrics.json"
OUT_DEFAULTS = ROOT / "data" / "processed" / "eta_defaults.json"
TARGET = "delivery_duration_minutes"
REQUIRED_SKLEARN = "1.5.2"

# Only low-cardinality columns can be native categories in
# HistGradientBoostingRegressor (hard limit: 255 unique values).
CAT_FEATURES = ["region_id", "aoi_type", "is_weekend"]
MAX_CARDINALITY = 255

PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
              min_samples_leaf=50)
TOP_N = 10


def main():
    print("scikit-learn version:", sklearn.__version__)
    if sklearn.__version__ != REQUIRED_SKLEARN:
        print("STOP: the deployed app uses scikit-learn %s. "
              "Activate venv and run again." % REQUIRED_SKLEARN)
        return

    df = pd.read_csv(DATA)
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET].astype(float)

    # Fix 2: only genuinely low-cardinality columns become categories.
    # aoi_id / courier_id stay numeric (they exceed the 255-value limit).
    cat_features = []
    for c in CAT_FEATURES:
        n_unique = X[c].nunique()
        if n_unique <= MAX_CARDINALITY:
            X[c] = X[c].astype("category")
            cat_features.append(c)
        else:
            print("Note: '%s' has %d unique values (> %d); keeping numeric."
                  % (c, n_unique, MAX_CARDINALITY))
    cat_maps = {c: list(X[c].cat.categories) for c in cat_features}
    print("Categorical features:", cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    # Fix 1: log-transform the skewed target (predict on log scale).
    y_train_log = np.log1p(y_train)

    # Fix 3: median + band models (p10 / p50 / p90).
    models = {}
    for name, kw in [
        ("p50", dict(loss="squared_error")),
        ("p10", dict(loss="quantile", quantile=0.10)),
        ("p90", dict(loss="quantile", quantile=0.90)),
    ]:
        m = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            categorical_features="from_dtype",
            **PARAMS, **kw,
        )
        m.fit(X_train, y_train_log)
        models[name] = m

    pred_p50 = np.expm1(models["p50"].predict(X_test))
    pred_p10 = np.expm1(models["p10"].predict(X_test))
    pred_p90 = np.expm1(models["p90"].predict(X_test))
    # keep the band monotone
    pred_p10 = np.minimum(pred_p10, pred_p50)
    pred_p90 = np.maximum(pred_p90, pred_p50)

    mae_p50 = float(mean_absolute_error(y_test, pred_p50))
    in_band = float(((y_test >= pred_p10) & (y_test <= pred_p90)).mean())
    print("Live model p50 MAE: %.2f min | p10-p90 coverage: %.1f%%"
          % (mae_p50, 100 * in_band))

    joblib.dump(
        {
            "models": models,
            "cat_maps": cat_maps,
            "feature_order": list(FEATURE_ORDER),
        },
        OUT_MODEL,
        compress=3,
    )
    size_mb = OUT_MODEL.stat().st_size / 1e6
    print("Wrote %s (%.2f MB)" % (OUT_MODEL, size_mb))

    # promised ETA + At Risk rule unchanged (same as src/predict.py)
    region_p75 = df.groupby("region_id")[TARGET].quantile(0.75)
    global_p75 = float(df[TARGET].quantile(0.75))
    promised = X_test["region_id"].astype(str).map(
        {str(k): v for k, v in region_p75.items()}
    ).fillna(global_p75)
    risk_live = (pred_p50 - promised.values) > DELAY_RISK_THRESHOLD_MINUTES

    metrics = {
        "model": "HistGradientBoostingRegressor (log1p target, categorical "
                 "region_id/aoi_type/is_weekend)",
        "params": PARAMS,
        "sklearn_version": sklearn.__version__,
        "feature_order": list(FEATURE_ORDER),
        "categorical_features": cat_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_minutes": round(mae_p50, 2),
        "p10_p90_coverage": round(in_band, 4),
        "at_risk_share_test": round(float(risk_live.mean()), 4),
        "file_mb": round(size_mb, 2),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))

    # region defaults (same shape as v1, unchanged consumers)
    regions = {}
    for rid, g in df.groupby("region_id"):
        couriers = g["courier_id"].value_counts().head(TOP_N).index.tolist()
        aois = (
            g.groupby(["aoi_id", "aoi_type"])
            .size()
            .sort_values(ascending=False)
            .head(TOP_N)
            .reset_index()[["aoi_id", "aoi_type"]]
            .values.tolist()
        )
        regions[str(int(rid))] = {
            "promised_eta_minutes": round(float(region_p75[rid]), 1),
            "median_distance_km": round(float(g["distance_km"].median()), 2),
            "p90_distance_km": round(float(g["distance_km"].quantile(0.9)), 2),
            "orders": int(len(g)),
            "couriers": [int(c) for c in couriers],
            "aois": [[int(a), int(t)] for a, t in aois],
        }
    OUT_DEFAULTS.write_text(json.dumps({
        "threshold_minutes": int(DELAY_RISK_THRESHOLD_MINUTES),
        "global_promised_eta_minutes": round(global_p75, 1),
        "regions": regions,
    }, indent=2))
    print("Wrote", OUT_METRICS)
    print("Wrote", OUT_DEFAULTS)


if __name__ == "__main__":
    main()