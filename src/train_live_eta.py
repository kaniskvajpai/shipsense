"""
ShipSense - Module 6: train the compact live ETA model.

Run with the app environment (venv); it must have scikit-learn 1.5.2, the same
version the deployed app pins:
    python src/train_live_eta.py

Writes:
    models/eta_live_model.pkl         (about 1 MB, safe to upload to GitHub)
    models/eta_live_metrics.json
    data/processed/eta_defaults.json  (region defaults for the Live ETA tab)
Does NOT touch eta_model.pkl.

The promised ETA and the At Risk rule are the same as in src/predict.py:
promised = 75th percentile of delivery time per region; At Risk when the
predicted ETA exceeds the promised ETA by more than DELAY_RISK_THRESHOLD_MINUTES.
"""
import json
import sys
from pathlib import Path

import joblib
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
PARAMS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
TOP_N = 10


def main():
    print("scikit-learn version:", sklearn.__version__)
    if sklearn.__version__ != REQUIRED_SKLEARN:
        print("STOP: the deployed app uses scikit-learn %s. "
              "Activate venv and run again." % REQUIRED_SKLEARN)
        return

    df = pd.read_csv(DATA)
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **PARAMS)
    model.fit(X_train, y_train)
    pred_live = model.predict(X_test)
    mae_live = float(mean_absolute_error(y_test, pred_live))
    joblib.dump(model, OUT_MODEL, compress=3)
    size_mb = OUT_MODEL.stat().st_size / 1e6
    print("Live model: MAE %.2f min, file %.2f MB" % (mae_live, size_mb))

    # same promised-ETA rule as src/predict.py
    region_p75 = df.groupby("region_id")[TARGET].quantile(0.75)
    global_p75 = float(df[TARGET].quantile(0.75))
    promised = X_test["region_id"].map(region_p75).fillna(global_p75)
    risk_live = (pred_live - promised) > DELAY_RISK_THRESHOLD_MINUTES

    # sanity check against the full model on the same rows
    mae_full = None
    agreement = None
    try:
        full = joblib.load(MODELS / "eta_model.pkl")
        pred_full = full.predict(X_test)
        mae_full = float(mean_absolute_error(y_test, pred_full))
        risk_full = (pred_full - promised) > DELAY_RISK_THRESHOLD_MINUTES
        agreement = float((risk_live.values == risk_full.values).mean())
        print("Full model: MAE %.2f min" % mae_full)
        print("At Risk share on the test rows: live %.1f%%, full %.1f%%" % (
            100.0 * float(risk_live.mean()), 100.0 * float(risk_full.mean())))
        print("At Risk label agreement, live vs full model: %.1f%%" % (
            100.0 * agreement))
        del full
    except Exception as exc:
        print("Could not score the full model:", exc)

    metrics = {
        "model": "HistGradientBoostingRegressor",
        "params": PARAMS,
        "sklearn_version": sklearn.__version__,
        "feature_order": list(FEATURE_ORDER),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_minutes": round(mae_live, 2),
        "full_model_mae_minutes": None if mae_full is None else round(mae_full, 2),
        "at_risk_agreement_with_full_model": (
            None if agreement is None else round(agreement, 4)
        ),
        "file_mb": round(size_mb, 2),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))

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
    defaults = {
        "threshold_minutes": int(DELAY_RISK_THRESHOLD_MINUTES),
        "global_promised_eta_minutes": round(global_p75, 1),
        "regions": regions,
    }
    OUT_DEFAULTS.write_text(json.dumps(defaults, indent=2))

    print("Regions in the defaults file:", len(regions))
    print("Wrote", OUT_MODEL)
    print("Wrote", OUT_METRICS)
    print("Wrote", OUT_DEFAULTS)


if __name__ == "__main__":
    main()