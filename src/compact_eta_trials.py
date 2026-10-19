"""
ShipSense - Module 6 feasibility test (round 2): compact ETA models.

Run with the app environment (venv), which must have the same scikit-learn
version as the deployed app (1.5.2):
    python src/compact_eta_trials.py

Trains five compact variants on the same train/test split as the main ETA model
and prints test error (MAE) and file size for each. It also scores the existing
full model on the same split as a sanity check. Writes models/eta_trial_1.pkl ...
eta_trial_5.pkl (git ignores models/*.pkl, so nothing is uploaded). Does NOT
touch eta_model.pkl.
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE  # noqa: E402
from src.train_model import FEATURE_ORDER  # noqa: E402

DATA = ROOT / "data" / "processed" / "cleaned_data.csv"
MODELS = ROOT / "models"
TARGET = "delivery_duration_minutes"
DEPLOYED_SKLEARN = "1.5.2"
CATEGORICAL = ["region_id", "courier_id", "aoi_id", "aoi_type"]


def raw_forest(**params):
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)


def onehot_forest(**params):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("model", raw_forest(**params))])


TRIALS = [
    ("1 raw forest: 40 trees, depth 12, leaf 10",
     lambda: raw_forest(n_estimators=40, max_depth=12, min_samples_leaf=10)),
    ("2 raw forest: 60 trees, depth 14, leaf 5",
     lambda: raw_forest(n_estimators=60, max_depth=14, min_samples_leaf=5)),
    ("3 one-hot forest: 40 trees, depth 14, leaf 5",
     lambda: onehot_forest(n_estimators=40, max_depth=14, min_samples_leaf=5)),
    ("4 one-hot forest: 30 trees, depth 12, leaf 10",
     lambda: onehot_forest(n_estimators=30, max_depth=12, min_samples_leaf=10)),
    ("5 gradient boosting: 300 rounds",
     lambda: HistGradientBoostingRegressor(
         max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
         random_state=RANDOM_STATE)),
]


def main():
    print("scikit-learn version in this environment:", sklearn.__version__)
    if sklearn.__version__ != DEPLOYED_SKLEARN:
        print("WARNING: the deployed app uses %s. Activate venv and run again." %
              DEPLOYED_SKLEARN)
    df = pd.read_csv(DATA)
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    full_path = MODELS / "eta_model.pkl"
    try:
        full = joblib.load(full_path)
        mae_full = mean_absolute_error(y_test, full.predict(X_test))
        print("Full model: MAE %.2f min, file %.1f MB" % (
            mae_full, full_path.stat().st_size / 1e6))
        del full
    except Exception as exc:
        print("Could not score the full model:", exc)

    for i, (name, build) in enumerate(TRIALS, start=1):
        start = time.time()
        model = build()
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        path = MODELS / ("eta_trial_%d.pkl" % i)
        joblib.dump(model, path, compress=3)
        print("trial %s | MAE %.2f min | file %.1f MB | %.0f s" % (
            name, mae, path.stat().st_size / 1e6, time.time() - start))

    print("Done. The full model file was not changed.")


if __name__ == "__main__":
    main()