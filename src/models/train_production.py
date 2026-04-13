"""
train_production.py
Trains the final Calibrated Binary Random Forest model on 100% of data.
Calibration is strictly required before applying Kelly Criterion staking logic.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def train_and_export_calibrated_model():
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")
    EXPORT_DIR = Path("models")
    MODEL_OUTPUT_PATH = EXPORT_DIR / "rf_uo_production_v1.joblib"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATA_PATH.exists():
        print(f"[ERROR] Processed dataset not found at {PROCESSED_DATA_PATH}")
        return

    print(
        f"[INFO] Loading processed data for production training from {PROCESSED_DATA_PATH}..."
    )
    df = pd.read_csv(PROCESSED_DATA_PATH)

    feature_cols = [col for col in df.columns if "rolling" in col]
    X = df[feature_cols]
    y = df["Over2.5"]

    print(
        f"[INFO] Training and Calibrating final model on 100% of data ({len(df)} matches)..."
    )

    # 1. Base Model
    base_clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )

    # 2. Calibrated wrapper (Isotonic Regression)
    # CV=5 means it splits the data 5 times internally to ensure the calibration is robust
    calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated_clf.fit(X, y)

    # 3. Export
    print("[INFO] Serializing calibrated model to disk...")
    joblib.dump(calibrated_clf, MODEL_OUTPUT_PATH)
    print(
        f"[SUCCESS] Calibrated production model exported successfully to {MODEL_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    train_and_export_calibrated_model()
