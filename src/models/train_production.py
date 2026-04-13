"""
train_production.py
Trains the final Binary Random Forest model on 100% of the available data
for the U/O 2.5 goals market and serializes it for real-time inference.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_and_export_model():
    """
    Loads processed features, trains the final production model without splitting,
    and exports it to disk using joblib.
    """
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")
    EXPORT_DIR = Path("models")
    MODEL_OUTPUT_PATH = EXPORT_DIR / "rf_uo_production_v1.joblib"

    # Ensure export directory exists
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATA_PATH.exists():
        print(f"[ERROR] Processed dataset not found at {PROCESSED_DATA_PATH}")
        return

    print(
        f"[INFO] Loading processed data for production training from {PROCESSED_DATA_PATH}..."
    )
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 1. Define Features and Target
    feature_cols = [col for col in df.columns if "rolling" in col]
    X = df[feature_cols]
    y = df["Over2.5"]

    print(f"[INFO] Training final model on 100% of data ({len(df)} matches)...")

    # 2. Initialize and Train the Model (Same hyper-parameters as the successful backtest)
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )

    clf.fit(X, y)

    # 3. Export the Model (Serialization)
    print("[INFO] Serializing model to disk...")
    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print(f"[SUCCESS] Production model exported successfully to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_export_model()
