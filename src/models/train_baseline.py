"""
train_baseline.py
Trains a baseline Random Forest classifier to predict match outcomes (H, D, A).
Uses strict Temporal Splitting to avoid data leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate_model(processed_data_path: Path):
    """
    Loads features, performs temporal split, trains a Random Forest,
    and outputs evaluation metrics.
    """
    if not processed_data_path.exists():
        print(f"[ERROR] Processed dataset not found at {processed_data_path}")
        return

    print(f"[INFO] Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)

    # 1. Define Features (X) and Target (y)
    # Automatically select all columns containing 'rolling'
    feature_cols = [col for col in df.columns if "rolling" in col]
    X = df[feature_cols]
    y = df["FTR"]  # Full Time Result: H, D, A

    print(f"[INFO] Features used for prediction: {len(feature_cols)}")
    print(f"[INFO] Target classes: {y.unique()}")

    # 2. Temporal Split (No Randomness)
    # Train on the first 80% (chronologically), Test on the final 20%
    split_index = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("\n[INFO] Temporal Split Applied:")
    print(f"       Train set size: {len(X_train)} matches (Past)")
    print(f"       Test set size: {len(X_test)} matches (Future)")

    # 3. Initialize and Train the Model
    # Random Forest is highly suitable for non-linear tabular data
    print("\n[INFO] Training Baseline Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Constrained depth to prevent extreme overfitting
        random_state=42,
        class_weight="balanced",  # Helps penalize errors on less frequent classes (like Draws)
    )

    clf.fit(X_train, y_train)

    # 4. Predict and Evaluate
    print("[INFO] Evaluating on Test set (Future unseen data)...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n================ MODEL EVALUATION ================")
    print(f"Baseline Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("==================================================")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")
    train_and_evaluate_model(PROCESSED_DATA_PATH)
