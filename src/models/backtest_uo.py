"""
backtest_uo.py
Financial simulator for the Under/Over 2.5 Goals Market.
Trains a Random Forest for binary classification and simulates a flat-staking EV strategy with Thresholding.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def simulate_uo_betting(
    test_df: pd.DataFrame,
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    min_ev_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Simulates betting based on binary model probabilities and bookmaker U/O odds.
    Applies a strict Minimum Expected Value threshold to filter out marginal bets.
    """
    STAKE = 10.0

    probs = clf.predict_proba(X_test)
    class_mapping = {class_name: idx for idx, class_name in enumerate(clf.classes_)}

    results = test_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "TotalGoals",
            "Over2.5",
            "B365>2.5",
            "B365<2.5",
        ]
    ].copy()

    results["Prob_Under"] = probs[:, class_mapping.get(0)]
    results["Prob_Over"] = probs[:, class_mapping.get(1)]

    results["EV_Under"] = (results["Prob_Under"] * results["B365<2.5"]) - 1
    results["EV_Over"] = (results["Prob_Over"] * results["B365>2.5"]) - 1

    bets_placed = []
    profits = []

    for _, row in results.iterrows():
        ev_dict = {"Under": row["EV_Under"], "Over": row["EV_Over"]}
        odds_dict = {"Under": row["B365<2.5"], "Over": row["B365>2.5"]}

        best_bet = max(ev_dict, key=ev_dict.get)
        max_ev = ev_dict[best_bet]

        # CORE CHANGE: Apply the Margin of Safety
        if max_ev > min_ev_threshold:
            bets_placed.append(best_bet)

            won_under = (best_bet == "Under") and (row["Over2.5"] == 0)
            won_over = (best_bet == "Over") and (row["Over2.5"] == 1)

            if won_under or won_over:
                profits.append((STAKE * odds_dict[best_bet]) - STAKE)
            else:
                profits.append(-STAKE)
        else:
            bets_placed.append("No Bet")
            profits.append(0.0)

    results["Bet_Placed"] = bets_placed
    results["Profit"] = profits

    return results


def main():
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")
    EV_THRESHOLD = 0.05  # Require a minimum 5% mathematical advantage

    if not PROCESSED_DATA_PATH.exists():
        print(f"[ERROR] Processed dataset not found at {PROCESSED_DATA_PATH}")
        return

    print("[INFO] Loading U/O processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    feature_cols = [col for col in df.columns if "rolling" in col]
    X = df[feature_cols]
    y = df["Over2.5"]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_df = df.iloc[split_index:].copy()

    print("[INFO] Training Binary Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    print(
        f"[INFO] Running Financial Simulation (EV Threshold: {EV_THRESHOLD * 100}%)..."
    )
    bt_results = simulate_uo_betting(
        test_df, clf, X_test, min_ev_threshold=EV_THRESHOLD
    )

    total_bets = len(bt_results[bt_results["Bet_Placed"] != "No Bet"])
    total_staked = total_bets * 10.0
    total_profit = bt_results["Profit"].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0

    print("\n================ FINANCIAL REPORT (U/O) ================")
    print(f"Total Matches Analyzed: {len(test_df)}")
    print(f"Total Bets Placed:      {total_bets}")
    print(f"Total Staked:           €{total_staked:.2f}")
    print(f"Net Profit/Loss:        €{total_profit:.2f}")
    print(f"ROI (Yield):            {roi:.2f}%")
    print("========================================================")

    log_path = Path("data/processed/backtest_uo_log.csv")
    bt_results.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()
