"""
backtest_1x2.py
Financial simulator for the 1X2 Betting System.
Calculates Expected Value (EV) and simulates a flat-staking strategy on the test set.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def simulate_betting(
    test_df: pd.DataFrame, clf: RandomForestClassifier, X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulates betting based on model probabilities and bookmaker odds.
    """
    STAKE = 10.0  # Flat bet of 10 units per wager

    # Extract probabilities for each class
    # clf.classes_ usually returns ['A', 'D', 'H'], we map them dynamically
    probs = clf.predict_proba(X_test)
    class_mapping = {class_name: idx for idx, class_name in enumerate(clf.classes_)}

    # Attach predictions and odds to a new analysis dataframe
    results = test_df[
        ["Date", "HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]
    ].copy()

    # Extract model probabilities
    results["Prob_H"] = probs[:, class_mapping.get("H", -1)]
    results["Prob_D"] = probs[:, class_mapping.get("D", -1)]
    results["Prob_A"] = probs[:, class_mapping.get("A", -1)]

    # Calculate Expected Value (EV) for each outcome
    results["EV_H"] = (results["Prob_H"] * results["B365H"]) - 1
    results["EV_D"] = (results["Prob_D"] * results["B365D"]) - 1
    results["EV_A"] = (results["Prob_A"] * results["B365A"]) - 1

    # Betting Logic: Find the highest EV. If it's > 0, place a bet.
    bets_placed = []
    profits = []

    for _, row in results.iterrows():
        # Dictionary of EV for this match
        ev_dict = {"H": row["EV_H"], "D": row["EV_D"], "A": row["EV_A"]}
        odds_dict = {"H": row["B365H"], "D": row["B365D"], "A": row["B365A"]}

        # Identify the outcome with the maximum EV
        best_bet = max(ev_dict, key=ev_dict.get)
        max_ev = ev_dict[best_bet]

        if max_ev > 0:  # Value Bet found
            bets_placed.append(best_bet)

            # Check if the bet won
            if best_bet == row["FTR"]:
                # Profit = Return - Stake
                profits.append((STAKE * odds_dict[best_bet]) - STAKE)
            else:
                # Loss = -Stake
                profits.append(-STAKE)
        else:
            # No Value Bet found, pass
            bets_placed.append("No Bet")
            profits.append(0.0)

    results["Bet_Placed"] = bets_placed
    results["Profit"] = profits

    return results


def main():
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")

    if not PROCESSED_DATA_PATH.exists():
        print(f"[ERROR] Processed dataset not found at {PROCESSED_DATA_PATH}")
        return

    print("[INFO] Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 1. Prepare Data
    feature_cols = [col for col in df.columns if "rolling" in col]
    X = df[feature_cols]
    y = df["FTR"]

    # 2. Temporal Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_df = df.iloc[split_index:].copy()

    # 3. Train Model
    print("[INFO] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 4. Execute Backtest
    print("[INFO] Running Financial Simulation on Test Set...")
    bt_results = simulate_betting(test_df, clf, X_test)

    # 5. Calculate Financial Metrics
    total_bets = len(bt_results[bt_results["Bet_Placed"] != "No Bet"])
    total_staked = total_bets * 10.0
    total_profit = bt_results["Profit"].sum()

    # Avoid division by zero
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0

    print("\n================ FINANCIAL REPORT ================")
    print(f"Total Matches Analyzed: {len(test_df)}")
    print(f"Total Bets Placed:      {total_bets}")
    print(f"Total Staked:           €{total_staked:.2f}")
    print(f"Net Profit/Loss:        €{total_profit:.2f}")
    print(f"ROI (Yield):            {roi:.2f}%")
    print("==================================================")

    # Optional: Save the detailed betting log for inspection
    log_path = Path("data/processed/backtest_log.csv")
    bt_results.to_csv(log_path, index=False)
    print(f"\n[INFO] Detailed betting log saved to {log_path}")


if __name__ == "__main__":
    main()
