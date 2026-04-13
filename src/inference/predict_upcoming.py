"""
predict_upcoming.py
Production Inference Pipeline with Fractional Kelly Criterion.
Recommends exact staking amounts based on Calibrated Probabilities.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

TEAM_MAPPING = {
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Atalanta BC": "Atalanta",
}


def calculate_kelly_stake(
    prob: float, decimal_odds: float, bankroll: float, fraction: float = 0.25
) -> float:
    """
    Calculates the recommended bet size using the Fractional Kelly Criterion.
    """
    b = decimal_odds - 1.0
    p = prob
    q = 1.0 - p

    kelly_pct = (p * b - q) / b

    if kelly_pct <= 0:
        return 0.0

    return bankroll * kelly_pct * fraction


def get_latest_team_stats(team_name: str, df: pd.DataFrame, is_home: bool) -> dict:
    team_matches = df[
        (df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)
    ].copy()
    if team_matches.empty:
        raise ValueError(f"Team '{team_name}' not found in historical data.")

    last_match = team_matches.sort_values("Date").iloc[-1]

    if last_match["HomeTeam"] == team_name:
        stats = {
            "rolling_GoalsScored_sum_5": last_match["Home_rolling_GoalsScored_sum_5"],
            "rolling_GoalsConceded_sum_5": last_match[
                "Home_rolling_GoalsConceded_sum_5"
            ],
        }
    else:
        stats = {
            "rolling_GoalsScored_sum_5": last_match["Away_rolling_GoalsScored_sum_5"],
            "rolling_GoalsConceded_sum_5": last_match[
                "Away_rolling_GoalsConceded_sum_5"
            ],
        }

    prefix = "Home_" if is_home else "Away_"
    return {
        f"{prefix}rolling_GoalsScored_sum_5": stats["rolling_GoalsScored_sum_5"],
        f"{prefix}rolling_GoalsConceded_sum_5": stats["rolling_GoalsConceded_sum_5"],
    }


def run_inference():
    CACHE_FILE = Path("data/interim/live_odds_cache.json")
    PROCESSED_DATA_PATH = Path("data/processed/Serie_A_features.csv")
    MODEL_PATH = Path("models/rf_uo_production_v1.joblib")

    # Financial Configuration
    BANKROLL = 1000.00
    EV_THRESHOLD = 0.05
    KELLY_FRACTION = 0.25  # Quarter Kelly

    if (
        not CACHE_FILE.exists()
        or not PROCESSED_DATA_PATH.exists()
        or not MODEL_PATH.exists()
    ):
        print("[ERROR] Missing required files. Cannot run inference.")
        return

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        upcoming_matches = json.load(f)

    hist_df = pd.read_csv(PROCESSED_DATA_PATH)
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])

    clf = joblib.load(MODEL_PATH)
    class_mapping = {class_name: idx for idx, class_name in enumerate(clf.classes_)}

    print("\n================= VALUE BETTING SCANNER =================")
    print(
        f"Bankroll: €{BANKROLL:.2f} | Strategy: Quarter-Kelly | Min EV: {EV_THRESHOLD * 100}%\n"
    )

    for match in upcoming_matches:
        raw_home = match["HomeTeam_API"]
        raw_away = match["AwayTeam_API"]
        home_team = TEAM_MAPPING.get(raw_home, raw_home)
        away_team = TEAM_MAPPING.get(raw_away, raw_away)

        try:
            home_stats = get_latest_team_stats(home_team, hist_df, is_home=True)
            away_stats = get_latest_team_stats(away_team, hist_df, is_home=False)
        except ValueError as e:
            print(f"[WARNING] Skipping {home_team} vs {away_team}: {e}")
            continue

        feature_dict = {**home_stats, **away_stats}
        X_infer = pd.DataFrame([feature_dict])

        expected_cols = [
            "Home_rolling_GoalsScored_sum_5",
            "Home_rolling_GoalsConceded_sum_5",
            "Away_rolling_GoalsScored_sum_5",
            "Away_rolling_GoalsConceded_sum_5",
        ]
        X_infer = X_infer[expected_cols]

        probs = clf.predict_proba(X_infer)[0]
        prob_under = probs[class_mapping.get(0)]
        prob_over = probs[class_mapping.get(1)]

        odds_under = match["Odds_Under_25"]
        odds_over = match["Odds_Over_25"]

        ev_under = (prob_under * odds_under) - 1
        ev_over = (prob_over * odds_over) - 1

        best_bet = None
        if ev_under > EV_THRESHOLD and ev_under > ev_over:
            best_bet = "Under 2.5"
            selected_ev = ev_under
            selected_prob = prob_under
            selected_odds = odds_under
        elif ev_over > EV_THRESHOLD and ev_over > ev_under:
            best_bet = "Over 2.5"
            selected_ev = ev_over
            selected_prob = prob_over
            selected_odds = odds_over

        print(
            f"Match: {home_team} vs {away_team} (Pinnacle Odds: O:{odds_over} / U:{odds_under})"
        )
        print(
            f"       -> Calibrated Probabilities: Over {prob_over * 100:.1f}% | Under {prob_under * 100:.1f}%"
        )

        if best_bet:
            recommended_stake = calculate_kelly_stake(
                selected_prob, selected_odds, BANKROLL, KELLY_FRACTION
            )
            print(f"       💰 ACTION REQUIRED: Bet {best_bet} @ {selected_odds}")
            print(f"       📈 Expected Value (EV): +{selected_ev * 100:.2f}%")
            print(f"       🎯 Recommended Stake: €{recommended_stake:.2f}")
        else:
            print("       ⏸️  PASS (No Value Found)")
        print("-" * 57)


if __name__ == "__main__":
    run_inference()
