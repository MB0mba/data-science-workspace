"""
dashboard.py
Streamlit Web Application for the Quant Betting System.
Provides a GUI to analyze upcoming Serie A matches and recommended Kelly stakes.
Calculates target Betfair Exchange odds including a 4.5% commission.
"""

import sys
from pathlib import Path

# --- Path Resolution Fix ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Suppress Ruff Linter warnings (E402) for the following imports
import json  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.inference.predict_upcoming import (  # noqa: E402
    TEAM_MAPPING,
    calculate_kelly_stake,
    get_latest_team_stats,
)

# --- Robust Caching (Version Agnostic) ---
try:
    cache_model = st.cache_resource
    cache_data = st.cache_data
except AttributeError:
    cache_model = st.cache(allow_output_mutation=True)
    cache_data = st.cache()


@cache_model
def load_model():
    """Loads the serialized machine learning model."""
    return joblib.load(Path("models/rf_uo_production_v1.joblib"))


@cache_data
def load_historical_data() -> pd.DataFrame:
    """Loads the processed feature matrix."""
    df = pd.read_csv(Path("data/processed/Serie_A_features.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@cache_data
def load_live_odds() -> list:
    """Loads the latest Pinnacle odds from the local JSON cache."""
    cache_file = Path("data/interim/live_odds_cache.json")
    if not cache_file.exists():
        return []
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Main Application ---
def main():
    st.set_page_config(page_title="Quant Betting Dashboard", layout="wide")

    st.title("📈 Quant Betting Dashboard - Serie A (U/O 2.5)")
    st.markdown("Automated Value Betting Scanner based on Calibrated Random Forest.")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Financial Settings")
    bankroll = st.sidebar.number_input(
        "Current Bankroll (€)", min_value=10.0, value=1000.0, step=50.0
    )
    ev_threshold = (
        st.sidebar.slider(
            "Min Expected Value (EV %)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
        )
        / 100.0
    )
    kelly_fraction = st.sidebar.selectbox(
        "Kelly Strategy",
        options=[0.10, 0.25, 0.50, 1.0],
        index=1,
        format_func=lambda x: f"{int(x * 100)}% Kelly",
    )

    # Fixed Betfair Commission for Italian Market
    BETFAIR_COMMISSION = 0.045

    # --- Load Components ---
    try:
        clf = load_model()
        hist_df = load_historical_data()
        upcoming_matches = load_live_odds()
    except Exception as e:
        st.error(f"System Initialization Error: {e}")
        return

    if not upcoming_matches:
        st.warning(
            "No matches available in the local cache. Please run the API fetcher first."
        )
        return

    class_mapping = {class_name: idx for idx, class_name in enumerate(clf.classes_)}

    # --- Prediction Engine ---
    results_data = []

    for match in upcoming_matches:
        raw_home = match["HomeTeam_API"]
        raw_away = match["AwayTeam_API"]
        home_team = TEAM_MAPPING.get(raw_home, raw_home)
        away_team = TEAM_MAPPING.get(raw_away, raw_away)

        try:
            home_stats = get_latest_team_stats(home_team, hist_df, is_home=True)
            away_stats = get_latest_team_stats(away_team, hist_df, is_home=False)
        except ValueError:
            continue

        feature_dict = {**home_stats, **away_stats}
        expected_cols = [
            "Home_rolling_GoalsScored_sum_5",
            "Home_rolling_GoalsConceded_sum_5",
            "Away_rolling_GoalsScored_sum_5",
            "Away_rolling_GoalsConceded_sum_5",
        ]
        X_infer = pd.DataFrame([feature_dict])[expected_cols]

        probs = clf.predict_proba(X_infer)[0]
        prob_under = probs[class_mapping.get(0)]
        prob_over = probs[class_mapping.get(1)]

        odds_under = match["Odds_Under_25"]
        odds_over = match["Odds_Over_25"]

        ev_under = (prob_under * odds_under) - 1
        ev_over = (prob_over * odds_over) - 1

        # Action Logic
        action = "PASS"
        target_odds = 0.0
        target_ev = 0.0
        stake = 0.0
        betfair_target = 0.0

        if ev_under > ev_threshold and ev_under > ev_over:
            action = "BET UNDER 2.5"
            target_odds = odds_under
            target_ev = ev_under
            stake = calculate_kelly_stake(
                prob_under, odds_under, bankroll, kelly_fraction
            )
            # Calculate required Betfair odds to match target EV
            betfair_target = ((target_odds - 1.0) / (1.0 - BETFAIR_COMMISSION)) + 1.0

        elif ev_over > ev_threshold and ev_over > ev_under:
            action = "BET OVER 2.5"
            target_odds = odds_over
            target_ev = ev_over
            stake = calculate_kelly_stake(
                prob_over, odds_over, bankroll, kelly_fraction
            )
            # Calculate required Betfair odds to match target EV
            betfair_target = ((target_odds - 1.0) / (1.0 - BETFAIR_COMMISSION)) + 1.0

        results_data.append(
            {
                "Match": f"{home_team} vs {away_team}",
                "Date (UTC)": match["CommenceTime"].replace("T", " ").replace("Z", ""),
                "Action": action,
                "Odds": target_odds if action != "PASS" else "-",
                "Betfair Target (4.5%)": f"{betfair_target:.3f}"
                if action != "PASS"
                else "-",
                "EV (%)": f"{target_ev * 100:.2f}%" if action != "PASS" else "-",
                "Stake (€)": f"€{stake:.2f}" if action != "PASS" else "-",
                "Prob Over": f"{prob_over * 100:.1f}%",
                "Prob Under": f"{prob_under * 100:.1f}%",
            }
        )

    # --- UI Rendering ---
    if results_data:
        df_results = pd.DataFrame(results_data)

        def highlight_bets(row):
            if "BET" in row["Action"]:
                return ["background-color: #004d00; color: white"] * len(row)
            return [""] * len(row)

        styled_df = df_results.style.apply(highlight_bets, axis=1)

        st.subheader("Actionable Intelligence")
        st.dataframe(styled_df)
    else:
        st.info("No matches analyzed.")


if __name__ == "__main__":
    main()
