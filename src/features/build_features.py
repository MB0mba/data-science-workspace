"""
build_features.py
Calculates rolling features and merges them back into a match-centric dataframe.
"""

from pathlib import Path

import pandas as pd


def calculate_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculates rolling sums for points, goals scored, and goals conceded per team.
    Merges these features back into the original match-centric dataframe.
    """
    print(f"\n[INFO] Calculating rolling features (window={window})...")

    # 1. Create a unified team timeline to calculate accurate rolling stats
    home_df = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
    home_df.columns = ["Date", "Team", "GoalsScored", "GoalsConceded", "Result"]
    home_df["Points"] = home_df["Result"].map({"H": 3, "D": 1, "A": 0})

    away_df = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].copy()
    away_df.columns = ["Date", "Team", "GoalsScored", "GoalsConceded", "Result"]
    away_df["Points"] = away_df["Result"].map({"A": 3, "D": 1, "H": 0})

    # Combine into a single timeline per team
    team_stats = pd.concat([home_df, away_df], axis=0)
    team_stats["Date"] = pd.to_datetime(team_stats["Date"])
    team_stats = team_stats.sort_values(by=["Team", "Date"])

    # 2. Calculate rolling metrics with shift(1) to prevent Data Leakage
    grouped = team_stats.groupby("Team")

    features = ["Points", "GoalsScored", "GoalsConceded"]
    for feature in features:
        team_stats[f"rolling_{feature}_sum_{window}"] = (
            grouped[feature].shift(1).rolling(window=window, min_periods=window).sum()
        )

    # Extract just the computed features to merge back
    computed_features = team_stats[
        [
            "Date",
            "Team",
            f"rolling_Points_sum_{window}",
            f"rolling_GoalsScored_sum_{window}",
            f"rolling_GoalsConceded_sum_{window}",
        ]
    ].copy()

    # 3. Merge back to the original Match-Centric dataframe
    print(
        "[INFO] Merging features back to preserve bookmaker odds and match structure..."
    )
    df["Date"] = pd.to_datetime(df["Date"])

    # Merge for Home Team
    df = df.merge(
        computed_features.add_prefix("Home_"),
        left_on=["Date", "HomeTeam"],
        right_on=["Home_Date", "Home_Team"],
        how="left",
    ).drop(columns=["Home_Date", "Home_Team"])

    # Merge for Away Team
    df = df.merge(
        computed_features.add_prefix("Away_"),
        left_on=["Date", "AwayTeam"],
        right_on=["Away_Date", "Away_Team"],
        how="left",
    ).drop(columns=["Away_Date", "Away_Team"])

    # 4. Handle Cold Start Problem
    initial_shape = df.shape[0]
    df = df.dropna()
    final_shape = df.shape[0]
    print(
        f"[INFO] Cold start handling: dropped {initial_shape - final_shape} early-season matches."
    )

    return df


def main():
    input_path = Path("data/interim/Serie_A_master.csv")
    output_path = Path("data/processed/Serie_A_features.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Interim dataset not found at {input_path}")
        return

    print(f"[INFO] Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    try:
        features_df = calculate_rolling_features(df, window=5)

        print(f"\n[INFO] Final dataset shape: {features_df.shape}")
        print(
            f"[INFO] Target Variables (Odds) preserved: {'B365H' in features_df.columns}"
        )
        print(f"[INFO] Saving processed features to {output_path}...")
        features_df.to_csv(output_path, index=False)
        print("[SUCCESS] Engineering complete. Matrix is ready for Machine Learning.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
