"""
build_features.py
Calculates rolling features for goals and creates the binary target variable (Over 2.5).
"""

from pathlib import Path

import pandas as pd


def calculate_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculates rolling sums for goals scored and conceded.
    Creates the binary target 'Over2.5'.
    """
    print(f"\n[INFO] Calculating rolling goal features (window={window})...")

    # 1. Create Target Variable (Binary: 1 if > 2.5, else 0)
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["Over2.5"] = (df["TotalGoals"] > 2.5).astype(int)

    # 2. Create unified team timeline
    home_df = df[["Date", "HomeTeam", "FTHG", "FTAG"]].copy()
    home_df.columns = ["Date", "Team", "GoalsScored", "GoalsConceded"]

    away_df = df[["Date", "AwayTeam", "FTAG", "FTHG"]].copy()
    away_df.columns = ["Date", "Team", "GoalsScored", "GoalsConceded"]

    team_stats = pd.concat([home_df, away_df], axis=0)
    team_stats["Date"] = pd.to_datetime(team_stats["Date"])
    team_stats = team_stats.sort_values(by=["Team", "Date"])

    # 3. Calculate rolling metrics with shift(1) to prevent Data Leakage
    grouped = team_stats.groupby("Team")

    features = ["GoalsScored", "GoalsConceded"]
    for feature in features:
        team_stats[f"rolling_{feature}_sum_{window}"] = (
            grouped[feature].shift(1).rolling(window=window, min_periods=window).sum()
        )

    computed_features = team_stats[
        [
            "Date",
            "Team",
            f"rolling_GoalsScored_sum_{window}",
            f"rolling_GoalsConceded_sum_{window}",
        ]
    ].copy()

    # 4. Merge back to the match-centric dataframe
    print(
        "[INFO] Merging features back to preserve bookmaker odds and match structure..."
    )
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.merge(
        computed_features.add_prefix("Home_"),
        left_on=["Date", "HomeTeam"],
        right_on=["Home_Date", "Home_Team"],
        how="left",
    ).drop(columns=["Home_Date", "Home_Team"])

    df = df.merge(
        computed_features.add_prefix("Away_"),
        left_on=["Date", "AwayTeam"],
        right_on=["Away_Date", "Away_Team"],
        how="left",
    ).drop(columns=["Away_Date", "Away_Team"])

    # 5. Handle Cold Start
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
        print(f"[INFO] Target Variable preserved: {'Over2.5' in features_df.columns}")
        print(f"[INFO] Target Odds preserved: {'B365>2.5' in features_df.columns}")
        print(f"[INFO] Saving processed features to {output_path}...")

        features_df.to_csv(output_path, index=False)
        print(
            "[SUCCESS] Engineering complete. Matrix is ready for Binary Classification."
        )

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
