"""
download_data.py
Extract phase of the ETL pipeline.
Downloads historical football data directly from football-data.co.uk into the raw data layer.
"""

import os

import pandas as pd


def fetch_historical_data(leagues: dict, seasons: list, raw_data_dir: str) -> None:
    """
    Downloads CSV datasets from football-data.co.uk and saves them locally.

    Args:
        leagues (dict): Mapping of league names to their URL codes (e.g., {'Serie_A': 'I1'}).
        seasons (list): List of season strings in 'YY_YY' format (e.g., ['2425', '2526']).
        raw_data_dir (str): Relative path to the raw data directory.
    """
    base_url = "https://www.football-data.co.uk/mmz4281/"

    # Ensure the target directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    for league_name, league_code in leagues.items():
        for season in seasons:
            # Construct the target URL (e.g., https://www.football-data.co.uk/mmz4281/2526/I1.csv)
            url = f"{base_url}{season}/{league_code}.csv"
            file_name = f"{league_name}_{season}.csv"
            file_path = os.path.join(raw_data_dir, file_name)

            print(f"[INFO] Attempting to download {league_name} (Season {season})...")

            try:
                # pandas read_csv can natively handle HTTP GET requests
                df = pd.read_csv(url)

                # Save the raw dataframe to our local vault without index
                df.to_csv(file_path, index=False)
                print(
                    f"[SUCCESS] Saved {file_name} with {df.shape[0]} matches and {df.shape[1]} features."
                )

            except Exception as e:
                print(f"[ERROR] Failed to retrieve data from {url}. Reason: {e}")


if __name__ == "__main__":
    # Define the infrastructure paths
    RAW_DATA_PATH = os.path.join("data", "raw")

    # Define our targets: Serie A (I1) and Serie B (I2) for the last two seasons
    TARGET_LEAGUES = {"Serie_A": "I1", "Serie_B": "I2"}
    TARGET_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]

    # Execute extraction
    fetch_historical_data(TARGET_LEAGUES, TARGET_SEASONS, RAW_DATA_PATH)
