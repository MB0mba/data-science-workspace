"""
process_data.py
Transform phase of the ETL pipeline (Over/Under 2.5 Goals Market).
Loads raw CSVs, enforces a strict schema, and concatenates them into a clean interim dataset.
"""

import glob
import os

import pandas as pd


def enforce_schema_and_merge(
    raw_data_dir: str, interim_data_dir: str, target_league: str
) -> None:
    """
    Reads raw CSV files, extracts core columns for the U/O 2.5 market,
    and saves a consolidated interim dataset.
    """
    # Define the strict data contract for Under/Over
    CORE_COLUMNS = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "B365>2.5",
        "B365<2.5",
    ]

    os.makedirs(interim_data_dir, exist_ok=True)
    file_pattern = os.path.join(raw_data_dir, f"{target_league}_*.csv")
    raw_files = glob.glob(file_pattern)

    if not raw_files:
        print(f"[ERROR] No raw files found for {target_league} in {raw_data_dir}.")
        return

    print(
        f"[INFO] Found {len(raw_files)} raw files for {target_league}. Starting extraction (U/O Market)..."
    )

    processed_dataframes = []

    for file_path in sorted(raw_files):
        try:
            df_raw = pd.read_csv(file_path)
            missing_cols = [col for col in CORE_COLUMNS if col not in df_raw.columns]

            if missing_cols:
                print(
                    f"[WARNING] Skipping {file_path}. Missing core columns: {missing_cols}"
                )
                continue

            df_filtered = df_raw[CORE_COLUMNS].copy()
            processed_dataframes.append(df_filtered)
            print(f"[SUCCESS] Processed {os.path.basename(file_path)}.")

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}. Reason: {e}")

    if processed_dataframes:
        master_df = pd.concat(processed_dataframes, ignore_index=True)
        master_df["Date"] = pd.to_datetime(
            master_df["Date"], format="mixed", dayfirst=True
        )
        master_df.sort_values("Date", inplace=True)

        output_path = os.path.join(interim_data_dir, f"{target_league}_master.csv")
        master_df.to_csv(output_path, index=False)

        print("\n[INFO] Interim dataset creation complete.")
        print(f"[INFO] Saved master dataset to: {output_path}")
        print(f"[INFO] Total matches recorded: {master_df.shape[0]}")
    else:
        print(f"[ERROR] No valid dataframes to merge for {target_league}.")


if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw")
    INTERIM_PATH = os.path.join("data", "interim")
    LEAGUE_TO_PROCESS = "Serie_A"
    enforce_schema_and_merge(RAW_PATH, INTERIM_PATH, LEAGUE_TO_PROCESS)
