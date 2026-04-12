"""
explore_data.py
Initial data loading and basic exploratory data analysis (EDA).
"""

import pandas as pd


def load_and_inspect_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame and prints basic structural info.
    """
    print(f"--- Loading data from: {file_path} ---")

    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Display hardware-relevant metrics (Memory usage)
        print("\n[INFO] Dataframe Info:")
        df.info(memory_usage="deep")

        # Display statistical summary
        print("\n[INFO] Statistical Summary:")
        print(df.describe())

        return df

    except FileNotFoundError:
        print(f"[ERROR] File {file_path} not found. Check your directory structure.")
        return None


if __name__ == "__main__":
    # Define the relative path to the dataset
    DATA_PATH = "data/housing.csv"

    # Execute the loading function
    housing_dataframe = load_and_inspect_data(DATA_PATH)
