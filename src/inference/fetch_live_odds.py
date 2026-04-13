"""
fetch_live_odds.py
Fetches live Serie A U/O 2.5 odds from Pinnacle via The Odds API.
Implements HTTP Header inspection for quota tracking and local JSON caching.
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def fetch_and_cache_live_odds():
    # 1. Secure Environment Loading
    load_dotenv()
    api_key = os.getenv("THE_ODDS_API_KEY")

    if not api_key or api_key == "inserisci_qui_la_tua_chiave":
        print("[ERROR] API Key is missing. Check your .env file.")
        return

    # Define paths
    CACHE_DIR = Path("data/interim")
    CACHE_FILE = CACHE_DIR / "live_odds_cache.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Authenticating with The Odds API (Targeting Pinnacle)...")

    url = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "totals",
        "bookmakers": "pinnacle",  # We enforce Pinnacle as requested
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        # Check if the request was successful before parsing JSON
        if response.status_code != 200:
            error_msg = response.json().get("message", "Unknown Error")
            print(f"[ERROR] API Error: {response.status_code} - {error_msg}")
            return

        # 2. Extract Quota from Headers (Crucial for production monitoring)
        calls_used = response.headers.get("x-requests-used", "Unknown")
        calls_left = response.headers.get("x-requests-remaining", "Unknown")
        print(f"[API QUOTA] Used: {calls_used} | Remaining: {calls_left}")

        data = response.json()
        print(f"[SUCCESS] Matches retrieved from API: {len(data)}")

        # 3. Parse and Clean Data
        clean_matches = []

        for match in data:
            match_obj = {
                "HomeTeam_API": match["home_team"],
                "AwayTeam_API": match["away_team"],
                "CommenceTime": match["commence_time"],
                "Odds_Over_25": None,
                "Odds_Under_25": None,
            }

            bookmakers = match.get("bookmakers", [])
            for bookie in bookmakers:
                if bookie["key"] == "pinnacle":
                    for market in bookie.get("markets", []):
                        if market["key"] == "totals":
                            for outcome in market.get("outcomes", []):
                                if outcome.get("point") == 2.5:
                                    if outcome["name"] == "Over":
                                        match_obj["Odds_Over_25"] = outcome["price"]
                                    elif outcome["name"] == "Under":
                                        match_obj["Odds_Under_25"] = outcome["price"]

            # Keep match only if odds were successfully mapped
            if match_obj["Odds_Over_25"] and match_obj["Odds_Under_25"]:
                clean_matches.append(match_obj)

        # 4. Save to Cache
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_matches, f, indent=4, ensure_ascii=False)

        print(
            f"[SUCCESS] Filtered {len(clean_matches)} matches with valid Pinnacle odds."
        )
        print(
            f"[SUCCESS] Data safely cached to {CACHE_FILE}. Use this file for testing."
        )

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network/Request Exception: {e}")
    except Exception as e:
        print(f"[ERROR] General Exception: {e}")


if __name__ == "__main__":
    fetch_and_cache_live_odds()
