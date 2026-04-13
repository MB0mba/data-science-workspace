"""
fetch_live_odds.py
Fetches live Serie A U/O 2.5 odds from Pinnacle via The Odds API.
Filters out matches occurring more than 7 days in the future to capture the full weekend.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


def fetch_and_cache_live_odds():
    load_dotenv()
    api_key = os.getenv("THE_ODDS_API_KEY")

    if not api_key or api_key == "inserisci_qui_la_tua_chiave":
        print("[ERROR] API Key is missing. Check your .env file.")
        return

    CACHE_DIR = Path("data/interim")
    CACHE_FILE = CACHE_DIR / "live_odds_cache.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Authenticating with The Odds API (Targeting Pinnacle)...")

    url = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "totals",
        "bookmakers": "pinnacle",
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            error_msg = response.json().get("message", "Unknown Error")
            print(f"[ERROR] API Error: {response.status_code} - {error_msg}")
            return

        calls_used = response.headers.get("x-requests-used", "Unknown")
        calls_left = response.headers.get("x-requests-remaining", "Unknown")
        print(f"[API QUOTA] Used: {calls_used} | Remaining: {calls_left}")

        data = response.json()
        print(f"[SUCCESS] Matches retrieved from API: {len(data)}")

        clean_matches = []
        now = datetime.now(timezone.utc)
        max_date = now + timedelta(days=7)  # Updated to 7-Day Horizon

        for match in data:
            match_time = datetime.fromisoformat(
                match["commence_time"].replace("Z", "+00:00")
            )
            if match_time > max_date:
                continue

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

            if match_obj["Odds_Over_25"] and match_obj["Odds_Under_25"]:
                clean_matches.append(match_obj)

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_matches, f, indent=4, ensure_ascii=False)

        print(
            f"[SUCCESS] Filtered {len(clean_matches)} matches within 7 days with Pinnacle odds."
        )
        print(f"[SUCCESS] Data safely cached to {CACHE_FILE}.")

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")


if __name__ == "__main__":
    fetch_and_cache_live_odds()
