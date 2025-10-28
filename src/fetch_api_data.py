# src/fetch_api_data.py
import os
import sqlite3
import requests
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from db import DB_PATH, init_db

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 135  # Serie A
HEADERS = {"x-apisports-key": API_KEY}

# --- DB Helpers ---
def insert_match(conn, m):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO matches (match_id, date, season, league, home, away, home_goals, away_goals, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        m["match_id"], m["date"], m["season"], m["league"], m["home"], m["away"],
        m["home_goals"], m["away_goals"], m["status"]
    ))
    conn.commit()

def insert_odds(conn, odds_rows):
    c = conn.cursor()
    c.executemany("""
        INSERT INTO odds (match_id, source, timestamp, odds_1, odds_x, odds_2)
        VALUES (?, ?, ?, ?, ?, ?)
    """, odds_rows)
    conn.commit()

def insert_team_stats(conn, stats_rows):
    c = conn.cursor()
    c.executemany("""
        INSERT INTO team_stats (match_id, team, xG, xGA, shots, possession)
        VALUES (?, ?, ?, ?, ?, ?)
    """, stats_rows)
    conn.commit()

# --- API Helpers ---
def api_get(endpoint, params):
    r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params)
    if r.status_code != 200:
        print(f"❌ Errore API {r.status_code}: {r.text}")
        return None
    return r.json().get("response", [])

def fetch_fixtures(season):
    data = api_get("fixtures", {"league": LEAGUE_ID, "season": season})
    return data or []

def fetch_odds(match_id):
    data = api_get("odds", {"fixture": match_id})
    rows = []
    for b in data:
        book = b.get("bookmaker", {}).get("name")
        timestamp = b.get("update")
        bets = b.get("bets", [])
        for bet in bets:
            if bet.get("name") == "Match Winner":
                values = {v["value"]: v["odd"] for v in bet["values"]}
                odds_1 = values.get("Home")
                odds_x = values.get("Draw")
                odds_2 = values.get("Away")
                rows.append((match_id, book, timestamp, odds_1, odds_x, odds_2))
    return rows

def fetch_stats(match_id):
    data = api_get("fixtures/statistics", {"fixture": match_id})
    rows = []
    for t in data:
        team = t["team"]["name"]
        stats = {s["type"]: s.get("value") for s in t["statistics"]}
        xG = stats.get("Expected Goals", None)
        xGA = stats.get("Expected Goals Against", None)
        shots = stats.get("Total Shots", None)
        possession = stats.get("Ball Possession", None)
        rows.append((match_id, team, xG, xGA, shots, possession))
    return rows

# --- Main ---
def fetch_season(season):
    print(f"📅 Scarico stagione {season}...")
    conn = sqlite3.connect(DB_PATH)
    fixtures = fetch_fixtures(season)
    print(f"✅ {len(fixtures)} partite trovate per {season}")

    for i, f in enumerate(fixtures, 1):
        fixture = f["fixture"]
        teams = f["teams"]
        goals = f["goals"]

        m = {
            "match_id": fixture["id"],
            "date": fixture["date"],
            "season": season,
            "league": f["league"]["name"],
            "home": teams["home"]["name"],
            "away": teams["away"]["name"],
            "home_goals": goals["home"],
            "away_goals": goals["away"],
            "status": fixture["status"]["short"]
        }

        insert_match(conn, m)

        # Fetch odds
        odds_rows = fetch_odds(fixture["id"])
        if odds_rows:
            insert_odds(conn, odds_rows)

        # Fetch stats
        stats_rows = fetch_stats(fixture["id"])
        if stats_rows:
            insert_team_stats(conn, stats_rows)

        if i % 10 == 0:
            print(f"⏳ Elaborate {i}/{len(fixtures)} partite...")
        time.sleep(1)  # throttle per evitare rate limit

    conn.close()
    print(f"🎯 Stagione {season} completata e salvata nel DB!")

def main():
    parser = argparse.ArgumentParser(description="Scarica dati API-Football per una stagione specifica")
    parser.add_argument("--season", type=int, required=True, help="Anno di inizio stagione (es. 2023)")
    args = parser.parse_args()

    init_db()
    fetch_season(args.season)

if __name__ == "__main__":
    main()
