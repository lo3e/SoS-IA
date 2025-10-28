# src/fetch_full_data.py
import os
import time
import argparse
import requests
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
DB_PATH = os.getenv("DB_PATH")

if not API_KEY:
    raise ValueError("❌ API_FOOTBALL_KEY mancante nel file .env")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 135  # Serie A
SLEEP = 1.2      # rate limit safe delay

# === UTILS ===
def call_api(endpoint: str, params: dict = None):
    """Chiama un endpoint API-Football e restituisce la lista 'response'."""
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"⚠️ Errore chiamando {endpoint}: {e}")
        return []

def connect_db():
    return sqlite3.connect(DB_PATH)

# === INSERT FUNCS ===
def insert_match(conn, m):
    c = conn.cursor()
    f, l, t, g = m["fixture"], m["league"], m["teams"], m["goals"]
    c.execute("""
        INSERT OR REPLACE INTO matches
        (match_id, date, season, league_id, league_name, home_team_id, away_team_id,
         home_team_name, away_team_name, home_goals, away_goals, status, venue_id, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        f.get("id"), f.get("date"), l.get("season"), l.get("id"), l.get("name"),
        t.get("home", {}).get("id"), t.get("away", {}).get("id"),
        t.get("home", {}).get("name"), t.get("away", {}).get("name"),
        g.get("home"), g.get("away"), f.get("status", {}).get("short"),
        f.get("venue", {}).get("id"), f.get("updated_at")
    ))

def insert_team(conn, t):
    c = conn.cursor()
    team, venue = t["team"], t.get("venue", {})
    c.execute("""
        INSERT OR REPLACE INTO teams (team_id, name, country, founded, last_update)
        VALUES (?, ?, ?, ?, ?)
    """, (team.get("id"), team.get("name"), team.get("country"),
          team.get("founded"), team.get("updated_at")))

def insert_team_stats(conn, fixture_id):
    data = call_api("/fixtures/statistics", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    for t in data:
        team = t["team"]
        for s in t["statistics"]:
            c.execute("""
                INSERT INTO team_stats (match_id, team_id, team_name, stat_type, stat_value)
                VALUES (?, ?, ?, ?, ?)
            """, (fixture_id, team["id"], team["name"], s["type"], str(s["value"]) if s["value"] else None))
    conn.commit()

def insert_players(conn, fixture_id):
    data = call_api("/fixtures/players", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    for team_data in data:
        team_id = team_data["team"]["id"]
        for p in team_data["players"]:
            player = p["player"]
            stats = p["statistics"][0] if p["statistics"] else {}
            c.execute("""
                INSERT INTO players (match_id, team_id, player_id, player_name, position, minutes,
                    rating, shots_total, shots_on, goals_total, assists, passes_total, passes_key,
                    tackles, interceptions, duels_total, duels_won, yellow_cards, red_cards)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id, team_id, player["id"], player["name"], player.get("position"),
                stats.get("games", {}).get("minutes"),
                stats.get("games", {}).get("rating"),
                stats.get("shots", {}).get("total"),
                stats.get("shots", {}).get("on"),
                stats.get("goals", {}).get("total"),
                stats.get("goals", {}).get("assists"),
                stats.get("passes", {}).get("total"),
                stats.get("passes", {}).get("key"),
                stats.get("tackles", {}).get("total"),
                stats.get("tackles", {}).get("interceptions"),
                stats.get("duels", {}).get("total"),
                stats.get("duels", {}).get("won"),
                stats.get("cards", {}).get("yellow"),
                stats.get("cards", {}).get("red")
            ))
    conn.commit()

def insert_lineups(conn, fixture_id):
    data = call_api("/fixtures/lineups", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    for team_data in data:
        team = team_data["team"]
        coach = team_data.get("coach", {}).get("name")
        formation = team_data.get("formation")
        # Start XI
        for p in team_data.get("startXI", []):
            c.execute("""
                INSERT INTO lineups (match_id, team_id, team_name, formation, coach_name, player_id, player_name, position, is_starter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (fixture_id, team["id"], team["name"], formation, coach,
                  p["player"]["id"], p["player"]["name"], p["player"]["pos"]))
        # Substitutes
        for p in team_data.get("substitutes", []):
            c.execute("""
                INSERT INTO lineups (match_id, team_id, team_name, formation, coach_name, player_id, player_name, position, is_starter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (fixture_id, team["id"], team["name"], formation, coach,
                  p["player"]["id"], p["player"]["name"], p["player"]["pos"]))
    conn.commit()

def insert_events(conn, fixture_id):
    data = call_api("/fixtures/events", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    for e in data:
        c.execute("""
            INSERT INTO events (match_id, time_elapsed, team_id, team_name, player_id, player_name, assist_name, type, detail, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id,
            e["time"].get("elapsed"),
            e["team"].get("id"),
            e["team"].get("name"),
            e["player"].get("id"),
            e["player"].get("name"),
            e.get("assist", {}).get("name"),
            e.get("type"), e.get("detail"), e.get("comments")
        ))
    conn.commit()

def insert_predictions(conn, fixture_id):
    data = call_api("/predictions", {"fixture": fixture_id})
    if not data:
        return
    p = data[0]["predictions"]
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (match_id, winner, win_or_draw, advice, prob_home, prob_draw, prob_away)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        fixture_id,
        p.get("winner", {}).get("name"),
        1 if p.get("win_or_draw") else 0,
        p.get("advice"),
        p.get("percent", {}).get("home"),
        p.get("percent", {}).get("draw"),
        p.get("percent", {}).get("away")
    ))
    conn.commit()

def insert_head2head(conn, home_id, away_id, season):
    data = call_api("/fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "league": LEAGUE_ID, "season": season})
    if not data:
        return
    c = conn.cursor()
    for match in data:
        f, g = match["fixture"], match["goals"]
        c.execute("""
            INSERT INTO head2head (home_team_id, away_team_id, match_id, home_goals, away_goals, season)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (home_id, away_id, f["id"], g["home"], g["away"], season))
    conn.commit()

def insert_injuries(conn, season):
    data = call_api("/injuries", {"league": LEAGUE_ID, "season": season})
    if not data:
        return
    c = conn.cursor()
    for inj in data:
        c.execute("""
            INSERT INTO injuries (match_id, player_id, player_name, team_id, reason, since, expected_return)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            inj.get("fixture", {}).get("id"),
            inj.get("player", {}).get("id"),
            inj.get("player", {}).get("name"),
            inj.get("team", {}).get("id"),
            inj.get("player", {}).get("reason"),
            inj.get("player", {}).get("since"),
            inj.get("player", {}).get("expected_return")
        ))
    conn.commit()

def insert_odds(conn, season):
    data = call_api("/odds", {"league": LEAGUE_ID, "season": season})
    if not data:
        return
    c = conn.cursor()
    for o in data:
        match_id = o["fixture"]["id"]
        for b in o["bookmakers"]:
            for bet in b.get("bets", []):
                for val in bet.get("values", []):
                    c.execute("""
                        INSERT INTO odds (match_id, bookmaker_id, bookmaker_name, market, outcome, odd, timestamp, source)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'api')
                    """, (
                        match_id, b["id"], b["name"], bet["name"],
                        val["value"], val["odd"]
                    ))
    conn.commit()

# === MAIN ===
def fetch_season(season):
    conn = connect_db()
    c = conn.cursor()
    print(f"\n📅 Scarico stagione {season} per Serie A...")

    teams = call_api("/teams", {"league": LEAGUE_ID, "season": season})
    for t in teams: insert_team(conn, t)
    print(f"✅ {len(teams)} squadre salvate")

    fixtures = call_api("/fixtures", {"league": LEAGUE_ID, "season": season})
    print(f"✅ {len(fixtures)} partite trovate\n")

    new_count = 0
    skipped_count = 0

    for i, m in enumerate(fixtures):
        fixture_id = m["fixture"]["id"]
        print(f"({i+1}/{len(fixtures)}) Controllo fixture {fixture_id}...")

        # --- 🔍 verifica se abbiamo già tutto ---
        c.execute("""
            SELECT 
                (SELECT COUNT(*) FROM matches WHERE match_id = ?) AS match_ok,
                (SELECT COUNT(*) FROM team_stats WHERE match_id = ?) AS stats_ok,
                (SELECT COUNT(*) FROM lineups WHERE match_id = ?) AS lineups_ok,
                (SELECT COUNT(*) FROM players WHERE match_id = ?) AS players_ok,
                (SELECT COUNT(*) FROM events WHERE match_id = ?) AS events_ok
        """, (fixture_id, fixture_id, fixture_id, fixture_id, fixture_id))
        
        match_ok, stats_ok, lineups_ok, players_ok, events_ok = c.fetchone()

        # ✅ se tutti i dati sono già presenti → skip
        if all(x > 0 for x in [match_ok, stats_ok, lineups_ok, players_ok, events_ok]):
            print(f"   ✅ Fixture {fixture_id} già completa, skip.")
            skipped_count += 1
            continue

        # 💾 altrimenti scarica e salva tutto
        print(f"   📥 Scarico dati completi per fixture {fixture_id}...")
        try:
            insert_match(conn, m)
            insert_team_stats(conn, fixture_id)
            insert_lineups(conn, fixture_id)
            insert_players(conn, fixture_id)
            insert_events(conn, fixture_id)
            conn.commit()
            time.sleep(SLEEP)
            new_count += 1
        except Exception as e:
            print(f"⚠️ Errore su fixture {fixture_id}: {e}")
            conn.commit()

    # --- 📊 Riepilogo finale ---
    print("\n📊 Riepilogo stagione:")
    print(f"   🟢 Nuove fixture scaricate: {new_count}")
    print(f"   🟡 Fixture già complete: {skipped_count}")
    print(f"   🔢 Totale fixture processate: {len(fixtures)}\n")

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scarica una stagione completa da API-Football")
    parser.add_argument("--season", type=int, required=True, help="Anno stagione es. 2015")
    args = parser.parse_args()
    fetch_season(args.season)
