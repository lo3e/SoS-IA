import os
import time
import json
import sqlite3
import requests
import argparse
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
SLEEP_TIME = 1.2

# === UTILITY ===
def call_api(endpoint, params=None):
    """Chiama endpoint API-Football e gestisce rate limit ed errori."""
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        if r.status_code == 429:
            print("🚫 Limite API raggiunto. Stop sicuro per non bruciare call.")
            return None
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as e:
        print(f"⚠️ Errore chiamando {endpoint}: {e}")
        return []

def connect_db():
    return sqlite3.connect(DB_PATH)

# === FUNZIONI DI CHECK ===
def exists(c, table, field, value):
    c.execute(f"SELECT 1 FROM {table} WHERE {field} = ? LIMIT 1", (value,))
    return c.fetchone() is not None

def exists_injuries_for_season(c, season):
    c.execute("SELECT 1 FROM injuries i JOIN matches m ON i.match_id = m.match_id WHERE m.season = ? LIMIT 1", (season,))
    return c.fetchone() is not None

# === INSERIMENTO DATI ===
def insert_team(conn, t):
    c = conn.cursor()
    team = t.get("team", {})
    c.execute("""
        INSERT OR REPLACE INTO teams (team_id, name, country, founded, last_update)
        VALUES (?, ?, ?, ?, ?)
    """, (
        team.get("id"),
        team.get("name"),
        team.get("country"),
        team.get("founded"),
        team.get("update")
    ))

def insert_match(conn, m):
    c = conn.cursor()
    fixture = m.get("fixture", {})
    league = m.get("league", {})
    teams = m.get("teams", {})
    goals = m.get("goals", {})

    c.execute("""
        INSERT OR REPLACE INTO matches (
            match_id, date, season, league_id, league_name,
            home_team_id, away_team_id, home_team_name, away_team_name,
            home_goals, away_goals, status, venue_id, last_update
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fixture.get("id"),
        fixture.get("date"),
        league.get("season"),
        league.get("id"),
        league.get("name"),
        teams.get("home", {}).get("id"),
        teams.get("away", {}).get("id"),
        teams.get("home", {}).get("name"),
        teams.get("away", {}).get("name"),
        goals.get("home"),
        goals.get("away"),
        fixture.get("status", {}).get("short"),
        fixture.get("venue", {}).get("id"),
        fixture.get("updated_at")
    ))

def insert_team_stats(conn, fixture_id):
    stats = call_api("/fixtures/statistics", {"fixture": fixture_id})
    if stats is None:
        return False
    if not stats:
        return True

    c = conn.cursor()
    for team_block in stats:
        team_id = team_block["team"]["id"]
        team_name = team_block["team"]["name"]
        for s in team_block["statistics"]:
            c.execute("""
                INSERT INTO team_stats (match_id, team_id, team_name, stat_type, stat_value)
                VALUES (?, ?, ?, ?, ?)
            """, (fixture_id, team_id, team_name, s.get("type"), str(s.get("value"))))
    conn.commit()
    return True

def insert_players(conn, fixture_id):
    """
    Scarica le statistiche giocatore-per-match da /fixtures/players
    e le inserisce nella tabella players con lo schema attuale.
    """
    data = call_api("/fixtures/players", {"fixture": fixture_id})
    if data is None:
        # limite API raggiunto -> fermiamoci senza crash
        return False
    if not data:
        # nessun dato per questo match -> non è un errore
        return True

    c = conn.cursor()

    for team_block in data:
        team_info = team_block.get("team", {})
        team_id = team_info.get("id")

        for player_block in team_block.get("players", []):
            player_info = player_block.get("player", {})
            stats_list = player_block.get("statistics", [])
            stats = stats_list[0] if stats_list else {}

            games_stats = stats.get("games", {}) or {}
            shots_stats = stats.get("shots", {}) or {}
            goals_stats = stats.get("goals", {}) or {}
            passes_stats = stats.get("passes", {}) or {}
            tackles_stats = stats.get("tackles", {}) or {}
            duels_stats = stats.get("duels", {}) or {}
            cards_stats = stats.get("cards", {}) or {}

            c.execute("""
                INSERT INTO players (
                    match_id,
                    team_id,
                    player_id,
                    player_name,
                    position,
                    minutes,
                    rating,
                    shots_total,
                    shots_on,
                    goals_total,
                    assists,
                    passes_total,
                    passes_key,
                    tackles,
                    interceptions,
                    duels_total,
                    duels_won,
                    yellow_cards,
                    red_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id,
                team_id,
                player_info.get("id"),
                player_info.get("name"),
                player_info.get("position"),
                games_stats.get("minutes"),
                games_stats.get("rating"),
                shots_stats.get("total"),
                shots_stats.get("on"),
                goals_stats.get("total"),
                goals_stats.get("assists"),
                passes_stats.get("total"),
                passes_stats.get("key"),
                tackles_stats.get("total"),
                tackles_stats.get("interceptions"),
                duels_stats.get("total"),
                duels_stats.get("won"),
                cards_stats.get("yellow"),
                cards_stats.get("red")
            ))

    conn.commit()
    return True

def insert_lineups(conn, fixture_id):
    data = call_api("/fixtures/lineups", {"fixture": fixture_id})
    if data is None:
        return False
    if not data:
        return True

    c = conn.cursor()

    for team_block in data:
        team = team_block.get("team", {})
        team_id = team.get("id")
        team_name = team.get("name")

        formation = team_block.get("formation")
        coach_name = team_block.get("coach", {}).get("name")

        # titolari
        for p in team_block.get("startXI", []):
            player_id = p.get("player", {}).get("id")
            player_name = p.get("player", {}).get("name")
            position = p.get("player", {}).get("pos")
            c.execute("""
                INSERT INTO lineups (
                    match_id, team_id, team_name,
                    formation, coach_name,
                    player_id, player_name, position, is_starter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                fixture_id,
                team_id,
                team_name,
                formation,
                coach_name,
                player_id,
                player_name,
                position
            ))

        # panchina
        for p in team_block.get("substitutes", []):
            player_id = p.get("player", {}).get("id")
            player_name = p.get("player", {}).get("name")
            position = p.get("player", {}).get("pos")
            c.execute("""
                INSERT INTO lineups (
                    match_id, team_id, team_name,
                    formation, coach_name,
                    player_id, player_name, position, is_starter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                fixture_id,
                team_id,
                team_name,
                formation,
                coach_name,
                player_id,
                player_name,
                position
            ))

    conn.commit()
    return True

def insert_events(conn, fixture_id):
    data = call_api("/fixtures/events", {"fixture": fixture_id})
    if data is None:
        return False
    if not data:
        return True

    c = conn.cursor()
    for e in data:
        c.execute("""
            INSERT INTO events (
                match_id,
                time_elapsed,
                team_id,
                team_name,
                player_id,
                player_name,
                assist_name,
                type,
                detail,
                comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id,
            e.get("time", {}).get("elapsed"),
            e.get("team", {}).get("id"),
            e.get("team", {}).get("name"),
            e.get("player", {}).get("id") if e.get("player") else None,
            e.get("player", {}).get("name") if e.get("player") else None,
            e.get("assist", {}).get("name") if e.get("assist") else None,
            e.get("type"),
            e.get("detail"),
            e.get("comments")
        ))

    conn.commit()
    return True

def insert_odds(conn, season):
    data = call_api("/odds", {"league": LEAGUE_ID, "season": season})
    if data is None or not data:
        print("⚠️ Nessuna quota disponibile per questa stagione.")
        return
    c = conn.cursor()
    for m in data:
        match_id = m["fixture"]["id"]
        for bookmaker in m.get("bookmakers", []):
            for bet in bookmaker.get("bets", []):
                market = bet.get("name")
                for val in bet.get("values", []):
                    c.execute("""
                        INSERT INTO odds (match_id, bookmaker_id, bookmaker_name, market, outcome, odd)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        match_id,
                        bookmaker.get("id"),
                        bookmaker.get("name"),
                        market,
                        val.get("value"),
                        val.get("odd")
                    ))
    conn.commit()

def insert_prediction(conn, match_id):
    data = call_api("/predictions", {"fixture": match_id})
    if data is None:
        return False
    if not data:
        return True

    p = data[0].get("predictions", {})
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (
            match_id,
            winner,
            win_or_draw,
            advice,
            prob_home,
            prob_draw,
            prob_away
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        match_id,
        p.get("winner", {}).get("name"),
        1 if p.get("win_or_draw") else 0,
        p.get("advice"),
        p.get("percent", {}).get("home"),
        p.get("percent", {}).get("draw"),
        p.get("percent", {}).get("away")
    ))
    conn.commit()
    return True

def insert_head2head(conn, home_id, away_id, season):
    data = call_api("/fixtures/headtohead", {
        "h2h": f"{home_id}-{away_id}",
        "league": LEAGUE_ID,
        "season": season
    })
    if data is None:
        return False
    if not data:
        return True

    c = conn.cursor()
    for match in data:
        fixture = match.get("fixture", {})
        goals = match.get("goals", {})
        c.execute("""
            INSERT INTO head2head (
                home_team_id,
                away_team_id,
                match_id,
                home_goals,
                away_goals,
                season
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            home_id,
            away_id,
            fixture.get("id"),
            goals.get("home"),
            goals.get("away"),
            season
        ))
    conn.commit()
    return True

def insert_injuries(conn, season):
    data = call_api("/injuries", {"league": LEAGUE_ID, "season": season})
    if data is None:
        return False
    if not data:
        print("   ⚠️ Nessun infortunio disponibile per questa stagione.")
        return True

    c = conn.cursor()
    for inj in data:
        c.execute("""
            INSERT INTO injuries (
                match_id,
                player_id,
                player_name,
                team_id,
                reason,
                since,
                expected_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
    return True

# === MAIN ===
def fetch_season(season):
    conn = connect_db()
    c = conn.cursor()
    print(f"\n📅 Scarico stagione {season} Serie A...")

    # 1️⃣ TEAMS
    teams = call_api("/teams", {"league": LEAGUE_ID, "season": season})
    for t in teams:
        insert_team(conn, t)
    print(f"✅ {len(teams)} squadre salvate")

    # 2️⃣ FIXTURES
    fixtures = call_api("/fixtures", {"league": LEAGUE_ID, "season": season})
    print(f"✅ {len(fixtures)} partite trovate")

    for idx, m in enumerate(fixtures, start=1):
        f_id = m["fixture"]["id"]
        print(f"\n({idx}/{len(fixtures)}) ▶️ Fixture {f_id}")

        if not exists(c, "matches", "match_id", f_id):
            insert_match(conn, m)

        insert_team_stats(conn, f_id)
        insert_players(conn, f_id)
        insert_lineups(conn, f_id)
        insert_events(conn, f_id)
        insert_prediction(conn, f_id)
        insert_head2head(conn, m["teams"]["home"]["id"], m["teams"]["away"]["id"], season)

        time.sleep(SLEEP_TIME)

    insert_odds(conn, season)

    if not exists_injuries_for_season(c, season):
        insert_injuries(conn, season)

    print(f"\n✅ Stagione {season} completata e salvata nel DB.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scarica tutti i dati per una stagione Serie A")
    parser.add_argument("--season", type=int, required=True, help="Anno della stagione (es. 2021)")
    args = parser.parse_args()

    fetch_season(args.season)
