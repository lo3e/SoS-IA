# src/daily_update.py
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import requests
from pathlib import Path
from dotenv import load_dotenv
from classify_injuries import classify_injuries

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

if not API_KEY:
    raise ValueError("❌ API_FOOTBALL_KEY mancante nel .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 135  # Serie A
SLEEP_TIME = 1.0
DAYS_AHEAD = 14  # orizzonte partite future


# ---------------------------------------------------------
# utility
# ---------------------------------------------------------
def call_api(endpoint: str, params: dict = None):
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=25)
        if r.status_code == 429:
            print("🚫 Rate limit raggiunto.")
            return None
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as e:
        print(f"⚠️ Errore API {endpoint}: {e}")
        return None


def connect_db():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------
# 1) aggiorna fixtures future (NS) entro N giorni
# ---------------------------------------------------------
def update_fixtures(conn):
    c = conn.cursor()

    # prendo le partite ancora da giocare
    c.execute("""
        SELECT match_id, date
        FROM matches
        WHERE status != 'FT'
    """)
    rows = c.fetchall()
    if not rows:
        print("📭 Nessuna fixture futura da aggiornare.")
        return

    now = datetime.now(timezone.utc)
    to_update = []
    for match_id, date_str in rows:
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "")).replace(tzinfo=timezone.utc)
            if 0 <= (dt - now).days <= DAYS_AHEAD:
                to_update.append(match_id)
        except Exception:
            continue

    print(f"📅 Fixtures da aggiornare entro {DAYS_AHEAD} giorni: {len(to_update)}")

    for idx, fixture_id in enumerate(to_update, start=1):
        print(f"  ({idx}/{len(to_update)}) 🔁 fixture {fixture_id}")
        data = call_api("/fixtures", {"id": fixture_id})
        if not data:
            continue
        f = data[0]
        # aggiorniamo solo status + goals se disponibili
        fixture = f.get("fixture", {})
        league = f.get("league", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})

        c.execute("""
            UPDATE matches
            SET status = ?,
                home_goals = COALESCE(?, home_goals),
                away_goals = COALESCE(?, away_goals)
            WHERE match_id = ?
        """, (
            fixture.get("status", {}).get("short"),
            goals.get("home"),
            goals.get("away"),
            fixture_id
        ))

        # se la partita è finita ora → scarichiamo stats, events, players, lineups
        if fixture.get("status", {}).get("short") == "FT":
            insert_team_stats(conn, fixture_id)
            insert_events(conn, fixture_id)
            insert_lineups(conn, fixture_id)
            insert_players(conn, fixture_id)

        time.sleep(SLEEP_TIME)

    conn.commit()
    print("✅ Fixtures aggiornate.")


# ---------------------------------------------------------
# 2) standings aggiornati
# ---------------------------------------------------------
def update_standings(conn):
    data = call_api("/standings", {"league": LEAGUE_ID, "season": current_season()})
    if not data:
        print("⚠️ Nessuna standings aggiornata.")
        return
    c = conn.cursor()
    # cancello standings per la stagione corrente e riscrivo
    c.execute("DELETE FROM standings WHERE season = ?", (current_season(),))

    standings_list = data[0].get("league", {}).get("standings", [[]])[0]
    for row in standings_list:
        team = row.get("team", {})
        stats = row.get("all", {})
        c.execute("""
            INSERT INTO standings (
                season, league_id, team_id, team_name,
                rank, points, goals_diff, form,
                played, win, draw, lose, goals_for, goals_against, last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            current_season(),
            LEAGUE_ID,
            team.get("id"),
            team.get("name"),
            row.get("rank"),
            row.get("points"),
            row.get("goalsDiff"),
            row.get("form"),
            stats.get("played"),
            stats.get("win"),
            stats.get("draw"),
            stats.get("lose"),
            stats.get("goals", {}).get("for"),
            stats.get("goals", {}).get("against")
        ))

    conn.commit()
    print("✅ Standings aggiornate.")


# ---------------------------------------------------------
# 3) injuries del giorno (solo se nuovi)
# ---------------------------------------------------------
def update_injuries(conn):
    # prendiamo solo quelli della stagione corrente
    data = call_api("/injuries", {"league": LEAGUE_ID, "season": current_season()})
    if not data:
        print("⚠️ Nessun infortunio nuovo.")
        return

    c = conn.cursor()
    inserted = 0
    for inj in data:
        fixture_id = inj.get("fixture", {}).get("id")
        player_id = inj.get("player", {}).get("id")
        team_id = inj.get("team", {}).get("id")

        # check: se esiste già, skip
        c.execute("""
            SELECT 1 FROM injuries
            WHERE match_id = ? AND player_id = ? AND team_id = ?
        """, (fixture_id, player_id, team_id))
        if c.fetchone():
            continue

        c.execute("""
            INSERT INTO injuries (
                match_id, player_id, player_name, team_id, reason
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            fixture_id,
            player_id,
            inj.get("player", {}).get("name"),
            team_id,
            inj.get("player", {}).get("reason")
        ))
        inserted += 1

    conn.commit()
    classify_injuries(conn)
    print(f"✅ Infortuni aggiornati (+{inserted} nuovi)")


# ---------------------------------------------------------
# 4) odds pre-match per partite future
# ---------------------------------------------------------
def update_prematch_odds(conn, days_ahead=DAYS_AHEAD):
    c = conn.cursor()
    c.execute("""
        SELECT match_id, date
        FROM matches
        WHERE status = 'NS'
    """)
    rows = c.fetchall()
    if not rows:
        print("📭 Nessuna partita NS trovata.")
        return

    now = datetime.now(timezone.utc)
    upcoming = []
    for match_id, date_str in rows:
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "")).replace(tzinfo=timezone.utc)
            if 0 <= (dt - now).days <= days_ahead:
                upcoming.append((match_id, dt))
        except Exception:
            continue

    print(f"🎯 Partite NS entro {days_ahead} giorni: {len(upcoming)}")

    for idx, (match_id, match_date) in enumerate(upcoming, start=1):
        # se abbiamo già odds per questo match, skip
        c.execute("SELECT COUNT(*) FROM odds WHERE match_id = ?", (match_id,))
        if c.fetchone()[0] > 0:
            print(f"  ({idx}/{len(upcoming)}) 🔁 odds già presenti per {match_id} → skip")
            continue

        print(f"  ({idx}/{len(upcoming)}) 📡 scarico odds per {match_id} ({match_date.date()})")
        odds_data = call_api("/odds", {"fixture": match_id})
        if odds_data is None:
            print("   🚫 stop per limite API")
            break
        save_odds(conn, match_id, odds_data)
        time.sleep(SLEEP_TIME)

    print("✅ Odds pre-match aggiornate.")


# ---------------------------------------------------------
# 5) stats, events, players, lineups per match (riuso di quelle del fetch grande)
# ---------------------------------------------------------
def insert_team_stats(conn, fixture_id):
    data = call_api("/fixtures/statistics", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    # cancelliamo e riscriviamo per quella partita
    c.execute("DELETE FROM team_stats WHERE match_id = ?", (fixture_id,))
    for team_block in data:
        team_id = team_block.get("team", {}).get("id")
        team_name = team_block.get("team", {}).get("name")
        for stat in team_block.get("statistics", []):
            c.execute("""
                INSERT INTO team_stats (match_id, team_id, team_name, stat_type, stat_value)
                VALUES (?, ?, ?, ?, ?)
            """, (
                fixture_id,
                team_id,
                team_name,
                stat.get("type"),
                str(stat.get("value")) if stat.get("value") is not None else None
            ))
    conn.commit()


def insert_events(conn, fixture_id):
    data = call_api("/fixtures/events", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    c.execute("DELETE FROM events WHERE match_id = ?", (fixture_id,))
    for ev in data:
        c.execute("""
            INSERT INTO events (
                match_id, time_elapsed, team_id, team_name,
                player_id, player_name, assist_name, type, detail, comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id,
            ev.get("time", {}).get("elapsed"),
            ev.get("team", {}).get("id"),
            ev.get("team", {}).get("name"),
            ev.get("player", {}).get("id"),
            ev.get("player", {}).get("name"),
            ev.get("assist", {}).get("name"),
            ev.get("type"),
            ev.get("detail"),
            ev.get("comments")
        ))
    conn.commit()


def insert_lineups(conn, fixture_id):
    data = call_api("/fixtures/lineups", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    c.execute("DELETE FROM lineups WHERE match_id = ?", (fixture_id,))
    for team_block in data:
        team = team_block.get("team", {})
        formation = team_block.get("formation")
        coach = team_block.get("coach", {})
        # titolari
        for player in team_block.get("startXI", []):
            p = player.get("player", {})
            c.execute("""
                INSERT INTO lineups (match_id, team_id, team_name, formation, coach_name,
                                     player_id, player_name, position, is_starter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                fixture_id,
                team.get("id"),
                team.get("name"),
                formation,
                coach.get("name"),
                p.get("id"),
                p.get("name"),
                p.get("pos")
            ))
        # panchina
        for player in team_block.get("substitutes", []):
            p = player.get("player", {})
            c.execute("""
                INSERT INTO lineups (match_id, team_id, team_name, formation, coach_name,
                                     player_id, player_name, position, is_starter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                fixture_id,
                team.get("id"),
                team.get("name"),
                formation,
                coach.get("name"),
                p.get("id"),
                p.get("name"),
                p.get("pos")
            ))
    conn.commit()


def insert_players(conn, fixture_id):
    data = call_api("/fixtures/players", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
    c.execute("DELETE FROM players WHERE match_id = ?", (fixture_id,))
    for team_block in data:
        team = team_block.get("team", {})
        for p in team_block.get("players", []):
            player = p.get("player", {})
            stats = p.get("statistics", [{}])[0]
            c.execute("""
                INSERT INTO players (
                    match_id, team_id, player_id, player_name,
                    position, minutes, rating,
                    shots_total, shots_on,
                    goals_total, assists,
                    passes_total, passes_key,
                    tackles, interceptions,
                    duels_total, duels_won,
                    yellow_cards, red_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id,
                team.get("id"),
                player.get("id"),
                player.get("name"),
                stats.get("games", {}).get("position"),
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
                stats.get("cards", {}).get("red"),
            ))
    conn.commit()

# ---------------------------------------------------------
# 6) aggiorna il campo "round" (giornata di campionato)
# ---------------------------------------------------------
def update_rounds(conn):
    c = conn.cursor()
    # prendo stagioni distinte già presenti nel DB
    c.execute("SELECT DISTINCT season FROM matches ORDER BY season;")
    seasons = [row[0] for row in c.fetchall()]
    print(f"🏁 Aggiornamento round per stagioni: {seasons}")

    for season in seasons:
        print(f"📅 Aggiorno round per stagione {season}...")
        fixtures = call_api("/fixtures", {"league": LEAGUE_ID, "season": season})
        if not fixtures:
            print(f"⚠️ Nessun dato API per stagione {season}.")
            continue

        updated = 0
        for fx in fixtures:
            match_id = fx.get("fixture", {}).get("id")
            round_val = fx.get("league", {}).get("round")
            if not match_id or not round_val:
                continue
            c.execute("UPDATE matches SET round = ? WHERE match_id = ?", (round_val, match_id))
            updated += 1

        conn.commit()
        print(f"   ✅ {updated} match aggiornati per stagione {season}")

    print("✅ Round aggiornati correttamente.")

# ---------------------------------------------------------
# helper
# ---------------------------------------------------------
def current_season():
    # oggi è 2025-10 → siamo nella stagione 2025
    today = datetime.now(timezone.utc)
    year = today.year
    # se siamo in estate puoi aggiungere logica, ma per la Serie A è 1:1
    return year


def save_odds(conn, match_id, odds_data):
    """stessa logica del fetch_api_data.py, ma per fixture singolo"""
    if not odds_data:
        return
    c = conn.cursor()
    for m in odds_data:
        fixture_id = m.get("fixture", {}).get("id", match_id)
        for bookmaker in m.get("bookmakers", []):
            for bet in bookmaker.get("bets", []):
                market = bet.get("name")
                for val in bet.get("values", []):
                    c.execute("""
                        INSERT INTO odds (
                            match_id, bookmaker_id, bookmaker_name,
                            market, outcome, odd
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id,
                        bookmaker.get("id"),
                        bookmaker.get("name"),
                        market,
                        val.get("value"),
                        val.get("odd")
                    ))
    conn.commit()


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    conn = connect_db()
    print("🚀 Avvio aggiornamento giornaliero Serie A")
    update_fixtures(conn)
    update_rounds(conn)
    update_standings(conn)
    update_injuries(conn)
    update_prematch_odds(conn, days_ahead=DAYS_AHEAD)
    conn.close()
    print("✅ Aggiornamento completato.")


if __name__ == "__main__":
    main()
