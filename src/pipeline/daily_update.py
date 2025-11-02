# src/pipeline/daily_update.py
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# <-- NUOVI IMPORT DAL CORE/NUOVI MODULI -->
from src.core.config import DB_PATH
from src.core.logger import get_logger
from src.data.classify_injuries import add_category_column, classify_injuries, create_view
from src.pipeline.optimize_db import optimize_database
from src.data.prepare_training_data_v2 import build_dataset
from src.training.train_model_optimized import train_with_cutoff
from src.evaluation.evaluate_2025 import evaluate_on_round
from src.betting.bolletta import generate_for_round

load_dotenv()
logger = get_logger(__name__)

API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("API_FOOTBALL_KEY mancante nel .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 135  # Serie A
SLEEP_TIME = 1.0
DAYS_AHEAD = 14  # orizzonte partite future


# ---------------------------------------------------------
# utility API
# ---------------------------------------------------------
def call_api(endpoint: str, params: dict = None):
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=25)
        if r.status_code == 429:
            logger.warning("🚫 Rate limit raggiunto.")
            return None
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as e:
        logger.exception(f"Errore API {endpoint}: {e}")
        return None


def current_season() -> int:
    # API-Football usa anno solare
    return datetime.now(timezone.utc).year


# ---------------------------------------------------------
# 1) fixtures da aggiornare
# ---------------------------------------------------------
def update_fixtures(conn):
    c = conn.cursor()
    c.execute("""
        SELECT match_id, date
        FROM matches
        WHERE status != 'FT'
           OR date(date) >= date('now', '-3 days')
    """)
    rows = c.fetchall()
    if not rows:
        logger.info("📭 Nessuna fixture da aggiornare.")
        return

    now = datetime.now(timezone.utc)
    to_update = []

    for match_id, date_str in rows:
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "")).replace(tzinfo=timezone.utc)
            delta_days = (dt - now).days
            if -3 <= delta_days <= DAYS_AHEAD:
                to_update.append(match_id)
        except Exception:
            continue

    logger.info(f"📅 Fixtures da aggiornare (-3 → +{DAYS_AHEAD} giorni): {len(to_update)}")

    for idx, fixture_id in enumerate(to_update, start=1):
        logger.info(f"  ({idx}/{len(to_update)}) 🔁 fixture {fixture_id}")
        data = call_api("/fixtures", {"id": fixture_id})
        if not data:
            continue
        f = data[0]
        fixture = f.get("fixture", {})
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
            fixture_id,
        ))

        # se è FT → scarico stats/eventi/lineups/players
        if fixture.get("status", {}).get("short") == "FT":
            insert_team_stats(conn, fixture_id)
            insert_events(conn, fixture_id)
            insert_lineups(conn, fixture_id)
            insert_players(conn, fixture_id)

        time.sleep(SLEEP_TIME)

    conn.commit()
    logger.info("✅ Fixtures aggiornate correttamente.")


# ---------------------------------------------------------
# 2) standings
# ---------------------------------------------------------
def update_standings(conn):
    data = call_api("/standings", {"league": LEAGUE_ID, "season": current_season()})
    if not data:
        logger.warning("⚠️ Nessuna standings aggiornata.")
        return
    c = conn.cursor()
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
            stats.get("goals", {}).get("against"),
        ))

    conn.commit()
    logger.info("✅ Standings aggiornate.")


# ---------------------------------------------------------
# 3) injuries
# ---------------------------------------------------------
def update_injuries(conn):
    data = call_api("/injuries", {"league": LEAGUE_ID, "season": current_season()})
    if not data:
        logger.info("⚠️ Nessun infortunio nuovo.")
        return

    c = conn.cursor()
    inserted = 0
    for inj in data:
        fixture_id = inj.get("fixture", {}).get("id")
        player_id = inj.get("player", {}).get("id")
        team_id = inj.get("team", {}).get("id")

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
            inj.get("player", {}).get("reason"),
        ))
        inserted += 1

    conn.commit()

    # <-- QUI USIAMO IL NUOVO MODULO DI CLASSIFICAZIONE -->
    add_category_column(conn)
    classify_injuries(conn)
    create_view(conn)

    logger.info(f"✅ Infortuni aggiornati (+{inserted} nuovi)")


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
        logger.info("📭 Nessuna partita NS trovata.")
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

    logger.info(f"🎯 Partite NS entro {days_ahead} giorni: {len(upcoming)}")

    for idx, (match_id, match_date) in enumerate(upcoming, start=1):
        c.execute("SELECT COUNT(*) FROM odds WHERE match_id = ?", (match_id,))
        if c.fetchone()[0] > 0:
            logger.info(f"  ({idx}/{len(upcoming)}) 🔁 odds già presenti per {match_id} → skip")
            continue

        logger.info(f"  ({idx}/{len(upcoming)}) 📡 scarico odds per {match_id} ({match_date.date()})")
        odds_data = call_api("/odds", {"fixture": match_id})
        if odds_data is None:
            logger.warning("🚫 stop per limite API")
            break
        save_odds(conn, match_id, odds_data)
        time.sleep(SLEEP_TIME)

    logger.info("✅ Odds pre-match aggiornate.")


# ---------------------------------------------------------
# 5) dettagli match (stats / events / lineups / players)
#    (ripresi pari pari dal tuo file originale)
# ---------------------------------------------------------
def insert_team_stats(conn, fixture_id):
    data = call_api("/fixtures/statistics", {"fixture": fixture_id})
    if not data:
        return
    c = conn.cursor()
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
                str(stat.get("value")) if stat.get("value") is not None else None,
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
            ev.get("comments"),
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
                INSERT INTO lineups (
                    match_id, team_id, team_name, formation, coach_name,
                    player_id, player_name, position, is_starter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                fixture_id,
                team.get("id"),
                team.get("name"),
                formation,
                coach.get("name"),
                p.get("id"),
                p.get("name"),
                p.get("pos"),
            ))
        # panchina
        for player in team_block.get("substitutes", []):
            p = player.get("player", {})
            c.execute("""
                INSERT INTO lineups (
                    match_id, team_id, team_name, formation, coach_name,
                    player_id, player_name, position, is_starter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                fixture_id,
                team.get("id"),
                team.get("name"),
                formation,
                coach.get("name"),
                p.get("id"),
                p.get("name"),
                p.get("pos"),
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
                    minutes, rating,
                    shots_total, shots_on,
                    goals_total, assists,
                    passes_total, passes_key,
                    tackles, interceptions,
                    duels_total, duels_won,
                    yellow_cards, red_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id,
                team.get("id"),
                player.get("id"),
                player.get("name"),
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


def save_odds(conn, match_id, odds_data):
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
# funzioni di supporto per ML
# ---------------------------------------------------------
def get_last_completed_round(conn, season: int):
    c = conn.cursor()
    c.execute("""
        SELECT round
        FROM matches
        WHERE season = ? AND status = 'FT'
        GROUP BY round
        ORDER BY MAX(date) DESC
        LIMIT 1;
    """, (season,))
    row = c.fetchone()
    return row[0] if row else None


def get_next_round(conn, season: int):
    c = conn.cursor()
    c.execute("""
        SELECT round
        FROM matches
        WHERE season = ? AND status = 'NS'
        ORDER BY date ASC
        LIMIT 1;
    """, (season,))
    row = c.fetchone()
    return row[0] if row else None


def round_to_int(round_str: str | None) -> int | None:
    if not round_str:
        return None
    parts = round_str.split("-")
    try:
        return int(parts[-1].strip())
    except ValueError:
        return None


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    logger.info("🚀 Avvio DAILY UPDATE SoS-IA")

    # 1) aggiornamento dati da API -> DB
    conn = sqlite3.connect(DB_PATH)
    update_fixtures(conn)
    update_standings(conn)
    update_injuries(conn)
    update_prematch_odds(conn, days_ahead=DAYS_AHEAD)
    conn.close()
    logger.info("✅ Aggiornamento API → DB completato.")

    # 2) Ottimizza DB dopo aggiornamenti
    logger.info("🧠 Ottimizzo e ricreo viste post-aggiornamento...")
    optimize_database()

    # 3) --- PARTE ML NUOVA ---
    conn = sqlite3.connect(DB_PATH)
    season = current_season()

    last_round_str = get_last_completed_round(conn, season)
    next_round_str = get_next_round(conn, season)
    conn.close()

    last_round_int = round_to_int(last_round_str) if last_round_str else None
    next_round_int = round_to_int(next_round_str) if next_round_str else None

    if last_round_int and last_round_int > 2:
        cutoff_round = last_round_int - 2  # allena fino a due giornate prima
        eval_round = last_round_int - 1    # valuta sulla penultima
    else:
        cutoff_round = None
        eval_round = None

    logger.info(f"📐 ML: cutoff={cutoff_round}, eval_round={eval_round}, next_round={next_round_int}")

    # 3a) rigenero dataset
    build_dataset(min_season=2021, max_season=season, cutoff_round=cutoff_round)
    _, dataset_path = build_dataset(min_season=2021, max_season=current_season(), cutoff_round=cutoff_round)

    # 3b) training
    model_path, meta_path = train_with_cutoff(cutoff_round=cutoff_round, min_season=2021, dataset_path=dataset_path)

    # 3c) evaluation
    if eval_round:
        evaluate_on_round(round_number=eval_round, season=season)
    else:
        logger.warning("⚠️ Nessun round di valutazione disponibile (stagione troppo all’inizio).")

    # 3d) bolletta
    if next_round_int:
        generate_for_round(next_round_int, season=season, mode="pure", model_path=model_path)
        generate_for_round(next_round_int, season=season, mode="value", model_path=model_path)
    else:
        logger.warning("⚠️ Nessuna prossima giornata trovata per generare la bolletta.")

    logger.info("✅ DAILY UPDATE SoS-IA COMPLETATO.")


if __name__ == "__main__":
    main()
