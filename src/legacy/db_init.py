# src/db.py
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
    -- === MATCHES ===
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY,
        date TEXT,
        season INTEGER,
        league_id INTEGER,
        league_name TEXT,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_team_name TEXT,
        away_team_name TEXT,
        home_goals INTEGER,
        away_goals INTEGER,
        status TEXT,
        venue_id INTEGER,
        last_update TEXT
    );

    -- === TEAMS ===
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT,
        country TEXT,
        founded INTEGER,
        last_update TEXT
    );

    -- === TEAM STATS ===
    CREATE TABLE IF NOT EXISTS team_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        team_id INTEGER,
        team_name TEXT,
        stat_type TEXT,
        stat_value TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === PLAYERS ===
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        team_id INTEGER,
        player_id INTEGER,
        player_name TEXT,
        position TEXT,
        minutes INTEGER,
        rating REAL,
        shots_total INTEGER,
        shots_on INTEGER,
        goals_total INTEGER,
        assists INTEGER,
        passes_total INTEGER,
        passes_key INTEGER,
        tackles INTEGER,
        interceptions INTEGER,
        duels_total INTEGER,
        duels_won INTEGER,
        yellow_cards INTEGER,
        red_cards INTEGER,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === LINEUPS ===
    CREATE TABLE IF NOT EXISTS lineups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        team_id INTEGER,
        team_name TEXT,
        formation TEXT,
        coach_name TEXT,
        player_id INTEGER,
        player_name TEXT,
        position TEXT,
        is_starter INTEGER,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === EVENTS ===
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        time_elapsed INTEGER,
        team_id INTEGER,
        team_name TEXT,
        player_id INTEGER,
        player_name TEXT,
        assist_name TEXT,
        type TEXT,
        detail TEXT,
        comments TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === PREDICTIONS ===
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        winner TEXT,
        win_or_draw INTEGER,
        advice TEXT,
        prob_home REAL,
        prob_draw REAL,
        prob_away REAL,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === HEAD2HEAD ===
    CREATE TABLE IF NOT EXISTS head2head (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        home_team_id INTEGER,
        away_team_id INTEGER,
        match_id INTEGER,
        home_goals INTEGER,
        away_goals INTEGER,
        season INTEGER,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === INJURIES ===
    CREATE TABLE IF NOT EXISTS injuries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        player_id INTEGER,
        player_name TEXT,
        team_id INTEGER,
        reason TEXT,
        since TEXT,
        expected_return TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- === ODDS ===
    CREATE TABLE IF NOT EXISTS odds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        bookmaker_id INTEGER,
        bookmaker_name TEXT,
        market TEXT,
        outcome TEXT,
        odd REAL,
        timestamp TEXT,
        source TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("✅ Database definitivo inizializzato:", DB_PATH)
