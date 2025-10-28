# src/db.py
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili dal file .env
load_dotenv()

DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
    -- Tabella partite (fixture)
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY,
        date TEXT,
        season INTEGER,
        league TEXT,
        home TEXT,
        away TEXT,
        home_goals INTEGER,
        away_goals INTEGER,
        status TEXT
    );

    -- Quote da diversi bookmaker
    CREATE TABLE IF NOT EXISTS odds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        bookmaker TEXT,
        odds_1 REAL,
        odds_x REAL,
        odds_2 REAL,
        last_update TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- Quote aggregate (media tra bookmaker)
    CREATE TABLE IF NOT EXISTS aggregated_odds (
        match_id INTEGER PRIMARY KEY,
        avg_odds_1 REAL,
        avg_odds_x REAL,
        avg_odds_2 REAL,
        last_update TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- Statistiche squadra per match
    CREATE TABLE IF NOT EXISTS team_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        team TEXT,
        xG REAL,
        shots INTEGER,
        possession REAL,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- Probabilità del modello Poisson
    CREATE TABLE IF NOT EXISTS model_probs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        p_home REAL,
        p_draw REAL,
        p_away REAL,
        timestamp TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- Storico scommesse
    CREATE TABLE IF NOT EXISTS bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        pick TEXT,
        odds REAL,
        stake REAL,
        ev REAL,
        placed_at TEXT,
        outcome TEXT,
        profit REAL,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    -- Squadre uniche
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    );
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("✅ Database inizializzato:", DB_PATH)
