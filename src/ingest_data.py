import os
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from db import DB_PATH

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

def normalize_date(date_str):
    """Converte date in formato YYYY-MM-DD"""
    try:
        return pd.to_datetime(date_str, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None

def infer_season(date):
    """Determina la stagione (es. 2023/2024) a partire dalla data"""
    year = int(date[:4])
    month = int(date[5:7])
    if month >= 7:
        return f"{year}/{year+1}"
    else:
        return f"{year-1}/{year}"

def ingest_file(csv_path):
    df = pd.read_csv(csv_path)
    print(f"📂 Importing {os.path.basename(csv_path)} ({len(df)} rows)")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    inserted_matches = 0

    for _, row in df.iterrows():
        date = normalize_date(row["Date"])
        if not date:
            continue

        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        match_id = f"{date}_{home}_{away}"
        season = infer_season(date)

        # --- Risultati finali ---
        home_goals = row.get("FTHG", None)
        away_goals = row.get("FTAG", None)
        result = row.get("FTR", None)  # 1, X, 2

        # --- Inserisci squadre ---
        for team in [home, away]:
            c.execute(
                "INSERT OR IGNORE INTO teams (name, season) VALUES (?, ?)",
                (team, season),
            )

        # --- Inserisci match ---
        c.execute(
            """INSERT OR REPLACE INTO matches 
               (match_id, date, season, home, away, competition, home_goals, away_goals, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (match_id, date, season, home, away, "Serie A", home_goals, away_goals, result),
        )

        # --- Quote principali (Bet365) ---
        odds_h, odds_d, odds_a = (
            row.get("B365H"),
            row.get("B365D"),
            row.get("B365A"),
        )

        if pd.notna(odds_h) and pd.notna(odds_d) and pd.notna(odds_a):
            c.execute(
                """INSERT INTO odds (match_id, source, timestamp, odds_1, odds_x, odds_2)
                   VALUES (?, ?, datetime('now'), ?, ?, ?)""",
                (match_id, "Bet365", odds_h, odds_d, odds_a),
            )

        inserted_matches += 1

    conn.commit()
    conn.close()

    print(f"✅ {inserted_matches} match importati da {os.path.basename(csv_path)}")

def main():
    files = [
        f for f in Path(DATA_PATH).glob("*.csv")
        if "SerieA" in f.name or "ITA" in f.name or "I1" in f.name
    ]
    if not files:
        print("⚠️ Nessun file CSV trovato in", DATA_PATH)
        return

    for file in sorted(files):
        ingest_file(file)

    print("\n🏁 Ingest completato.\n")

if __name__ == "__main__":
    main()
