# src/update_team_stats.py
import sqlite3
import requests
import time
from db import DB_PATH
from dotenv import load_dotenv
import os

# Carica API key
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# --- Funzione per scaricare e salvare le statistiche complete (stesso schema di prima)
def insert_team_stats(conn, fixture_id: int):
    """Scarica e salva TUTTE le statistiche per una partita (fixture)."""
    url = f"{BASE_URL}/fixtures/statistics"
    params = {"fixture": fixture_id}

    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("response", [])
    except Exception as e:
        print(f"⚠️ Errore chiamando API per fixture {fixture_id}: {e}")
        return 0

    # Se non ci sono dati, stampa e passa oltre
    if not data:
        print(f"   ⚠️ Nessuna statistica disponibile per fixture {fixture_id}")
        return 0

    c = conn.cursor()
    # Cancella eventuali statistiche precedenti per quella partita
    c.execute("DELETE FROM team_stats WHERE match_id = ?", (fixture_id,))

    inserted = 0
    for team_data in data:
        team_info = team_data.get("team", {})
        team_id = team_info.get("id")
        team_name = team_info.get("name")
        stats = team_data.get("statistics", [])

        if not stats:
            print(f"   ⚠️ Nessuna statistica dettagliata per {team_name}")
            continue

        for stat in stats:
            stat_type = stat.get("type")
            stat_value = stat.get("value")

            c.execute("""
                INSERT INTO team_stats (
                    match_id, team_id, team_name, stat_type, stat_value
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                fixture_id,
                team_id,
                team_name,
                stat_type,
                str(stat_value) if stat_value is not None else None
            ))
            inserted += 1

    conn.commit()

    if inserted == 0:
        print(f"   ⚠️ Nessuna statistica salvata per fixture {fixture_id}")
    else:
        print(f"   ✅ {inserted} statistiche salvate per fixture {fixture_id}")

    return inserted



# --- Funzione principale ---
def update_season_stats(season: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print(f"📅 Aggiorno statistiche per la stagione {season}...")

    # prendo tutti i match_id della stagione
    c.execute("SELECT match_id FROM matches WHERE season = ?", (season,))
    matches = [row[0] for row in c.fetchall()]

    for i, fixture_id in enumerate(matches, 1):
        # controlla se ci sono già statistiche
        c.execute("SELECT COUNT(*) FROM team_stats WHERE match_id = ?", (fixture_id,))
        count = c.fetchone()[0]

        if count > 0:
            print(f"  ({i}/{len(matches)}) fixture {fixture_id}: già presenti ({count} righe) → skip")
            continue

        print(f"  ({i}/{len(matches)}) scarico statistiche fixture {fixture_id}...")
        insert_team_stats(conn, fixture_id)
        time.sleep(1.1)  # per evitare rate limit

    conn.close()
    print(f"✅ Stagione {season} aggiornata.")


if __name__ == "__main__":
    # esempio: aggiorna 2010–2018
    #for year in range(2010, 2019):
    update_season_stats(2010)
