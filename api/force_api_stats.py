import sqlite3
import requests
from db import DB_PATH
from dotenv import load_dotenv
import os
import time

# Carica chiave API
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

def insert_team_stats(conn, fixture_id: int):
    """Scarica e salva TUTTE le statistiche per un dato fixture, sovrascrivendo quelle esistenti."""
    url = f"{BASE_URL}/fixtures/statistics"
    params = {"fixture": fixture_id}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("response", [])
    except Exception as e:
        print(f"⚠️ Errore fixture {fixture_id}: {e}")
        return 0

    if not data:
        print(f"   ⚠️ Nessuna statistica disponibile per fixture {fixture_id}")
        return 0

    c = conn.cursor()
    c.execute("DELETE FROM team_stats WHERE match_id = ?", (fixture_id,))
    inserted = 0

    for team_data in data:
        team_info = team_data.get("team", {})
        team_id = team_info.get("id")
        team_name = team_info.get("name")
        stats = team_data.get("statistics", [])
        for s in stats:
            stat_type = s.get("type")
            stat_value = s.get("value")
            c.execute("""
                INSERT INTO team_stats (match_id, team_id, team_name, stat_type, stat_value)
                VALUES (?, ?, ?, ?, ?)
            """, (fixture_id, team_id, team_name, stat_type, str(stat_value) if stat_value is not None else None))
            inserted += 1

    conn.commit()
    print(f"   ✅ {inserted} statistiche salvate per fixture {fixture_id}")
    return inserted


def force_update_stats_for_season(season: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print(f"\n📅 Aggiorno statistiche (forzato) per stagione {season}...")

    c.execute("SELECT match_id FROM matches WHERE season=?", (season,))
    fixtures = [row[0] for row in c.fetchall()]
    print(f"🔹 {len(fixtures)} partite trovate per la stagione {season}")

    for i, fixture_id in enumerate(fixtures, start=1):
        print(f"  ({i}/{len(fixtures)}) fixture {fixture_id} → scarico statistiche...")
        insert_team_stats(conn, fixture_id)
        time.sleep(1.2)  # anti-rate limit

    conn.close()
    print(f"✅ Completato aggiornamento stagione {season}")


if __name__ == "__main__":
    # Esegui su una stagione alla volta
    #for s in [2015, 2016, 2017, 2018]:
    force_update_stats_for_season(2016)
