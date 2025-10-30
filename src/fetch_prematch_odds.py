import os
import time
import sqlite3
from datetime import datetime, timezone
import requests
from pathlib import Path
from dotenv import load_dotenv
import argparse
from dateutil import parser


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
SLEEP_TIME = 1.2  # rate safety
MAX_DAYS_AHEAD = 14  # scarica quote solo per match entro 14 giorni

def call_api(endpoint, params):
    """Chiamata API generica"""
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code == 429:
        print("🚫 Limite API raggiunto. Stop per oggi.")
        return None
    r.raise_for_status()
    return r.json().get("response", [])

def connect_db():
    return sqlite3.connect(DB_PATH)

def have_odds(c, match_id):
    """Verifica se già ci sono quote per la partita"""
    c.execute("SELECT COUNT(*) FROM odds WHERE match_id = ?", (match_id,))
    return c.fetchone()[0] > 0

def save_odds(conn, match_id, odds_data):
    """
    Salva tutte le quote (odds) per una singola partita (fixture).
    Struttura identica alla funzione insert_odds() usata per i dati storici.
    """
    if not odds_data:
        print(f"   ⚠️ Nessuna quota disponibile per match {match_id}")
        return

    c = conn.cursor()

    for m in odds_data:
        fixture_id = m.get("fixture", {}).get("id")
        for bookmaker in m.get("bookmakers", []):
            for bet in bookmaker.get("bets", []):
                market = bet.get("name")
                for val in bet.get("values", []):
                    c.execute("""
                        INSERT INTO odds (
                            match_id, bookmaker_id, bookmaker_name, market, outcome, odd
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
    print(f"   ✅ Quote salvate correttamente per match {match_id}")


def fetch_all_prematch_odds():
    parser_cli = argparse.ArgumentParser(description="Scarica tutte le pre-match odds")
    parser_cli.add_argument("--days", type=int, default=14, help="Scarica quote per match entro N giorni (default 14)")
    args = parser_cli.parse_args()

    conn = connect_db()
    c = conn.cursor()

    # Prendo tutte le partite non iniziate
    c.execute("""
        SELECT match_id, date
        FROM matches
        WHERE status = 'NS'
        ORDER BY date
    """)
    matches = c.fetchall()

    now = datetime.now(timezone.utc)
    upcoming = []
    for mid, d in matches:
        try:
            dt = parser.parse(d)  # parser robusto
            delta_days = (dt - now).days
            if 0 <= delta_days <= args.days:
                upcoming.append((mid, dt))
        except Exception as e:
            print(f"⚠️ Errore parsing data {d}: {e}")
            continue

    print(f"📅 {len(upcoming)} partite 'NS' entro {args.days} giorni trovate")

    new_odds = 0
    skipped = 0

    for idx, (match_id, match_date) in enumerate(upcoming, start=1):
        print(f"\n({idx}/{len(upcoming)}) ▶️ Match {match_id} ({match_date.date()})")

        if have_odds(c, match_id):
            print("   🔁 Odds già presenti → skip")
            skipped += 1
            continue

        print("   📡 Chiamata all’API /odds ...")
        odds_data = call_api("/odds", {"fixture": match_id})
        print(f"   📦 {len(odds_data[0].get('bookmakers', []))} bookmaker trovati per match {match_id}")

        if odds_data is None:
            print("   🚫 Stop: limite API raggiunto.")
            break

        if odds_data:
            save_odds(conn, match_id, odds_data)
            print(f"   ✅ Salvate {len(odds_data)} bookmaker con mercati multipli.")
            new_odds += 1
        else:
            print("   ⚠️ Nessuna odds disponibile (ancora troppo presto)")

        time.sleep(SLEEP_TIME)

    print("\n📊 RIEPILOGO:")
    print(f"   🟢 Match con nuove odds: {new_odds}")
    print(f"   🟡 Match già coperti: {skipped}")
    print(f"   ⏱ Totale analizzati: {len(upcoming)}")
    conn.close()
    print("✅ Fine script.")

if __name__ == "__main__":
    fetch_all_prematch_odds()
