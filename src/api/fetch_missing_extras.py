# src/fetch_missing_extras.py
import os
import time
import sqlite3
import requests
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
LEAGUE_ID = 135
SLEEP_TIME = 1.0

def call_api(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    if r.status_code == 429:
        print("🚫 Limite API raggiunto — stop per salvare credito.")
        return None
    r.raise_for_status()
    return r.json().get("response", [])

def connect_db():
    return sqlite3.connect(DB_PATH)

# === CHECK SE GIA PRESENTI ===
def have_prediction(c, match_id):
    c.execute("SELECT COUNT(*) FROM predictions WHERE match_id = ?", (match_id,))
    return c.fetchone()[0] > 0

def have_head2head(c, home_id, away_id, season):
    c.execute("""
        SELECT COUNT(*) FROM head2head
        WHERE home_team_id = ? AND away_team_id = ? AND season = ?
    """, (home_id, away_id, season))
    return c.fetchone()[0] > 0

def have_injuries(c, fixture_id):
    c.execute("SELECT COUNT(*) FROM injuries WHERE match_id = ?", (fixture_id,))
    return c.fetchone()[0] > 0

# === SALVATAGGI ===
def save_prediction(conn, match_id, pred):
    p = pred.get("predictions", {})
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
    c.execute("""
        INSERT INTO predictions (
            match_id, winner, win_or_draw, advice,
            prob_home, prob_draw, prob_away
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

def save_head2head(conn, home_id, away_id, season, data):
    if not data:
        return
    c = conn.cursor()
    for match in data:
        f = match.get("fixture", {})
        g = match.get("goals", {})
        c.execute("""
            INSERT OR IGNORE INTO head2head (
                home_team_id, away_team_id, match_id,
                home_goals, away_goals, season
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            home_id, away_id, f.get("id"), g.get("home"), g.get("away"), season
        ))
    conn.commit()

def save_injuries(conn, fixture_id, data):
    if not data:
        return
    c = conn.cursor()
    for inj in data:
        c.execute("""
            INSERT OR IGNORE INTO injuries (
                match_id, player_id, player_name, team_id,
                reason, since, expected_return
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

# === PROCESSO PRINCIPALE ===
def complete_missing_for_season(season: int):
    conn = connect_db()
    c = conn.cursor()

    print(f"\n🧩 Completo dati mancanti per stagione {season} (Serie A)")
    c.execute("SELECT match_id, home_team_id, away_team_id FROM matches WHERE season = ?", (season,))
    fixtures = c.fetchall()

    for idx, (match_id, home_id, away_id) in enumerate(fixtures, start=1):
        print(f"\n({idx}/{len(fixtures)}) ▶️ Match {match_id} {home_id} vs {away_id}")

        # === PREDICTIONS ===
        if have_prediction(c, match_id):
            print("   🔁 Prediction già presente")
        else:
            data = call_api("/predictions", {"fixture": match_id})
            if data is None:
                print("   🚫 Stop API — limite raggiunto.")
                break
            if data:
                save_prediction(conn, match_id, data[0])
                print("   ✅ Prediction salvata")
            else:
                print("   ⚠️ Nessuna prediction")
            time.sleep(SLEEP_TIME)

        # === HEAD2HEAD ===
        if not home_id or not away_id:
            print("   ⚠️ Head2Head skip (ID mancanti)")
        elif have_head2head(c, home_id, away_id, season):
            print("   🔁 Head2Head già presente")
        else:
            data = call_api("/fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "league": LEAGUE_ID, "season": season})
            if data is None:
                print("   🚫 Stop API — limite raggiunto.")
                break
            if data:
                save_head2head(conn, home_id, away_id, season, data)
                print("   ✅ Head2Head salvato")
            else:
                print("   ⚠️ Nessun head2head")
            time.sleep(SLEEP_TIME)

        # === INJURIES (una sola volta per stagione, come nello script principale) ===
        print(f"\n💊 Controllo infortuni per la stagione {season} ...")
        c.execute("""
            SELECT COUNT(*) FROM injuries
            WHERE match_id IN (
                SELECT match_id FROM matches WHERE season = ?
            )
        """, (season,))
        count_inj = c.fetchone()[0]

        if count_inj > 0:
            print(f"   🔁 Injuries già presenti → {count_inj} record trovati → skip")
        else:
            print("   📡 Scarico injuries stagione intera ...")
            data_inj = call_api("/injuries", {"league": LEAGUE_ID, "season": season})
            if data_inj is None:
                print("   🚫 Limite API raggiunto — fermo qui per non bruciare altre call.")
            elif not data_inj:
                print("   ⚠️ Nessun infortunio restituito dall'API per questa stagione")
            else:
                for inj in data_inj:
                    c.execute("""
                        INSERT OR IGNORE INTO injuries (
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
                print(f"   ✅ Salvati {len(data_inj)} infortuni per la stagione {season}")

    conn.close()
    print(f"\n✅ Stagione {season} completata (solo dati mancanti aggiornati).")

# === ENTRYPOINT ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python fetch_missing_extras.py <season>")
        sys.exit(1)
    season = int(sys.argv[1])
    complete_missing_for_season(season)
