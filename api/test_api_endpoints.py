import os
import requests
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 135  # Serie A
SEASON = 2020

def call_api(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json().get("response", [])

def preview(data, depth=1, max_depth=2):
    """Mostra le chiavi principali in modo leggibile"""
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and depth <= max_depth:
        return {k: preview(v, depth + 1, max_depth) for k, v in data.items()}
    return type(data).__name__

def main():
    print(f"🔍 Analisi API-Football - Serie A {SEASON}")
    print("--------------------------------------------------")

    # 1️⃣ Recupera una partita della stagione
    fixtures = call_api("/fixtures", {"league": LEAGUE_ID, "season": SEASON})
    if not fixtures:
        print("❌ Nessuna partita trovata.")
        return
    first_match = fixtures[0]
    fixture_id = first_match["fixture"]["id"]
    home = first_match["teams"]["home"]["name"]
    away = first_match["teams"]["away"]["name"]
    home_id = first_match["teams"]["home"]["id"]
    away_id = first_match["teams"]["away"]["id"]
    print(f"🎯 Test su partita: {home} vs {away} (id={fixture_id})\n")

    endpoints = {
        "statistics": f"/fixtures/statistics?fixture={fixture_id}",
        "predictions": f"/predictions?fixture={fixture_id}",
        "lineups": f"/fixtures/lineups?fixture={fixture_id}",
        "players": f"/fixtures/players?fixture={fixture_id}",
        "events": f"/fixtures/events?fixture={fixture_id}",
        "head2head": f"/fixtures/headtohead?h2h={home_id}-{away_id}&league={LEAGUE_ID}&season={SEASON}",
        "injuries": f"/injuries?league={LEAGUE_ID}&season={SEASON}"
    }

    for name, endpoint in endpoints.items():
        print(f"📦 {name.upper()} → {endpoint}")
        try:
            data = call_api(endpoint)
            if not data:
                print("   ⚠️ Nessun dato disponibile.")
            else:
                print(f"   ✅ {len(data)} record trovati")
                print("   🔑 Chiavi principali:", list(data[0].keys())[:10])
                print("   🧱 Struttura:", preview(data))
        except Exception as e:
            print(f"   ❌ Errore chiamando {name}: {e}")
        print("-" * 70)

if __name__ == "__main__":
    main()
