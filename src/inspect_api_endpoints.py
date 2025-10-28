# src/api_inspector.py
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# === CONFIGURAZIONE ===
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("❌ API_FOOTBALL_KEY mancante nel file .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "api_summary.json"

LEAGUE_ID = 135  # Serie A
SEASON = 2024

# === ENDPOINTS CHIAVE ===
ENDPOINTS = {
    "leagues": f"/leagues?id={LEAGUE_ID}",
    "teams": f"/teams?league={LEAGUE_ID}&season={SEASON}",
    "standings": f"/standings?league={LEAGUE_ID}&season={SEASON}",
    "fixtures": f"/fixtures?league={LEAGUE_ID}&season={SEASON}",
    "fixtures_statistics": f"/fixtures/statistics?league={LEAGUE_ID}&season={SEASON}",
    "odds": f"/odds?league={LEAGUE_ID}&season={SEASON}",
    "bookmakers": "/odds/bookmakers",
    "players": f"/players?league={LEAGUE_ID}&season={SEASON}",
    "lineups": f"/fixtures/lineups?league={LEAGUE_ID}&season={SEASON}",
    "injuries": f"/injuries?league={LEAGUE_ID}&season={SEASON}",
    "coaches": f"/coachs?league={LEAGUE_ID}&season={SEASON}",
}

# === FUNZIONI ===
def call_api(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"⚠️ Errore su {endpoint}: {e}")
        return None


def inspect_structure(data, depth=0, max_depth=2):
    """Analizza struttura annidata fino a 2 livelli."""
    if isinstance(data, list) and data:
        return inspect_structure(data[0], depth + 1, max_depth)
    elif isinstance(data, dict) and depth < max_depth:
        return {k: inspect_structure(v, depth + 1, max_depth) for k, v in data.items()}
    else:
        return type(data).__name__


def summarize_endpoint(name, endpoint):
    print(f"📦 {name} → {endpoint}")
    data = call_api(endpoint)
    if not data:
        print(f"   ⚠️ Nessun dato disponibile")
        return None

    structure = inspect_structure(data)
    print(f"   ✅ {len(data)} record - chiavi principali: {list(structure.keys())[:8]}")
    return structure


def main():
    print("🔍 Esplorazione struttura API-Football (Serie A)\n")
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    summary = {}

    for name, endpoint in ENDPOINTS.items():
        structure = summarize_endpoint(name, endpoint)
        if structure:
            summary[name] = structure
        print("-" * 70)

    # Salvataggio in JSON leggibile
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analisi completata. Struttura salvata in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
