# src/inspect_fixture_details.py
import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# === CONFIG === #
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("❌ API_FOOTBALL_KEY mancante nel file .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fixture_inspection.json"

LEAGUE_ID = 135   # Serie A
SEASON = 2025     # stagione corrente

# === Helper === #
def call_api(endpoint: str):
    """Chiama l’endpoint API e restituisce la risposta 'response'."""
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as e:
        print(f"⚠️ Errore su {endpoint}: {e}")
        return None


def pick_fixture():
    """Sceglie una partita valida da usare come riferimento."""
    fixtures = call_api(f"/fixtures?league={LEAGUE_ID}&season={SEASON}")
    valid = [f for f in fixtures if f.get("teams", {}).get("home") and f.get("teams", {}).get("away")]
    if not valid:
        raise ValueError("❌ Nessuna partita trovata.")
    return valid[0]["fixture"]["id"], valid[0]


def compact_stats(stats):
    """Estrae solo le statistiche utili per il modello Poisson."""
    useful_keys = {"Shots on Goal", "Shots off Goal", "Total Shots", "Expected Goals", "Ball Possession", "Fouls", "Yellow Cards", "Red Cards"}
    filtered = {}
    for s in stats:
        team = s.get("team", {}).get("name")
        if not team:
            continue
        team_stats = {i["type"]: i["value"] for i in s.get("statistics", []) if i["type"] in useful_keys}
        filtered[team] = team_stats
    return filtered


def compact_odds(odds):
    """Estrae le quote 1X2 da tutti i bookmaker disponibili."""
    all_odds = {}
    for o in odds:
        bookmaker = o.get("bookmaker", {}).get("name")
        markets = o.get("bookmaker", {}).get("bets", [])
        for bet in o.get("bets", []):
            if bet.get("name") == "Match Winner":
                outcomes = {v["label"]: v["odd"] for v in bet["values"]}
                all_odds[bookmaker] = outcomes
    return all_odds


def main():
    print("🔍 Analisi dettagliata fixture API-Football (Serie A 2025)")
    fixture_id, fixture_data = pick_fixture()
    print(f"📌 Partita di riferimento: {fixture_id} ({fixture_data['teams']['home']['name']} vs {fixture_data['teams']['away']['name']})")

    results = {"fixture_id": fixture_id}

    # === Fixture base === #
    results["fixture_info"] = {
        "date": fixture_data["fixture"]["date"],
        "status": fixture_data["fixture"]["status"]["short"],
        "season": fixture_data["league"]["season"],
        "home": fixture_data["teams"]["home"]["name"],
        "away": fixture_data["teams"]["away"]["name"],
        "goals_home": fixture_data["goals"]["home"],
        "goals_away": fixture_data["goals"]["away"],
    }

    # === Statistiche === #
    stats = call_api(f"/fixtures/statistics?fixture={fixture_id}")
    if stats:
        results["statistics"] = compact_stats(stats)
    else:
        results["statistics"] = None

    # === Odds === #
    odds = call_api(f"/odds?fixture={fixture_id}")
    if odds:
        results["odds"] = compact_odds(odds)
    else:
        results["odds"] = None

    # === Predictions === #
    preds = call_api(f"/predictions?fixture={fixture_id}")
    if preds:
        p = preds[0].get("predictions", {})
        results["predictions"] = {
            "winner": p.get("winner", {}).get("name"),
            "prob_home": p.get("percent", {}).get("home"),
            "prob_draw": p.get("percent", {}).get("draw"),
            "prob_away": p.get("percent", {}).get("away"),
        }
    else:
        results["predictions"] = None

    # === Lineups (facoltativo) === #
    lineups = call_api(f"/fixtures/lineups?fixture={fixture_id}")
    if lineups:
        results["lineups_info"] = {
            team["team"]["name"]: team["formation"] for team in lineups
        }
    else:
        results["lineups_info"] = None

    # === Salvataggio === #
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analisi completata. File salvato in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
