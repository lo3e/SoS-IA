from src.pipeline.daily_update import call_api, current_season, LEAGUE_ID
from pprint import pprint

def inspect_injuries(team_name_filter=None, max_records=3):
    print("🔍 Analisi struttura live endpoint /injuries...\n")

    data = call_api("/injuries", {"league": LEAGUE_ID, "season": current_season()})
    if not data:
        print("⚠️ Nessun dato ricevuto.")
        return

    print(f"✅ Totale record ricevuti: {len(data)}")

    # 🔹 Trova subito la prima squadra utile (es. Lazio)
    subset = [
        inj for inj in data
        if not team_name_filter or team_name_filter.lower() in str(inj.get("team", {}).get("name", "")).lower()
    ]

    if not subset:
        print(f"⚠️ Nessun record trovato per filtro: {team_name_filter}")
        return

    print(f"🎯 Mostro i primi {min(max_records, len(subset))} record per '{team_name_filter or 'tutte le squadre'}':\n")

    for i, inj in enumerate(subset[:max_records], 1):
        print(f"--- RECORD {i} ---")
        pprint(inj)
        print("\nChiavi di primo livello:", list(inj.keys()))
        print("Chiavi player:", list(inj.get("player", {}).keys()))
        print("Chiavi team:", list(inj.get("team", {}).keys()))
        print("Chiavi fixture:", list(inj.get("fixture", {}).keys()))
        print("-" * 50, "\n")

if __name__ == "__main__":
    inspect_injuries("Lazio")  # puoi anche passare None per tutte le squadre
