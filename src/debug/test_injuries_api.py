from datetime import datetime, timedelta, UTC
from src.pipeline.daily_update import call_api, current_season, LEAGUE_ID
from src.core.logger import get_logger
import pandas as pd

logger = get_logger("injuries_preview")

def preview_injuries(team_filter=None, only_active=True):
    """
    Mostra infortuni/assenze recenti o attuali (senza modificare il DB)
    leggendo correttamente il motivo da player.reason e player.type.
    """
    logger.info("🏥 Recupero infortuni in corso...")
    season = current_season()
    cutoff_date = datetime.now(UTC) - timedelta(days=20)

    data = call_api("/injuries", {"league": LEAGUE_ID, "season": season})
    if not data:
        logger.warning("⚠️ Nessun dato ricevuto dall'API.")
        return

    rows = []
    now = datetime.now(UTC)
    for inj in data:
        fixture = inj.get("fixture", {})
        fixture_date = fixture.get("date") or fixture.get("timestamp")
        date = None

        # 🕒 Parse robusto della data
        if fixture_date:
            try:
                date = datetime.fromisoformat(str(fixture_date).replace("Z", "+00:00"))
            except Exception:
                date = now

        player = inj.get("player", {})
        team = inj.get("team", {})
        team_name = team.get("name", "Unknown Team")

        if team_filter and team_filter.lower() not in team_name.lower():
            continue

        # ✅ Prende motivo e tipo direttamente dal player
        reason = player.get("reason") or "Unknown"
        abs_type = player.get("type") or "Unknown"

        # Filtriamo solo quelli "attivi" o recenti
        if only_active and date and date < cutoff_date:
            continue

        status = "🕒 Future" if date and date > now else "🔥 Active"

        rows.append({
            "status": status,
            "date": date.date().isoformat() if date else "n/d",
            "player_name": player.get("name"),
            "team_name": team_name,
            "type": abs_type,
            "reason": reason,
        })

    if not rows:
        logger.info("😶 Nessun infortunio trovato.")
        return

    df = pd.DataFrame(rows).sort_values("date", ascending=False)
    logger.info(f"✅ Infortuni trovati: {len(df)}")

    print("\n🩹 Infortuni (attivi o futuri):")
    for row in df.head(25).itertuples(index=False):
        print(f" - {row.player_name:<25} ({row.team_name}) [{row.status}] → {row.reason} ({row.type}, {row.date})")

    df.to_csv("injuries_preview.csv", index=False)
    print("\n💾 Salvato anche in: injuries_preview.csv")


if __name__ == "__main__":
    # Esegui: preview_injuries("Lazio") oppure preview_injuries() per tutte le squadre
    preview_injuries("Lazio")
