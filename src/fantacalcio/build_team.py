import pandas as pd
import sqlite3
import os
import unicodedata
from difflib import get_close_matches
from datetime import datetime
from src.core.config import PATHS, DB_PATH


def normalize_name(name: str) -> str:
    """Normalizza nomi rimuovendo accenti, simboli, caratteri speciali e varianti Unicode."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()

    # sostituzioni specifiche per lettere “strane”
    substitutions = {
        "ı": "i",
        "İ": "i",
        "ł": "l",
        "ø": "o",
        "ð": "d",
        "þ": "th",
        "œ": "oe",
        "æ": "ae",
    }
    for k, v in substitutions.items():
        name = name.replace(k, v)

    # normalizzazione Unicode
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # rimuovo caratteri speciali, spazi e apostrofi
    name = (
        name.replace("’", "")
        .replace("‘", "")
        .replace("'", "")
        .replace("`", "")
        .replace("-", "")
        .replace(" ", "")
    )

    # mantengo solo lettere
    name = "".join(c for c in name if c.isalpha())
    return name


def build_user_team():
    print("⚽ Inserisci la tua rosa per il Fantacalcio (solo cognomi o cognomi + iniziali).")

    # === Nome squadra utente ===
    user_team_name = input("\n🏟️  Inserisci il nome della tua squadra: ").strip()
    user_team_name = user_team_name.title().replace(" ", "")
    print(f"✅ Nome squadra registrato: {user_team_name}\n")

    # === Ruoli ===
    roles = {"G": 3, "D": 8, "M": 8, "F": 6}
    team = {}

    for role, n in roles.items():
        players = input(f"Inserisci i {n} {role} separati da virgola: ").strip()
        team[role] = [p.strip().title() for p in players.split(",") if p.strip()]

    # === Carica dati dal DB ===
    conn = sqlite3.connect(DB_PATH)
    df_players = pd.read_sql("SELECT * FROM player_form_ranking WHERE season = 2025", conn)
    conn.close()
    df_players["normalized_name"] = df_players["player_name"].apply(normalize_name)

    matched_players = []
    for role, names in team.items():
        for name in names:
            normalized = normalize_name(name)
            candidates = df_players[df_players["normalized_name"].str.contains(normalized, na=False)]

            if len(candidates) == 1:
                player = candidates.iloc[0]
                if player["position"] != role:
                    confirm = input(
                        f"⚠️ '{player.player_name}' risulta {player.position}, non {role}. "
                        f"Vuoi comunque aggiungerlo come {role}? (s/n): "
                    ).strip().lower()
                    if confirm != "s":
                        print("⏭️  Saltato.")
                        continue
                    # ✅ Forza il ruolo scelto dall’utente
                    player["position"] = role
                matched_players.append(player)

            elif len(candidates) > 1:
                print(f"⚠️ Più giocatori trovati per '{name}', seleziona uno:")
                for i, row in enumerate(candidates.itertuples(), 1):
                    print(f"{i}. {row.player_name} ({row.team_name}, ruolo={row.position})")
                choice = int(input("Numero scelta: "))
                player = candidates.iloc[choice - 1]
                # forza comunque il ruolo scelto dall’utente
                player["position"] = role
                matched_players.append(player)

            else:
                # 🔍 Fuzzy match per caratteri strani (es. Yildiz → Yıldız)
                all_names = df_players["normalized_name"].tolist()
                close = get_close_matches(normalized, all_names, n=1, cutoff=0.55)
                if close:
                    match = df_players[df_players["normalized_name"] == close[0]].iloc[0]
                    print(f"🤔 Nessun match perfetto per '{name}', uso '{match.player_name}' ({match.team_name}).")
                    if match["position"] != role:
                        confirm = input(
                            f"⚠️ '{match.player_name}' risulta {match.position}, non {role}. "
                            f"Vuoi comunque aggiungerlo come {role}? (s/n): "
                        ).strip().lower()
                        if confirm != "s":
                            print("⏭️  Saltato.")
                            continue
                        # ✅ Forza anche qui il ruolo dell’utente
                        match["position"] = role
                    else:
                        match["position"] = role
                    matched_players.append(match)
                else:
                    print(f"❌ Nessun giocatore trovato per '{name}'. Verifica ortografia o caratteri speciali.")

    # === Salva rosa ===
    if not matched_players:
        print("\n❌ Nessun giocatore trovato, rosa vuota.")
        return None

    df_team = pd.DataFrame(matched_players)
    df_team["user_team_name"] = user_team_name
    df_team["input_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_dir = PATHS["teams"]
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{user_team_name}_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(save_dir, filename)
    df_team.to_csv(path, index=False)

    print(f"\n✅ Rosa salvata in: {path}")
    print(f"📊 Totale giocatori: {len(df_team)} (squadra: {user_team_name})")
    return df_team


if __name__ == "__main__":
    df_team = build_user_team()
