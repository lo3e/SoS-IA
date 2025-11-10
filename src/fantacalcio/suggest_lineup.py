import os
import sqlite3
import pandas as pd
from datetime import datetime
from src.core.config import PATHS, DB_PATH
from src.core.logger import get_logger  # usa il tuo logger di progetto

logger = get_logger(__name__)


# --------------------------------------------------
# util
# --------------------------------------------------
def normalize_name(name):
    return str(name).split()[0][0] + ". " + str(name).split()[-1] if " " in str(name) else name


def round_to_int(r):
    if r is None:
        return None
    s = str(r)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def suggest_best_lineup():
    # --------------------------------------------------
    # 1) carica rosa utente
    # --------------------------------------------------
    teams_dir = PATHS["teams"]
    files = sorted(
        [f for f in os.listdir(teams_dir) if f.endswith(".csv")],
        reverse=True,
    )
    if not files:
        raise RuntimeError("❌ Nessuna rosa trovata in data/fantacalcio/squadre")
    latest = files[0]
    team_path = os.path.join(teams_dir, latest)
    df_team = pd.read_csv(team_path)
    logger.info(f"📂 Squadra caricata: {team_path}")

    # allinea nomi colonne del csv
    if "position" in df_team.columns and "position_user" not in df_team.columns:
        df_team = df_team.rename(columns={"position": "position_user"})
    if "team_name" in df_team.columns and "team_name_user" not in df_team.columns:
        df_team = df_team.rename(columns={"team_name": "team_name_user"})

    df_team["player_name_norm"] = df_team["player_name"].apply(normalize_name)

    # --------------------------------------------------
    # 2) UNA sola connessione DB
    # --------------------------------------------------
    conn = sqlite3.connect(DB_PATH)

    # --------------------------------------------------
    # 3) trova prossima giornata da matches
    # --------------------------------------------------
    df_matches = pd.read_sql("SELECT * FROM matches WHERE status = 'NS'", conn)
    if df_matches.empty:
        conn.close()
        raise RuntimeError("⚠️ Nessuna partita NS trovata in matches.")

    round_col = None
    for col in df_matches.columns:
        if col.lower() in ["round", "round_name", "round_label"]:
            round_col = col
            break

    if round_col is None:
        conn.close()
        raise RuntimeError(f"❌ Nessuna colonna round in matches. Colonne: {list(df_matches.columns)}")

    df_matches["round_int"] = df_matches[round_col].apply(round_to_int)
    next_round = int(df_matches["round_int"].min())
    logger.info(f"📅 Prossima giornata identificata: {next_round}")

    # --------------------------------------------------
    # 4) indisponibili SOLO per quel turno
    # (NON chiudiamo il DB qui)
    # --------------------------------------------------
    try:
        df_unavail = pd.read_sql(
            f"""
            SELECT DISTINCT player_name, team_name, category, reason
            FROM current_unavailable_players
            WHERE match_id IN (
                SELECT match_id FROM matches
                WHERE status = 'NS'
                  AND CAST(REPLACE({round_col}, 'Regular Season - ', '') AS INTEGER) = {next_round}
            )
            """,
            conn,
        )
    except Exception:
        # fallback: tutta la view
        df_unavail = pd.read_sql(
            "SELECT DISTINCT player_name, team_name, category, reason FROM current_unavailable_players",
            conn,
        )

    if df_unavail.empty:
        logger.info("✅ Nessun indisponibile trovato per la prossima giornata (nella view).")
    else:
        logger.info(f"🩹 Giocatori indisponibili trovati: {len(df_unavail)}")

    df_unavail["player_name_norm"] = df_unavail["player_name"].apply(normalize_name)

    # --------------------------------------------------
    # 5) carica forma giocatori
    # (sempre stessa conn)
    # --------------------------------------------------
    df_form = pd.read_sql("SELECT * FROM player_form_ranking WHERE season = 2025", conn)
    df_form["player_name_norm"] = df_form["player_name"].apply(normalize_name)

    # --------------------------------------------------
    # 6) merge rosa utente + forma
    # --------------------------------------------------
    df_merged = pd.merge(
        df_team,
        df_form[[
            "player_id",
            "player_name",
            "team_name",
            "position",
            "avg_rating",
            "goals",
            "assists",
            "performance_index",
            "player_name_norm",
        ]],
        on="player_name_norm",
        how="left",
        suffixes=("_team", "_form"),
    )

    df_merged["player_name"] = df_merged["player_name_team"].fillna(df_merged["player_name_form"])
    df_merged["team_name"] = (
        df_merged["team_name_user"].fillna(df_merged["team_name_form"])
        if "team_name_form" in df_merged.columns
        else df_merged["team_name_user"]
    )
    df_merged["effective_role"] = df_merged["position_user"].fillna(df_merged["position"])

    # === minutaggio ===
    logger.info("⏱️ Recupero dati di minutaggio e continuità dai player_stats_clean_view...")
    df_minutes = pd.read_sql_query(
        """
        SELECT player_id, player_name, SUM(minutes) AS total_minutes
        FROM player_stats_clean_view
        WHERE match_id IN (
            SELECT match_id FROM matches WHERE season = 2025
        )
        GROUP BY player_id, player_name
        """,
        conn,
    )
    logger.info(f"✅ Record minutaggio trovati: {len(df_minutes)}")

    df_minutes["player_name_norm"] = df_minutes["player_name"].str.lower().str.strip()

    # === Merge con minutaggio ===
    if "player_id_team" in df_merged.columns:
        df_merged = df_merged.merge(
            df_minutes[["player_id", "total_minutes"]],
            left_on="player_id_team",
            right_on="player_id",
            how="left"
        )
    else:
        # fallback su nome normalizzato
        df_minutes["player_name_norm"] = df_minutes["player_name"].str.lower().str.strip()
        df_merged = df_merged.merge(
            df_minutes[["player_name_norm", "total_minutes"]],
            on="player_name_norm",
            how="left"
        )

    df_merged["total_minutes"] = df_merged["total_minutes"].fillna(0)
    df_merged["minutes_factor"] = df_merged["total_minutes"].apply(lambda x: min(1.0, x / 900))

    logger.info("⚖️ Esempio fattori di minutaggio aggiornati (merge by player_id se possibile):")
    print(df_merged[["player_name", "player_id_team", "total_minutes", "minutes_factor"]].head(10))

    # === FORMA RECENTE (ultimi 5 match) ===
    logger.info("⏱️ Calcolo forma recente (ultimi 5 match)...")
    try:
        df_recent = pd.read_sql_query(
            """
            WITH recent_matches AS (
                SELECT DISTINCT match_id
                FROM player_stats_clean_view
                ORDER BY match_id DESC
                LIMIT 5
            )
            SELECT 
                ps.player_id,
                ps.player_name,
                AVG(ps.rating) AS avg_rating_recent,
                SUM(ps.goals_total) AS goals_recent,
                SUM(ps.assists) AS assists_recent
            FROM player_stats_clean_view ps
            INNER JOIN recent_matches rm ON ps.match_id = rm.match_id
            GROUP BY ps.player_id, ps.player_name
            """,
            conn,
        )
        logger.info(f"✅ Record forma recente trovati: {len(df_recent)}")
        df_merged = df_merged.merge(df_recent, on=["player_id", "player_name"], how="left")
    except Exception as e:
        logger.warning(f"⚠️ Errore durante il calcolo forma recente: {e}")

        # === DIFFICOLTÀ PROSSIMO MATCH ===
    logger.info("🎯 Calcolo difficoltà prossima partita per ciascun giocatore...")

    try:
        # prendiamo SOLO le partite del prossimo turno
        df_difficulty = pd.read_sql_query(f"""
            SELECT 
                m.{round_col} AS round,
                m.home_team_name,
                m.away_team_name,
                t1.rank AS home_rank,
                t2.rank AS away_rank
            FROM matches m
            LEFT JOIN standings t1 ON m.home_team_name = t1.team_name
            LEFT JOIN standings t2 ON m.away_team_name = t2.team_name
            WHERE m.season = 2025
              AND m.status = 'NS'
              AND CAST(REPLACE(m.{round_col}, 'Regular Season - ', '') AS INTEGER) = {next_round}
        """, conn)

        def difficulty_value(rank):
            if pd.isna(rank):
                return 1.0
            if rank <= 5:
                return 1.2
            elif rank <= 10:
                return 1.1
            elif rank <= 15:
                return 1.0
            else:
                return 0.9

        # calcolo difficoltà per casa/trasferta
        df_difficulty["home_diff"] = df_difficulty["away_rank"].apply(difficulty_value)
        df_difficulty["away_diff"] = df_difficulty["home_rank"].apply(difficulty_value)

        # "stendiamo" in 2 righe (una per squadra) ma SOLO per quel turno
        df_difficulty_melt = pd.melt(
            df_difficulty,
            id_vars=["round"],
            value_vars=["home_team_name", "away_team_name"],
            var_name="venue",
            value_name="team_name"
        )

        # assegniamo la difficoltà giusta
        def pick_diff(row):
            base = df_difficulty[
                (df_difficulty["round"] == row["round"]) &
                (
                    (df_difficulty["home_team_name"] == row["team_name"]) |
                    (df_difficulty["away_team_name"] == row["team_name"])
                )
            ]
            if row["venue"] == "home_team_name":
                return float(base["home_diff"].iloc[0])
            else:
                return float(base["away_diff"].iloc[0])

        df_difficulty_melt["match_difficulty"] = df_difficulty_melt.apply(pick_diff, axis=1)

        # merge (ora non esplode più)
        df_merged = df_merged.merge(
            df_difficulty_melt[["team_name", "match_difficulty"]],
            on="team_name",
            how="left"
        )
        df_merged["match_difficulty"] = df_merged["match_difficulty"].fillna(1.0)
        logger.info("✅ Match difficulty calcolata e aggiornata (solo prossimo turno).")

    except Exception as e:
        logger.warning(f"⚠️ Errore durante il calcolo match difficulty: {e}")
        df_merged["match_difficulty"] = 1.0

    # --------------------------------------------------
    # 7) marca indisponibili della tua rosa
    # --------------------------------------------------
    unavail_set = set(df_unavail["player_name_norm"].tolist())

    def is_unavailable(row):
        pn = row["player_name_norm"]
        if pn in unavail_set:
            return True
        for bad in unavail_set:
            if pn and pn in bad:
                return True
            if bad and bad in pn:
                return True
        return False

    df_merged["is_unavailable"] = df_merged.apply(is_unavailable, axis=1)

    if df_merged["is_unavailable"].any():
        logger.info("🚑 Giocatori indisponibili nella tua rosa:")
        for n in sorted(set(df_merged.loc[df_merged["is_unavailable"], "player_name"])):  # <- set()
            logger.info(f"   - {n}")


    before = len(df_merged)
    df_merged = df_merged[df_merged["is_unavailable"] == False].copy()
    removed = before - len(df_merged)
    if removed > 0:
        logger.info(f"🚫 Rimossi {removed} giocatori indisponibili dal calcolo formazione.")

    # deduplica
    before = len(df_merged)
    df_merged = (
        df_merged.sort_values(by=["player_id_team", "avg_rating_team", "avg_rating_form"], ascending=[True, False, False])
        .drop_duplicates(subset=["player_id_team"], keep="first")
    )
    after = len(df_merged)
    logger.info(f"🧹 Rimossi {before - after} duplicati dal dataframe dei giocatori.")

    # === 9) Calcolo punteggio ===
    print("\n🧩 Colonne disponibili nel df_merged:")
    print(df_merged.columns.tolist())

    print("\n🔎 Esempio dati (prime 5 righe):")
    cols_debug = [
        "player_name",
        "effective_role",
        "avg_rating_team",
        "avg_rating_form",
        "performance_index_team",
        "performance_index_form",
        "goals_team",
        "goals_form",
        "assists_team",
        "assists_form",
    ]
    existing_cols = [c for c in cols_debug if c in df_merged.columns]
    print(df_merged[existing_cols].head(10))

    def compute_score(row):
        w_rating_team = 0.4
        w_rating_recent = 0.6
        w_goal = 0.6
        w_assist = 0.3

        rating = (row.get("avg_rating_team", 0) * w_rating_team) + (
            row.get("avg_rating_recent", row.get("avg_rating_form", 0)) * w_rating_recent
        )

        goals = row.get("goals_recent", row.get("goals_team", 0))
        assists = row.get("assists_recent", row.get("assists_team", 0))
        bonus = (goals * w_goal) + (assists * w_assist)

        minute_factor = row.get("minutes_factor", 1.0)
        difficulty = row.get("match_difficulty", 1.0)

        if row.get("effective_role") == "F":
            bonus *= 1.2
        elif row.get("effective_role") == "M":
            bonus *= 1.1
        elif row.get("effective_role") == "G":
            difficulty = max(0.8, 2.0 - difficulty)

        score = (rating + bonus) * minute_factor / difficulty
        return round(score, 3)

    df_merged["score"] = df_merged.apply(compute_score, axis=1)

    cols_debug = [
        "player_name",
        "effective_role",
        "avg_rating_team",
        "avg_rating_form",
        "goals_team",
        "assists_team",
        "match_difficulty",
        "minutes_factor",
        "score",
    ]
    if set(cols_debug).issubset(df_merged.columns):
        logger.info("📊 Tabella punteggi dettagliati (prime 15 righe):")
        print(df_merged[cols_debug].sort_values(by="score", ascending=False).head(15))

        debug_csv_path = os.path.join(
            PATHS["suggestions"], f"suggested_lineup_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_merged[cols_debug].sort_values(by="score", ascending=False).to_csv(debug_csv_path, index=False)
        logger.info(f"💾 CSV dei punteggi salvato in: {debug_csv_path}")
    else:
        missing = set(cols_debug) - set(df_merged.columns)
        logger.warning(f"⚠️ Colonne mancanti per debug: {missing}")

    # === 10) selezione formazione
    df_sorted = df_merged.sort_values("score", ascending=False)
    best_players = []

    role_limits = {"G": 1, "D": 3, "M": 4, "F": 3}
    for role, limit in role_limits.items():
        subset = df_sorted[df_sorted["effective_role"].str.startswith(role)].head(limit)
        best_players.append(subset)

    best_players = pd.concat(best_players)

    role_order = {"G": 0, "D": 1, "M": 2, "F": 3}
    best_players = best_players.sort_values(by="effective_role", key=lambda x: x.map(role_order))

    total_score = round(best_players["score"].sum(), 2)
    logger.info(f"🏆 Miglior formazione trovata: 3-4-3 (score totale: {total_score})\n")

    for _, row in best_players.iterrows():
        logger.info(f" - {row['player_name']} ({row['team_name']}, {row['effective_role']}) → {row['score']}")

    # 🔚 QUI chiudiamo davvero
    conn.close()


if __name__ == "__main__":
    suggest_best_lineup()
