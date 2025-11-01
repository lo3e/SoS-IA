# src/prepare_training_data_v2.py
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
DB_PATH = os.getenv("DB_PATH")

if not DATA_PATH or not DB_PATH:
    raise EnvironmentError("❌ Variabili DATA_PATH o DB_PATH mancanti nel file .env")

DATA_PATH = Path(DATA_PATH)
DB_PATH = Path(DB_PATH)

OUTPUT_PATH = DATA_PATH / "training_data_v2.csv"


# === FUNZIONI ===
def load_matches(conn):
    """
    Carica le partite concluse (FT) fino alla stagione 2024,
    con le feature di classifica pre-partita prese da teams_standings.
    """
    query = """
        SELECT
            m.match_id,
            m.date,
            m.season,
            m.round AS round_number,
            m.league_id,
            m.status,
            m.home_team_id,
            m.away_team_id,
            m.home_team_name,
            m.away_team_name,
            m.home_goals,
            m.away_goals,

            -- TARGET SUPERVISIONE
            CASE
                WHEN m.home_goals > m.away_goals THEN 1
                WHEN m.home_goals = m.away_goals THEN 0
                ELSE -1
            END AS target_num,
            CASE
                WHEN m.home_goals > m.away_goals THEN '1'
                WHEN m.home_goals = m.away_goals THEN 'X'
                ELSE '2'
            END AS target_1x2,

            -- FEATURE PRE-MATCH: standings PRIMA della partita
            s_home.rank AS home_rank,
            s_away.rank AS away_rank,
            s_home.points AS home_points,
            s_away.points AS away_points,
            s_home.goals_for AS home_goals_for,
            s_home.goals_against AS home_goals_against,
            s_away.goals_for AS away_goals_for,
            s_away.goals_against AS away_goals_against

        FROM matches m
        LEFT JOIN standings s_home
            ON m.home_team_id = s_home.team_id
            AND m.league_id = s_home.league_id
            AND m.season = s_home.season
        LEFT JOIN standings s_away
            ON m.away_team_id = s_away.team_id
            AND m.league_id = s_away.league_id
            AND m.season = s_away.season
        WHERE
            m.status = 'FT'
            AND m.season > 2020 AND m.season <= 2024
    """
    return pd.read_sql_query(query, conn)

def add_injuries(conn):
    """Crea dataframe con numero e peso degli infortuni per squadra."""
    query = """
        SELECT
            i.team_name,
            COUNT(DISTINCT i.player_name) AS num_injuries,
            ROUND(SUM(COALESCE(p.performance_index, 0)), 2) AS injury_impact
        FROM current_unavailable_players i
        LEFT JOIN player_form_ranking p
            ON i.player_name = p.player_name
        GROUP BY i.team_name
    """
    return pd.read_sql_query(query, conn)

def compute_h2h_features(df):
    """
    Aggiunge statistiche head-to-head (ultimi 5 incontri) tra le due squadre.
    Evita leakage usando solo match precedenti alla data corrente.
    """
    print("⚔️  Calcolo feature Head-to-Head (ultimi 5 match tra le due squadre)...")

    df = df.sort_values("date")
    h2h_data = []

    for i, row in df.iterrows():
        home = row["home_team_id"]
        away = row["away_team_id"]
        date = row["date"]

        past = df[
            (
                ((df["home_team_id"] == home) & (df["away_team_id"] == away)) |
                ((df["home_team_id"] == away) & (df["away_team_id"] == home))
            )
            & (df["date"] < date)
        ].tail(5)

        if past.empty:
            h2h_data.append((0, 0, 0, 0, 0))
            continue

        # Statistiche dirette
        home_wins = ((past["home_team_id"] == home) & (past["target_num"] == 1)).sum() + \
                    ((past["away_team_id"] == home) & (past["target_num"] == -1)).sum()
        away_wins = ((past["home_team_id"] == away) & (past["target_num"] == 1)).sum() + \
                    ((past["away_team_id"] == away) & (past["target_num"] == -1)).sum()
        draws = (past["target_num"] == 0).sum()

        goal_diff_avg = (
            ((past["home_team_id"] == home) * (past["home_goals"] - past["away_goals"])) +
            ((past["away_team_id"] == home) * (past["away_goals"] - past["home_goals"]))
        ).mean()

        total = len(past)
        h2h_data.append((home_wins, away_wins, draws, goal_diff_avg, total))

    df[["h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_goal_diff_avg", "h2h_total_matches"]] = \
        pd.DataFrame(h2h_data, index=df.index)

    return df

def compute_rolling_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature di forma recente (ultime 5 partite) basate su punti, gol fatti/subiti.
    Usa uno shift(1) per evitare leakage (la partita corrente non influisce sulla previsione).
    """
    def rolling_features(df: pd.DataFrame, team_type: str) -> pd.DataFrame:
        team_col = f"{team_type}_team_id"
        goals_for_col = f"{team_type}_goals_for"
        goals_against_col = f"{team_type}_goals_against"

        # punti guadagnati in questa partita
        df[f"{team_type}_points_match"] = np.where(
            df["home_goals"] > df["away_goals"], 3,
            np.where(df["home_goals"] == df["away_goals"], 1, 0)
        )
        if team_type == "away":
            df[f"{team_type}_points_match"] = np.where(
                df["away_goals"] > df["home_goals"], 3,
                np.where(df["away_goals"] == df["home_goals"], 1, 0)
            )

        # rolling medio delle ultime 5 partite (spostato di 1 → pre-match)
        df[f"{team_type}_recent_points_avg"] = (
            df.groupby(team_col)[f"{team_type}_points_match"]
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{team_type}_recent_goals_for_avg"] = (
            df.groupby(team_col)["home_goals" if team_type == "home" else "away_goals"]
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{team_type}_recent_goals_against_avg"] = (
            df.groupby(team_col)["away_goals" if team_type == "home" else "home_goals"]
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        return df

    matches = matches.sort_values(["season", "date"])
    matches = rolling_features(matches, "home")
    matches = rolling_features(matches, "away")

    # Rimuovo i punti della partita (non servono nel modello)
    matches = matches.drop(columns=["home_points_match", "away_points_match"], errors="ignore")

    return matches

def main():
    print("🧠 Costruzione training_data_v2...")

    with sqlite3.connect(DB_PATH) as conn:
        matches = load_matches(conn)
        injuries = add_injuries(conn)

    # Merge infortuni
    matches = matches.merge(injuries, left_on="home_team_name", right_on="team_name", how="left")
    matches = matches.rename(columns={
        "num_injuries": "home_num_injuries",
        "injury_impact": "home_injury_impact"
    }).drop(columns=["team_name"])

    matches = matches.merge(injuries, left_on="away_team_name", right_on="team_name", how="left")
    matches = matches.rename(columns={
        "num_injuries": "away_num_injuries",
        "injury_impact": "away_injury_impact"
    }).drop(columns=["team_name"])

    # Calcolo differenze gol e punti per match
    matches["home_goals_diff"] = matches["home_goals"] - matches["away_goals"]
    matches["away_goals_diff"] = -matches["home_goals_diff"]

    matches["home_points_match"] = matches.apply(
        lambda x: 3 if x["home_goals"] > x["away_goals"] else (1 if x["home_goals"] == x["away_goals"] else 0), axis=1
    )
    matches["away_points_match"] = matches.apply(
        lambda x: 3 if x["away_goals"] > x["home_goals"] else (1 if x["away_goals"] == x["home_goals"] else 0), axis=1
    )

    # Rolling stats
    matches = compute_rolling_stats(matches)

    # Head-to-Head features
    matches = compute_h2h_features(matches)

    # --- 6️⃣ FEATURE DI EQUILIBRIO MATCH (per migliorare la predizione dei pareggi) ---

    print("⚖️  Calcolo feature di equilibrio...")

    # Differenza tra rank
    if "home_rank" in matches.columns and "away_rank" in matches.columns:
        matches["rank_diff"] = (matches["home_rank"] - matches["away_rank"]).abs()
    else:
        matches["rank_diff"] = 0

    # Differenza tra punti in classifica
    if "home_points" in matches.columns and "away_points" in matches.columns:
        matches["points_diff"] = (matches["home_points"] - matches["away_points"]).abs()
    else:
        matches["points_diff"] = 0

    # Differenza media punti recenti (forma)
    if "home_recent_points_avg" in matches.columns and "away_recent_points_avg" in matches.columns:
        matches["form_diff"] = (matches["home_recent_points_avg"] - matches["away_recent_points_avg"]).abs()
    else:
        matches["form_diff"] = 0

    # Bonus: indice sintetico di equilibrio (0=molto sbilanciato, 1=molto equilibrato)
    max_rank_diff = matches["rank_diff"].max() if matches["rank_diff"].max() != 0 else 1
    matches["match_balance_index"] = 1 - (matches["rank_diff"] / max_rank_diff)

    print("✅ Feature di equilibrio aggiunte: rank_diff, points_diff, form_diff, match_balance_index")


    matches = matches.fillna(0)

    matches.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Dataset v2 salvato in: {OUTPUT_PATH}")
    print(f"📊 Totale righe: {len(matches)} | Totale colonne: {len(matches.columns)}")


if __name__ == "__main__":
    main()
