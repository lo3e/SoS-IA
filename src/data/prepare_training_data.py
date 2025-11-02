# src/prepare_training_data.py
import os
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH")  # dove salvare il csv

if not DB_PATH:
    raise ValueError("DB_PATH non definito nel .env")


def _clean_prob_series(s: pd.Series) -> pd.Series:
    """
    Converte valori tipo '50%' o '72' o None in float 0.50, 0.72, 0.0
    """
    # tutto in stringa
    s = s.astype(str)
    # togli % se c'è
    s = s.str.replace("%", "", regex=False)
    # vuoti o 'None' -> 0
    s = s.replace({"None": 0, "nan": 0, "": 0})
    # in numero
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    # da 0-100 a 0-1
    s = s / 100.0
    return s


def prepare_training_data(conn):
    # 1) partite giocate 2021-2024
    matches = pd.read_sql_query(
        """
        SELECT
            m.match_id,
            m.date,
            m.season,
            m.league_id,
            m.home_team_id,
            m.away_team_id,
            m.home_team_name,
            m.away_team_name,
            m.home_goals,
            m.away_goals,
            m.status
        FROM matches m
        WHERE m.status = 'FT'
          AND m.season BETWEEN 2021 AND 2024
        ORDER BY m.date
        """,
        conn,
    )

    # 2) standings (la joiniamo 2 volte)
    standings = pd.read_sql_query(
        """
        SELECT
            season,
            team_id,
            rank,
            points,
            goals_diff,
            goals_for,
            goals_against
        FROM standings
        """,
        conn,
    )

    # home
    matches = matches.merge(
        standings.add_prefix("home_"),
        left_on=["season", "home_team_id"],
        right_on=["home_season", "home_team_id"],
        how="left",
    )
    # away
    matches = matches.merge(
        standings.add_prefix("away_"),
        left_on=["season", "away_team_id"],
        right_on=["away_season", "away_team_id"],
        how="left",
    )

    # colonne di servizio
    for col in ("home_season", "away_season"):
        if col in matches.columns:
            matches = matches.drop(columns=[col])

    # 3) injuries “condensati” (quanti assenti e quanto pesano)
    injuries = pd.read_sql_query(
        """
        SELECT
            i.team_name AS team_name,
            COUNT(DISTINCT i.player_name) AS num_injuries,
            ROUND(SUM(COALESCE(p.performance_index, 0)), 2) AS injury_impact
        FROM current_unavailable_players i
        LEFT JOIN player_form_ranking p
            ON i.player_name = p.player_name
        GROUP BY i.team_name
        """,
        conn,
    )
    injuries_home = injuries.add_prefix("home_")
    injuries_away = injuries.add_prefix("away_")

    matches = matches.merge(
        injuries_home,
        left_on="home_team_name",
        right_on="home_team_name",
        how="left",
    )
    matches = matches.merge(
        injuries_away,
        left_on="away_team_name",
        right_on="away_team_name",
        how="left",
    )

    # 4) predictions API
    preds = pd.read_sql_query(
        """
        SELECT
            match_id,
            prob_home,
            prob_draw,
            prob_away
        FROM predictions
        """,
        conn,
    )

    df = matches.merge(preds, on="match_id", how="left")

    # 4bis) convertiamo le 3 prob in numerico vero (0-1)
    for col in ("prob_home", "prob_draw", "prob_away"):
        if col in df.columns:
            df[col] = _clean_prob_series(df[col])
        else:
            df[col] = 0.0

    # 5) target 1X2
    def outcome(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"

    df["target_1x2"] = df.apply(outcome, axis=1)

    # 6) togliamo eventuali duplicati di match (dopo tutte le join può capitare)
    df = df.drop_duplicates(subset=["match_id"], keep="first")

    # 7) riempiamo i NaN numerici con 0 (standings di inizio stagione ecc.)
    numeric_like = [
        "prob_home", "prob_draw", "prob_away",
        "home_rank", "away_rank",
        "home_points", "away_points",
        "home_goals_diff", "away_goals_diff",
        "home_goals_for", "home_goals_against",
        "away_goals_for", "away_goals_against",
        "home_num_injuries", "away_num_injuries",
        "home_injury_impact", "away_injury_impact",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 8) aggiungiamo una colonna di comodo: outcome numerico
    df["target_num"] = df["target_1x2"].map({"H": 1, "D": 0, "A": -1})

    return df


def main():
    conn = sqlite3.connect(DB_PATH)
    df = prepare_training_data(conn)
    conn.close()

    if not DATA_PATH:
        out_path = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"
    else:
        out_path = Path(DATA_PATH) / "training_data.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Dataset di training salvato in: {out_path}")
    print(f"📦 Righe: {len(df)} | Colonne: {len(df.columns)}")


if __name__ == "__main__":
    main()
