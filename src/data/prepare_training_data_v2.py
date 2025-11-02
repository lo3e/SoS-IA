# src/data/prepare_training_data_v2.py
"""
Genera il dataset di training per il modello SoS-IA.
Integra dati da matches, odds, standings, injuries e altre feature.
Compatibile con il core refattorizzato (config, logger, db, utils).
"""

import pandas as pd
import numpy as np
from src.core.config import PATHS
from src.core.db import fetch_df
from src.core.logger import get_logger
from src.core.utils import save_dataframe

logger = get_logger(__name__)

def encode_form(form_str):
        """
        Converte una stringa di forma come 'WWDLW' in un indice numerico tra 0 e 1.
        - W = 3 punti
        - D = 1 punto
        - L = 0 punti
        """
        if not isinstance(form_str, str) or len(form_str.strip()) == 0:
            return np.nan
        points = {"W": 3, "D": 1, "L": 0}
        total = sum(points.get(c, 0) for c in form_str)
        return round(total / (len(form_str) * 3), 3)

def build_dataset(min_season: int = 2021, max_season: int = 2025, cutoff_round: int | None = None) -> pd.DataFrame:
    """
    Genera e salva il dataset di training.
    
    Args:
        min_season (int): prima stagione inclusa
        max_season (int): ultima stagione inclusa
        cutoff_round (int | None): opzionale, se specificato include solo partite fino a quel round della stagione max_season
    
    Returns:
        pd.DataFrame: dataset completo pronto per il training
    """
    logger.info(f"📊 Costruzione dataset: stagioni {min_season} → {max_season}, cutoff_round={cutoff_round}")

    # -------------------------------
    # 1️⃣ Lettura tabelle principali
    # -------------------------------
    matches_query = f"""
        SELECT match_id, season, round, date, home_team_id, away_team_id,
               home_team_name, away_team_name, home_goals, away_goals, status
        FROM matches
        WHERE season BETWEEN {min_season} AND {max_season}
          AND status = 'FT'
    """
    matches = fetch_df(matches_query)

    # Estrai round come numero intero (es. "Regular Season - 17" → 17)
    matches["round_number"] = matches["round"].str.extract(r"(\d+)").astype(float)

    # Opzionale: normalizza rispetto al numero massimo di giornate note
    max_round = matches["round_number"].max()
    matches["round_progress"] = matches["round_number"] / max_round

    if cutoff_round:
        matches = matches[
            ~((matches["season"] == max_season) & (matches["round"].apply(_round_to_int) > cutoff_round))
        ]
        logger.info(f"🔪 Tagliate partite oltre il round {cutoff_round} per la stagione {max_season}")

    logger.info(f"✅ Matches caricati: {len(matches)} righe")

    # -------------------------------
    # 2️⃣ Predictions API (probabilità base)
    # -------------------------------
    preds_query = """
        SELECT match_id, prob_home, prob_draw, prob_away
        FROM predictions
    """
    preds = fetch_df(preds_query)
    logger.info(f"📊 Predictions API caricate: {len(preds)} righe")

    # -------------------------------
    # 3️⃣ Standings (rank, forma, punti)
    # -------------------------------
    standings_query = """
        SELECT team_id, season, rank, points, goals_diff, form
        FROM standings
    """
    standings = fetch_df(standings_query)
    standings.rename(columns=lambda x: f"home_{x}" if x != "season" and x != "team_id" else x, inplace=True)
    away_standings = standings.rename(
        columns={c: c.replace("home_", "away_") for c in standings.columns if "home_" in c}
    )

    # join doppia per home e away
    matches = (
        matches.merge(standings, left_on=["home_team_id", "season"], right_on=["team_id", "season"], how="left")
        .merge(away_standings, left_on=["away_team_id", "season"], right_on=["team_id", "season"], how="left")
    )

    # Calcola l'indice forma per home e away
    # Calcola forma solo se le colonne esistono nel dataset
    if "form_home" in matches.columns:
        matches["home_form_index"] = matches["form_home"].apply(encode_form)
    else:
        matches["home_form_index"] = np.nan

    if "form_away" in matches.columns:
        matches["away_form_index"] = matches["form_away"].apply(encode_form)
    else:
        matches["away_form_index"] = np.nan


    logger.info("📈 Join standings completato")

    # -------------------------------
    # 4️⃣ Injuries
    # -------------------------------
    injuries_query = """
        SELECT match_id,
               SUM(CASE WHEN category='injury' THEN 1 ELSE 0 END) AS total_injuries,
               SUM(CASE WHEN category='suspension' THEN 1 ELSE 0 END) AS total_suspensions
        FROM injuries
        GROUP BY match_id
    """
    injuries = fetch_df(injuries_query)
    matches = matches.merge(injuries, on="match_id", how="left")
    logger.info("🩼 Join injuries completato")

    # -------------------------------
    # 5️⃣ Predictions + labels
    # -------------------------------
    dataset = matches.merge(preds, on="match_id", how="left")

    # -------------------------------
    # 6️⃣ Feature derivate e di equilibrio
    # -------------------------------

    # Conversione percentuali "60%" → 0.60
    for col in ["prob_home", "prob_away", "prob_draw"]:
        if col in dataset.columns:
            dataset[col] = (
                dataset[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float)
                / 100
            )

    # Conversione sicura delle altre colonne numeriche
    numeric_cols = [
        "home_points", "away_points",
        "home_rank", "away_rank",
        "home_form_index", "away_form_index"
    ]
    for col in numeric_cols:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    # Differenze dirette
    dataset["points_diff"] = dataset["home_points"] - dataset["away_points"]
    dataset["rank_diff"] = dataset["home_rank"] - dataset["away_rank"]
    dataset["form_diff"] = dataset["home_form_index"] - dataset["away_form_index"]

    # Equilibrio e tendenza al pareggio
    dataset["expected_draw_tendency"] = 1 - abs(dataset["prob_home"] - dataset["prob_away"])
    dataset["rank_balance"] = 1 / (1 + abs(dataset["rank_diff"]))
    dataset["points_balance"] = 1 / (1 + abs(dataset["points_diff"]))
    dataset["form_balance"] = 1 / (1 + abs(dataset["form_diff"]))

    logger.info("⚖️  Feature derivate e di equilibrio calcolate correttamente.")

    # Etichetta target (1=Home Win, 0=Draw, -1=Away Win)
    dataset["label_result"] = dataset.apply(_label_result, axis=1)

    # -------------------------------
    # 6️⃣ Pulizia finale e salvataggio
    # -------------------------------
    dataset.fillna(0, inplace=True)

    out_name = f"training_data_{min_season}_{max_season}.csv"
    save_path = save_dataframe(dataset, out_name)
    logger.info(f"💾 Dataset salvato: {save_path}")

    return dataset, save_path


# ----------------------------------------------------
# Helper interni
# ----------------------------------------------------
def _label_result(row):
    if row["home_goals"] > row["away_goals"]:
        return 1
    elif row["home_goals"] < row["away_goals"]:
        return -1
    return 0


def _round_to_int(round_str: str | None) -> int:
    if not round_str:
        return -1
    try:
        return int(round_str.split("-")[-1].strip())
    except ValueError:
        return -1


# ----------------------------------------------------
# Entry point CLI
# ----------------------------------------------------
def main():
    """Esegui direttamente lo script."""
    df = build_dataset(min_season=2021, max_season=2025)
    logger.info(f"Dataset finale: {df.shape[0]} righe, {df.shape[1]} colonne")


if __name__ == "__main__":
    main()
