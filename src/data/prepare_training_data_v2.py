# src/data/prepare_training_data_v2.py
"""
Genera il dataset di training per il modello SoS-IA.
Integra dati da matches, predictions, standings e applica il feature engineering centralizzato.
"""

import os
import pandas as pd

from src.core.config import PATHS, FEATURE_CONFIG, compute_result
from src.core.db import fetch_df
from src.core.logger import get_logger
from src.core.utils import save_dataframe
from src.features.feature_engineering import compute_features

logger = get_logger(__name__)


def build_dataset(
    min_season: int = 2021,
    max_season: int = 2025,
    cutoff_round: int | None = None,
) -> pd.DataFrame:
    logger.info(
        "📊 Costruzione dataset: stagioni %s → %s, cutoff_round=%s",
        min_season,
        max_season,
        cutoff_round,
    )

    # -------------------------------------------------
    # 1) Prendo TUTTI i match FT tra min_season e max_season
    # -------------------------------------------------
    matches_query = f"""
        SELECT match_id, season, round, date,
               home_team_id, away_team_id,
               home_team_name, away_team_name,
               home_goals, away_goals, status
        FROM matches
        WHERE season BETWEEN {min_season} AND {max_season}
          AND status = 'FT'
    """
    matches = fetch_df(matches_query)

    # estraggo numero giornata
    matches["round_number"] = matches["round"].str.extract(r"(\d+)").astype(float)

    # progressione (serve solo come feature)
    max_round_overall = matches["round_number"].max()
    matches["round_progress"] = matches["round_number"] / max_round_overall

    # -------------------------------------------------
    # 2) cutoff SOLO sulla stagione max_season
    #    (quello che ti dava 400 righe era perché tagliava tutto)
    # -------------------------------------------------
    if cutoff_round is not None:
        before = len(matches)
        mask_other_seasons = matches["season"] < max_season
        mask_current_season = (matches["season"] == max_season) & (matches["round_number"] < cutoff_round)
        matches = matches[mask_other_seasons | mask_current_season]
        after = len(matches)
        logger.info(
            "✂️ Cutoff applicato solo alla stagione %s: %s → %s partite (fino al round %s)",
            max_season,
            before,
            after,
            cutoff_round,
        )

    # -------------------------------------------------
    # 3) altre tabelle
    # -------------------------------------------------
    preds = fetch_df(
        "SELECT match_id, prob_home, prob_draw, prob_away FROM predictions"
    )

    standings = fetch_df(
        """
        SELECT team_id, season, points, rank, form
        FROM standings
        """
    )

    # -------------------------------------------------
    # 3bis) Limitiamo standings al round precedente (evita data leakage)
    # -------------------------------------------------
    # Trova per ogni stagione l'ultima giornata effettivamente conclusa (status='FT')
    last_rounds = (
        matches[matches["status"] == "FT"]
        .groupby("season")["round_number"]
        .max()
        .reset_index()
        .rename(columns={"round_number": "last_round"})
    )

    # Sottrai 1 per avere il round precedente (ma non andare sotto 1)
    last_rounds["valid_round"] = last_rounds["last_round"].apply(lambda x: max(1, x - 1))

    # Aggiungi la colonna 'season' per il join
    standings = standings.merge(last_rounds[["season", "valid_round"]], on="season", how="left")

    # (opzionale: se standings ha round_number, filtra)
    if "round_number" in standings.columns:
        standings = standings[standings["round_number"] <= standings["valid_round"]]

    # -------------------------------------------------
    # 4) mergeone
    # -------------------------------------------------
    dataset = (
        matches
        .merge(preds, on="match_id", how="left")
        .merge(
            standings.add_prefix("home_"),
            left_on=["home_team_id", "season"],
            right_on=["home_team_id", "home_season"],
            how="left",
        )
        .merge(
            standings.add_prefix("away_"),
            left_on=["away_team_id", "season"],
            right_on=["away_team_id", "away_season"],
            how="left",
        )
    )

    # -------------------------------------------------
    # 5) label (H / D / A) — una sola definizione, presa dal core
    # -------------------------------------------------
    dataset["label_result"] = dataset.apply(compute_result, axis=1)

    # -------------------------------------------------
    # 6) feature engineering centralizzato
    # -------------------------------------------------
    dataset = compute_features(dataset, mode="train", cutoff_round=cutoff_round, season=max_season)

    # -------------------------------------------------
    # 7) selezione colonne utili (ma salviamo comunque tutto nel CSV)
    # -------------------------------------------------
    exclude_cols = FEATURE_CONFIG["exclude_cols"]
    feature_cols = [c for c in dataset.columns if c not in exclude_cols]
    logger.info("🧮 Feature utili per il training: %s", len(feature_cols))

    # -------------------------------------------------
    # 8) salvataggio
    # -------------------------------------------------
    out_dir = PATHS["processed"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"training_data_{max_season}.csv")

    save_dataframe(dataset, out_path)
    logger.info("💾 Dataset di training salvato in: %s", out_path)

    return dataset, out_path
