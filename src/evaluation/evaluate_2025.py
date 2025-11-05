# src/evaluation/evaluate_2025.py
"""
Valutazione del modello sulle partite completate di una specifica stagione e round.
Usa le stesse logiche di feature engineering di prepare_training_data_v2.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.core.config import PATHS, FEATURE_CONFIG, compute_result
from src.core.db import fetch_df
from src.core.logger import get_logger
from src.features.feature_engineering import compute_features
from src.core.utils import form_to_index, log_model_run
from src.features.advanced_stats import build_advanced_stats

logger = get_logger(__name__)


def evaluate_on_round(round_number: int, season: int, model_path: str | None = None):
    logger.info(
        "📊 Avvio valutazione per stagione %s, round %s...", season, round_number
    )

    # -------------------------------------------------
    # 1️⃣ Carico partite concluse fino al round indicato
    # -------------------------------------------------
    matches_query = f"""
        SELECT m.match_id, m.date, m.round, m.season,
               m.home_team_id, m.away_team_id,
               m.home_team_name, m.away_team_name,
               m.home_goals, m.away_goals, m.status
        FROM matches m
        WHERE m.season = {season}
          AND m.status = 'FT'
          AND m.round = 'Regular Season - {round_number}'
    """
    df = fetch_df(matches_query)

    # 🧩 Calcolo del round_progress basato sul round passato dal daily
    try:
        max_round_db = fetch_df(
            f"SELECT MAX(CAST(SUBSTR(round, INSTR(round, '-') + 1) AS INTEGER)) AS max_r "
            f"FROM matches WHERE season = {season}"
        )["max_r"].iloc[0]
        if not max_round_db or max_round_db == 0:
            max_round_db = round_number  # fallback se DB non restituisce nulla
    except Exception:
        max_round_db = round_number  # fallback totale in caso di errore query

    df["round_progress"] = round_number / max_round_db
    logger.info(f"🧮 round_progress calcolato come {round_number}/{max_round_db} = {round_number / max_round_db:.2f}")


    if df.empty:
        logger.warning("⚠️ Nessuna partita trovata per la stagione %s, round %s", season, round_number)
        return None, None

    logger.info("✅ Partite da valutare: %s", len(df))

    # -------------------------------------------------
    # 2️⃣ Aggiungo le info da predictions e standings
    # -------------------------------------------------
    preds = fetch_df("SELECT match_id, prob_home, prob_draw, prob_away FROM predictions")
    #standings = fetch_df("SELECT team_id, season, points, rank, form FROM standings")

    # ----------------------------------------------
    # 🧱 Standings limitate fino al round precedente
    # ----------------------------------------------
    standings = fetch_df("SELECT team_id, season, points, rank, form FROM standings")

    # Calcoliamo round precedente
    valid_round_limit = max(1, round_number - 1)

    # Determina qual è l’ultima giornata conclusa per la stagione
    last_round_df = fetch_df(
        f"""
        SELECT MAX(CAST(SUBSTR(round, INSTR(round, '-') + 1) AS INTEGER)) AS max_r
        FROM matches
        WHERE season = {season} AND status = 'FT'
        """
    )
    last_round_completed = int(last_round_df["max_r"].iloc[0] or 1)

    # Se il round corrente è superiore all’ultima giornata finita, limitiamo comunque a quella
    if valid_round_limit > last_round_completed:
        valid_round_limit = last_round_completed

    logger.info(
        f"🔒 Standings caricate (senza round_number nel DB) fino al round {valid_round_limit} per evitare data leakage."
    )

    df = (
        df.merge(preds, on="match_id", how="left")
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
    # 2️⃣ Conversione 'form' in indici numerici
    df["home_form_index"] = df["home_form"].apply(form_to_index)
    df["away_form_index"] = df["away_form"].apply(form_to_index)

    # -------------------------------------------------
    # 3️⃣ Calcolo risultato reale
    # -------------------------------------------------
    df["label_result"] = df.apply(compute_result, axis=1)

    # -------------------------------------------------
    # 4️⃣ Applico lo stesso feature engineering del training
    # -------------------------------------------------
    df = compute_features(df, mode="eval", cutoff_round=round_number - 1, season=season)
    logger.info("🔁 Feature numeriche e derivate calcolate correttamente per la valutazione.")

    # -------------------------------------------------
    # 5️⃣ Carico modello
    # -------------------------------------------------
    if model_path is None:
        model_dir = PATHS["models"]
        models = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
            reverse=True,
        )
        if not models:
            raise FileNotFoundError("❌ Nessun modello trovato in models/.")
        model_path = os.path.join(model_dir, models[0])

    model = joblib.load(model_path)
    logger.info("📦 Modello caricato: %s", model_path)

    # -------------------------------------------------
    # 6️⃣ Preparo dati per la valutazione
    # -------------------------------------------------
    exclude_cols = FEATURE_CONFIG["exclude_cols"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"[DEBUG] Feature usate per la valutazione: {feature_cols}")

    X_eval = df[feature_cols]
    y_true = df["label_result"]

    # predizione
    y_pred = model.predict(X_eval)

    # -------------------------------------------------
    # 7️⃣ Calcolo metriche
    # -------------------------------------------------
    valid_classes = FEATURE_CONFIG["valid_classes"]

    # filtro eventuali valori strani
    y_true = [y if y in valid_classes else "D" for y in y_true]
    y_pred = [y if y in valid_classes else "D" for y in y_pred]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    logger.info(
        "📈 Accuracy: %.3f, F1: %.3f, Precision: %.3f, Recall: %.3f",
        acc,
        f1,
        precision,
        recall,
    )
    logger.info("🧩 Confusion matrix:\n%s", cm)

    # -------------------------------------------------
    # 8️⃣ Salvo report valutazione
    # -------------------------------------------------
    report_dir = PATHS["reports"]
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(
        report_dir,
        f"evaluation_round_{season}_{round_number}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "season": season,
                "round": round_number,
                "metrics": {
                    "accuracy": acc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                },
                "confusion_matrix": cm.tolist(),
                "report": report,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info("💾 Report valutazione salvato in: %s", report_path)

    metrics_dict = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "samples": len(df),
        "dataset_size": len(X_eval.columns)
    }

    model_name = os.path.basename(model_path)
    log_model_run(
        report_path=report_path,
        metrics=metrics_dict,
        model_name=model_name,
        season=season,
        round_number=round_number,
        note="daily_evaluation"
    )

    logger.info("🧾 Log metriche aggiunto a model_runs.csv.")

    return report, report_path
