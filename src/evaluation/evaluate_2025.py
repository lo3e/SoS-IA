# src/evaluation/evaluate_2025.py
"""
Modulo di valutazione del modello SoS-IA sulle partite concluse (FT).

Consente di valutare:
- una stagione intera (es. 2025)
- oppure un singolo round (es. l'ultima giornata completata)

Salva un report JSON completo e logga i risultati.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
)

from src.core.config import PATHS
from src.core.logger import get_logger
from src.core.db import fetch_df

logger = get_logger(__name__)

def encode_form(form_string):
        """Converte sequenze tipo 'WWDL' in un punteggio medio numerico."""
        if not isinstance(form_string, str) or form_string.strip() == "":
            return 0.0
        mapping = {"W": 3, "D": 1, "L": 0}
        values = [mapping.get(ch, 0) for ch in form_string.strip() if ch in mapping]
        return np.mean(values) if values else 0.0

# =========================================================
# ⚙️ Funzione principale
# =========================================================
def evaluate_on_round(round_number: int, season: int = 2025) -> str:
    """
    Valuta il modello sull’ultima giornata completata o su un round specifico.
    Legge direttamente i dati dal DB (non dal CSV) e calcola tutte le metriche principali.
    """
    logger.info(f"📊 Avvio valutazione per stagione {season}, round {round_number}...")

    # ------------------------------
    # 1️⃣ Query al DB
    # ------------------------------
    query = """
        SELECT 
            m.match_id,
            m.season,
            m.round,
            m.date,
            m.home_team_name,
            m.away_team_name,
            m.home_goals,
            m.away_goals,
            p.prob_home,
            p.prob_draw,
            p.prob_away,
            sh.points AS home_points,
            sa.points AS away_points,
            sh.rank AS home_rank,
            sa.rank AS away_rank,
            sh.form AS home_form,
            sa.form AS away_form
        FROM matches m
        LEFT JOIN predictions p ON m.match_id = p.match_id
        LEFT JOIN standings sh ON sh.team_id = m.home_team_id AND sh.season = m.season
        LEFT JOIN standings sa ON sa.team_id = m.away_team_id AND sa.season = m.season
        WHERE m.season = ?
          AND m.round = ?
          AND m.status = 'FT'
    """
    round_str = f"Regular Season - {round_number}"
    df = fetch_df(query, params=(season, round_str))

    if df.empty:
        raise ValueError(f"⚠️ Nessuna partita trovata per round {round_number} (season {season}).")

    logger.info(f"✅ Partite da valutare: {len(df)}")

    # 🔧 Cast dei gol a numerico, coerente col training
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    
    # Calcola etichetta risultato come nel training
    df["label_result"] = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] < df["away_goals"]
        ],
        ["H", "A"],
        default="D"
    )

    # -------------------------------
    # 🧹 Normalizzazione dati grezzi
    # -------------------------------

    # Conversione percentuali "60%" → 0.60
    for col in ["prob_home", "prob_away", "prob_draw"]:
        if col in df.columns:
            df[col] = (
                df[col]
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------
    # 🧠 Conversione forma in indici numerici
    # -------------------------------

    if "form_home" in df.columns:
        df["home_form_index"] = df["form_home"].apply(encode_form)
    else:
        df["home_form_index"] = 0.0

    if "form_away" in df.columns:
        df["away_form_index"] = df["form_away"].apply(encode_form)
    else:
        df["away_form_index"] = 0.0

    logger.info("🧠 Form dei team convertita in indici numerici.")

    # -------------------------------
    # 🔁 Feature derivate e di equilibrio
    # -------------------------------
    df["points_diff"] = df["home_points"] - df["away_points"]
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["form_diff"] = df["home_form_index"] - df["away_form_index"]

    df["expected_draw_tendency"] = 1 - abs(df["prob_home"] - df["prob_away"])
    df["rank_balance"] = 1 / (1 + abs(df["rank_diff"]))
    df["points_balance"] = 1 / (1 + abs(df["points_diff"]))
    df["form_balance"] = 1 / (1 + abs(df["form_diff"]))

    logger.info("🔁 Feature numeriche e derivate calcolate correttamente per la valutazione.")

    # ------------------------------
    # 2️⃣ Encoding “form”
    # ------------------------------
    def encode_form(s):
        if not isinstance(s, str):
            return 0
        mapping = {"W": 3, "D": 1, "L": 0}
        vals = [mapping.get(ch, 0) for ch in s]
        return sum(vals) / len(vals) if vals else 0

    df["home_form_index"] = df["home_form"].apply(encode_form)
    df["away_form_index"] = df["away_form"].apply(encode_form)

    # ------------------------------
    # 4️⃣ Caricamento modello più recente
    # ------------------------------
    model_dir = PATHS["models"]
    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not models:
        raise FileNotFoundError("❌ Nessun modello trovato in models/.")
    models.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    latest_model = os.path.join(model_dir, models[0])

    logger.info(f"📦 Carico modello: {latest_model}")
    model = joblib.load(latest_model)

    # ------------------------------
    # 5️⃣ Prepara feature numeriche
    # ------------------------------
    feature_cols = [
        "prob_home", "prob_draw", "prob_away",
        "home_points", "away_points", "points_diff",
        "home_rank", "away_rank", "rank_diff",
        "home_form_index", "away_form_index", "form_diff",
        "expected_draw_tendency", "rank_balance", "points_balance", "form_balance"
    ]
    X_eval = df[feature_cols].fillna(0)
    y_true = df["label_result"]

    # ------------------------------
    # 6️⃣ Predizioni e metriche
    # ------------------------------
    y_pred = model.predict(X_eval)

    # 🔄 Conversione numerica → stringa, per coerenza con y_true
    label_map = {0: "H", 1: "D", 2: "A"}
    y_pred = [label_map.get(y, "D") for y in y_pred]

    valid_classes = ["H", "D", "A"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    report = classification_report(y_true, y_pred, output_dict=True)

    logger.info(f"📈 Accuracy: {acc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    logger.info(f"🧩 Confusion matrix:\n{cm}")

    # ------------------------------
    # 7️⃣ Salvataggio report JSON
    # ------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"evaluation_round_{season}_{round_number}_{timestamp}.json"
    report_path = os.path.join(PATHS["reports"], report_name)

    os.makedirs(PATHS["reports"], exist_ok=True)

    metrics = {
        "timestamp": timestamp,
        "season": season,
        "round": round_number,
        "model_used": os.path.basename(latest_model),
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "n_matches": len(df),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"💾 Report valutazione salvato in: {report_path}")
    return report_path


# =========================================================
# 🧮 Helper
# =========================================================
def _round_to_int(round_str: str | None) -> int:
    if not round_str:
        return -1
    try:
        return int(round_str.split("-")[-1].strip())
    except ValueError:
        return -1


# =========================================================
# 🚀 Entry point CLI
# =========================================================
def main():
    """Esecuzione manuale per test"""
    path = evaluate_on_round(round_number=10, season=2025)
    logger.info(f"Risultati salvati in: {path}")


if __name__ == "__main__":
    main()
