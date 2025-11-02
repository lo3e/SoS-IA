# src/betting/bolletta.py
"""
Generatore di schedine per SoS-IA (Serie A e altri campionati).

Produce:
- schedina "pura" → previsione più probabile (Home / Draw / Away)
- schedina "value bet" → solo match in cui la probabilità modello > implicita quota

Compatibile con il nuovo core e con i moduli di training/evaluation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.config import PATHS
from src.core.db import fetch_df
from src.core.logger import get_logger

logger = get_logger(__name__)

# =========================================================
# ⚙️ Funzioni principali
# =========================================================
def generate_for_round(round_number: int, season: int = 2025, mode: str = "pure", model_path: str | None = None) -> str:
    """
    Genera la schedina per una giornata specifica.

    Args:
        round_number (int): numero del turno da pronosticare
        season (int): stagione (default 2025)
        mode (str): "pure" per pronostico puro, "value" per value bet

    Returns:
        str: percorso del file CSV salvato
    """

    logger.info(f"🎯 Generazione bolletta: stagione {season}, round {round_number}, modalità {mode}")

    # ------------------------------
    # 1️⃣ Carico modello più recente
    # ------------------------------
        # ------------------------------
    # modello
    # ------------------------------
    if model_path is None:
        # fallback: prendo comunque l’ultimo disponibile
        model_dir = PATHS["models"]
        models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if not models:
            raise FileNotFoundError("❌ Nessun modello trovato in models/.")
        models.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
        model_path = os.path.join(model_dir, models[0])
        logger.info(f"📦 Modello (fallback) caricato: {os.path.basename(model_path)}")
    else:
        logger.info(f"📦 Modello (passato dal daily) caricato: {os.path.basename(model_path)}")

    model = joblib.load(model_path)

    # ------------------------------
    # 2️⃣ Query prossime partite
    # ------------------------------
    query = f"""
        SELECT 
            m.match_id, m.date, m.round, m.season,
            m.home_team_name, m.away_team_name,
            p.prob_home, p.prob_draw, p.prob_away,
            o.odd_home, o.odd_draw, o.odd_away,
            sh.points AS home_points, sa.points AS away_points,
            sh.rank AS home_rank, sa.rank AS away_rank,
            sh.form AS form_home, sa.form AS form_away
        FROM matches m
        LEFT JOIN predictions p ON m.match_id = p.match_id
        LEFT JOIN (
            SELECT match_id,
                   MAX(CASE WHEN market='Match Winner' AND outcome='Home' THEN odd END) AS odd_home,
                   MAX(CASE WHEN market='Match Winner' AND outcome='Draw' THEN odd END) AS odd_draw,
                   MAX(CASE WHEN market='Match Winner' AND outcome='Away' THEN odd END) AS odd_away
            FROM odds
            GROUP BY match_id
        ) o ON m.match_id = o.match_id
        LEFT JOIN standings sh ON sh.team_id = m.home_team_id AND sh.season = m.season
        LEFT JOIN standings sa ON sa.team_id = m.away_team_id AND sa.season = m.season
        WHERE m.season = {season}
          AND m.round = 'Regular Season - {round_number}'
          AND m.status IN ('NS', 'TBD', 'PST')
    """
    fixtures = fetch_df(query)
    if fixtures.empty:
        raise ValueError(f"⚠️ Nessuna partita trovata per il round {round_number}")

    logger.info(f"📅 Partite trovate: {len(fixtures)}")

    # ------------------------------
    # 3️⃣ Predizioni del modello
    # ------------------------------
    features = ["prob_home", "prob_draw", "prob_away"]  # se servono altre feature, si estende
    X_pred = fixtures[features].copy()

    # 🧹 normalizzo le probabilità che arrivano dall’API (tipo "63%")
    for col in ["prob_home", "prob_draw", "prob_away"]:
        if col in X_pred.columns:
            X_pred[col] = (
                X_pred[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float)
                / 100
            )

        # ------------------------------
    # 🧹 Normalizzazione e feature engineering come in training/evaluate
    # ------------------------------

    # 1️⃣ Conversione percentuali (es. "65%") in float
    for col in ["prob_home", "prob_away", "prob_draw"]:
        if col in fixtures.columns:
            fixtures[col] = (
                fixtures[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float)
                / 100
            )

    # 2️⃣ Conversione forma in indice numerico
    def encode_form(form_string):
        mapping = {"W": 3, "D": 1, "L": 0}
        if not isinstance(form_string, str):
            return 0.0
        values = [mapping.get(ch, 0) for ch in form_string if ch in mapping]
        return np.mean(values) if values else 0.0

    if "form_home" in fixtures.columns:
        fixtures["home_form_index"] = fixtures["form_home"].apply(encode_form)
    else:
        fixtures["home_form_index"] = 0.0

    if "form_away" in fixtures.columns:
        fixtures["away_form_index"] = fixtures["form_away"].apply(encode_form)
    else:
        fixtures["away_form_index"] = 0.0

    # 3️⃣ Feature derivate
    fixtures["points_diff"] = fixtures["home_points"] - fixtures["away_points"]
    fixtures["rank_diff"] = fixtures["home_rank"] - fixtures["away_rank"]
    fixtures["form_diff"] = fixtures["home_form_index"] - fixtures["away_form_index"]

    # equilibrio
    fixtures["expected_draw_tendency"] = 1 - abs(fixtures["prob_home"] - fixtures["prob_away"])
    fixtures["rank_balance"] = 1 / (1 + abs(fixtures["rank_diff"]))
    fixtures["points_balance"] = 1 / (1 + abs(fixtures["points_diff"]))
    fixtures["form_balance"] = 1 / (1 + abs(fixtures["form_diff"]))

    logger.info("🔁 Feature numeriche e derivate calcolate correttamente per la bolletta.")

    # 4️⃣ Preparo X_pred con lo stesso schema del modello
    feature_cols = [
        "prob_home", "prob_draw", "prob_away",
        "home_points", "away_points", "points_diff",
        "home_rank", "away_rank", "rank_diff",
        "home_form_index", "away_form_index", "form_diff",
        "expected_draw_tendency", "rank_balance", "points_balance", "form_balance"
    ]
    X_pred = fixtures[feature_cols].fillna(0)
        
    preds = model.predict_proba(X_pred)
    preds_df = pd.DataFrame(preds, columns=["p_away", "p_draw", "p_home"])  # RF è inverso nei classi a volte
    fixtures = pd.concat([fixtures, preds_df], axis=1)

    # ------------------------------
    # 4️⃣ Pronostico puro
    # ------------------------------
    def get_prediction_label(row):
        probs = {"Home": row["p_home"], "Draw": row["p_draw"], "Away": row["p_away"]}
        return max(probs, key=probs.get)

    fixtures["prediction"] = fixtures.apply(get_prediction_label, axis=1)
    fixtures["confidence"] = fixtures[["p_home", "p_draw", "p_away"]].max(axis=1)

    # ------------------------------
    # 5️⃣ Value bet (se richiesta)
    # ------------------------------
    if mode == "value":
        def value_bet(row):
            if pd.isna(row["odd_home"]) or pd.isna(row["odd_draw"]) or pd.isna(row["odd_away"]):
                return None
            implied_probs = {
                "Home": 1 / row["odd_home"] if row["odd_home"] else 0,
                "Draw": 1 / row["odd_draw"] if row["odd_draw"] else 0,
                "Away": 1 / row["odd_away"] if row["odd_away"] else 0,
            }
            pred_probs = {"Home": row["p_home"], "Draw": row["p_draw"], "Away": row["p_away"]}
            value = {
                k: (pred_probs[k] / implied_probs[k]) if implied_probs[k] > 0 else 0
                for k in pred_probs
            }
            best = max(value, key=value.get)
            if value[best] > 1.1:  # threshold value bet
                return best
            return None

        fixtures["value_bet"] = fixtures.apply(value_bet, axis=1)
        fixtures = fixtures[fixtures["value_bet"].notnull()]
        fixtures.rename(columns={"value_bet": "prediction"}, inplace=True)
        logger.info(f"💰 Value bets trovate: {len(fixtures)}")

    # ------------------------------
    # 6️⃣ Output finale
    # ------------------------------
    fixtures["output"] = fixtures.apply(
        lambda r: f"{r['home_team_name']} vs {r['away_team_name']} → {r['prediction']} ({r['confidence']:.2f})",
        axis=1
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"bolletta_{mode}_{season}_R{round_number}_{timestamp}.csv"
    out_path = os.path.join(PATHS["reports"], out_name)
    os.makedirs(PATHS["reports"], exist_ok=True)
    fixtures.to_csv(out_path, index=False)

    logger.info(f"✅ Bolletta salvata in: {out_path}")
    logger.info("\n" + "\n".join(fixtures["output"].tolist()))

    return out_path


# =========================================================
# 🚀 Entry point CLI
# =========================================================
def main():
    """Esecuzione manuale"""
    generate_for_round(round_number=11, season=2025, mode="pure")


if __name__ == "__main__":
    main()
