# src/betting/bolletta.py
"""
Generatore di schedine per SoS-IA.

Produce:
- schedina "pura"  → previsione più probabile (Home / Draw / Away)
- schedina "value" → solo match dove la probabilità del modello è > probabilità implicita della quota

È allineato con:
- prepare_training_data_v2 (stesse feature base)
- evaluate_2025 (stessa normalizzazione)
- train_model_optimized (usa FEATURE_CONFIG per scegliere le colonne)
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.core.config import PATHS, FEATURE_CONFIG, MODEL_CONFIG
from src.core.db import fetch_df
from src.core.logger import get_logger
from src.features.feature_engineering import compute_features  # usiamo la stessa funzione degli altri

logger = get_logger(__name__)


def _load_model(model_path: str | None = None) -> tuple[object, str]:
    """
    Carica il modello. Se non viene passato un path, prende l'ultimo .pkl in PATHS["models"].
    """
    if model_path is None:
        model_dir = PATHS["models"]
        models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if not models:
            raise FileNotFoundError("❌ Nessun modello trovato in models/.")
        # ordino per data di modifica, più recente per primo
        models.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
        model_path = os.path.join(model_dir, models[0])
        logger.info("📦 Modello (fallback) caricato: %s", os.path.basename(model_path))
    else:
        logger.info("📦 Modello (passato dal daily) caricato: %s", os.path.basename(model_path))

    model = joblib.load(model_path)
    return model, model_path


def _fetch_fixtures_with_context(season: int, round_number: int) -> pd.DataFrame:
    """
    Recupera dal DB le partite non iniziate di quel turno con:
    - prob API
    - odds aggregate
    - standings (punti, rank, form)
    """
    query = f"""
        SELECT 
            m.match_id, m.date, m.round, m.season,
            m.home_team_name, m.away_team_name,
            p.prob_home, p.prob_draw, p.prob_away,
            o.odd_home, o.odd_draw, o.odd_away,
            sh.points AS home_points, sa.points AS away_points,
            sh.rank AS home_rank, sa.rank AS away_rank,
            sh.form AS home_form, sa.form AS away_form
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
    df = fetch_df(query)

    if df.empty:
        raise ValueError(f"⚠️ Nessuna partita trovata per il round {round_number}")

    # aggiungo round_progress esattamente come in evaluate
    try:
        max_round_db = fetch_df(
            f"""
            SELECT MAX(CAST(SUBSTR(round, INSTR(round, '-') + 1) AS INTEGER)) AS max_r
            FROM matches
            WHERE season = {season}
            """
        )["max_r"].iloc[0]
        if not max_round_db or max_round_db == 0:
            max_round_db = round_number
    except Exception:
        max_round_db = round_number

    df["round_progress"] = round_number / max_round_db

    return df


def _probs_from_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Ottiene le proba dal modello e le rimappa sempre su Home/Draw/Away
    in base a model.classes_ (così non sbagliamo l'ordine).
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)  # es. ["A","D","H"] o ["H","D","A"]

    # inizializzo con NaN
    out = pd.DataFrame(index=X.index, columns=["p_home", "p_draw", "p_away"], dtype=float)

    for i, cls in enumerate(classes):
        if cls in ("H", "Home"):
            out["p_home"] = proba[:, i]
        elif cls in ("D", "Draw"):
            out["p_draw"] = proba[:, i]
        elif cls in ("A", "Away"):
            out["p_away"] = proba[:, i]

    # se qualcuna è rimasta vuota, la metto a 0
    out = out.fillna(0.0)
    return out


def generate_for_round(
    round_number: int,
    season: int = 2025,
    mode: str = "pure",
    model_path: str | None = None,
) -> str:
    """
    Genera la schedina per una giornata specifica.
    mode:
      - "pure"  → previsione più probabile
      - "value" → filtra solo le value bet
    """
    logger.info("🎯 Generazione bolletta: stagione %s, round %s, modalità %s", season, round_number, mode)

    # 1) modello
    model, used_model_path = _load_model(model_path)

    # 2) partite
    fixtures = _fetch_fixtures_with_context(season, round_number)
    logger.info("📅 Partite trovate: %s", len(fixtures))

    # 3) normalizzazione e feature engineering identica al resto
    #    (usa la funzione comune che abbiamo messo in src.features.feature_engineering)
    fixtures = compute_features(fixtures, mode="predict", season=season)  # crea home_form_index, diff, balance, ecc.

    # 4) preparo X con le stesse colonne usate in training
    #    NB: togliamo anche home_form / away_form perché sono stringhe
    exclude_cols = FEATURE_CONFIG["exclude_cols"]
    feature_cols = [c for c in fixtures.columns if c not in exclude_cols]

    # 🧭 Allineamento colonne con quelle del modello
    feature_list_path = os.path.join(
        os.path.dirname(used_model_path),
        f"{MODEL_CONFIG['feature_list_prefix']}{os.path.basename(used_model_path).replace('.pkl', '')}{MODEL_CONFIG['feature_list_ext']}"
    )

    if os.path.exists(feature_list_path):
        with open(feature_list_path, "r", encoding="utf-8") as f:
            model_features = [line.strip() for line in f.readlines() if line.strip()]
        # tieni solo le colonne comuni e nell'ordine corretto
        X_pred = fixtures[feature_cols].fillna(0)
        X_pred = X_pred.reindex(columns=model_features, fill_value=0)
        logger.info(f"✅ Colonne allineate al modello ({len(model_features)} features).")
    else:
        logger.warning("⚠️ Feature list non trovata, si usa X_pred come attuale (potrebbe causare mismatch).")

    # 5) predizioni
    proba_df = _probs_from_model(model, X_pred)
    fixtures = pd.concat([fixtures, proba_df], axis=1)

    # 6) pronostico puro (scegliamo la max)
    def _best_label(row: pd.Series) -> str:
        probs = {
            "Home": row.get("p_home", 0.0),
            "Draw": row.get("p_draw", 0.0),
            "Away": row.get("p_away", 0.0),
        }
        return max(probs, key=probs.get)

    fixtures["prediction"] = fixtures.apply(_best_label, axis=1)
    fixtures["confidence"] = fixtures[["p_home", "p_draw", "p_away"]].max(axis=1)

    # 7) modalità value bet
    if mode == "value":
        def _value_pick(row: pd.Series) -> str | None:
            # se non ho quote non posso fare value bet
            if pd.isna(row["odd_home"]) or pd.isna(row["odd_draw"]) or pd.isna(row["odd_away"]):
                return None

            implied = {
                "Home": 1 / row["odd_home"] if row["odd_home"] else 0,
                "Draw": 1 / row["odd_draw"] if row["odd_draw"] else 0,
                "Away": 1 / row["odd_away"] if row["odd_away"] else 0,
            }
            model_p = {
                "Home": row.get("p_home", 0.0),
                "Draw": row.get("p_draw", 0.0),
                "Away": row.get("p_away", 0.0),
            }

            # rapporto valore
            value_ratio = {}
            for k in ("Home", "Draw", "Away"):
                if implied[k] > 0:
                    value_ratio[k] = model_p[k] / implied[k]
                else:
                    value_ratio[k] = 0.0

            best = max(value_ratio, key=value_ratio.get)

            # soglia semplice: 10% di overvalue
            if value_ratio[best] > 1.1:
                return best
            return None

        fixtures["value_bet"] = fixtures.apply(_value_pick, axis=1)
        fixtures = fixtures[fixtures["value_bet"].notnull()].copy()
        fixtures["prediction"] = fixtures["value_bet"]
        fixtures.drop(columns=["value_bet"], inplace=True)
        logger.info("💰 Value bets trovate: %s", len(fixtures))

    # 8) costruisco output leggibile
    fixtures["output"] = fixtures.apply(
        lambda r: f"{r['home_team_name']} vs {r['away_team_name']} → {r['prediction']} ({r['confidence']:.2f})",
        axis=1
    )

    # 9) salvo
    os.makedirs(PATHS["reports"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"bolletta_{mode}_{season}_R{round_number}_{ts}.csv"
    out_path = os.path.join(PATHS["reports"], out_name)
    fixtures.to_csv(out_path, index=False, encoding="utf-8")

    logger.info("✅ Bolletta salvata in: %s", out_path)
    logger.info("\n" + "\n".join(fixtures["output"].tolist()))
    logger.info("📦 Modello usato: %s", os.path.basename(used_model_path))

    return out_path


def main():
    # test manuale
    generate_for_round(round_number=11, season=2025, mode="pure")


if __name__ == "__main__":
    main()
