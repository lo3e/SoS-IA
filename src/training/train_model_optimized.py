# src/training/train_model_optimized.py
"""
Training del modello Random Forest ottimizzato e calibrato per SoS-IA.

- Usa il dataset generato da prepare_training_data_v2
- Filtra dinamicamente fino al cutoff_round definito
- Applica GridSearchCV + CalibratedClassifierCV
- Salva modello e metadati versionati in PATHS["models"]
"""

import os
import json
from datetime import datetime
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.core.config import PATHS
from src.core.logger import get_logger

logger = get_logger(__name__)


# =========================================================
# 🔧 1️⃣ Funzione principale
# =========================================================
def train_with_cutoff(
    dataset_path: str | None = None,
    cutoff_round: int | None = None,
    min_season: int = 2021,
    model_version: str | None = None,
) -> str:
    """
    Addestra il modello Random Forest ottimizzato fino al cutoff specificato.

    Args:
        dataset_path (str): path al CSV del dataset. Se None, cerca l’ultimo generato.
        cutoff_round (int | None): round massimo incluso (solo per stagione corrente)
        min_season (int): stagione minima da includere (esclude 2020)
        model_version (str | None): nome opzionale del modello; se None → timestamp

    Returns:
        str: percorso completo del modello salvato (.pkl)
    """

    # --------------------------
    # 1️⃣ Caricamento dataset
    # --------------------------
    if dataset_path is None:
        # usa l'ultimo training_data_*.csv nel path processed
        processed_dir = PATHS["processed"]
        csv_files = [f for f in os.listdir(processed_dir) if f.startswith("training_data_") and f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("❌ Nessun dataset disponibile in processed/.")
        csv_files.sort(reverse=True)
        dataset_path = os.path.join(processed_dir, csv_files[0])

    logger.info(f"📂 Carico dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # --------------------------
    # 2️⃣ Filtro dinamico
    # --------------------------
    if cutoff_round is not None:
        df = df[
            ~((df["season"] == df["season"].max()) & (df["round"].apply(_round_to_int) >= cutoff_round))
        ]
        logger.info(f"🔪 Filtro applicato: esclusi match >= round {cutoff_round}")

    df = df[df["season"] >= min_season]
    logger.info(f"✅ Dataset finale per training: {len(df)} righe")

    # --------------------------
    # 3️⃣ Definizione feature
    # --------------------------
    feature_cols = [
        "prob_home", "prob_draw", "prob_away",
        "home_points", "away_points", "points_diff",
        "home_rank", "away_rank", "rank_diff",
        "home_form_index", "away_form_index", "form_diff",
        "expected_draw_tendency", "rank_balance", "points_balance", "form_balance"
    ]

    X = df[feature_cols].fillna(0)
    y = df["label_result"]

    # split train/test (solo per valutazione interna)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # --------------------------
    # 4️⃣ Grid Search
    # --------------------------
    logger.info("🔍 Avvio GridSearchCV per RandomForest...")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, 16],
        "min_samples_split": [4, 6, 8],
        "min_samples_leaf": [2, 3, 4],
    }

    grid = GridSearchCV(
        rf, param_grid, cv=3, scoring="f1_macro", verbose=1, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    logger.info(f"🏆 Migliori parametri trovati: {grid.best_params_}")

    # --------------------------
    # 5️⃣ Calibrazione del modello
    # --------------------------
    logger.info("⚖️ Calibrazione del modello (sigmoid)...")
    calibrated = CalibratedClassifierCV(best_rf, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # --------------------------
    # 6️⃣ Valutazione interna
    # --------------------------
    y_pred = calibrated.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"📈 Accuracy: {acc:.3f} | F1-macro: {f1:.3f}")
    logger.info(f"🧩 Confusion matrix:\n{cm}")

    # --------------------------
    # 7️⃣ Salvataggio modello e metadati
    # --------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = model_version or f"rf_v3_{timestamp}"

    model_dir = PATHS["models"]
    model_path = os.path.join(model_dir, f"{model_version}.pkl")
    meta_path = os.path.join(model_dir, f"{model_version}.json")

    joblib.dump(calibrated, model_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
    }

    metadata_path = PATHS["models"] / "rf_match_predictor_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"💾 Modello salvato: {model_path}")
    logger.info(f"🧾 Metadati salvati: {meta_path}")

    return str(model_path), str(meta_path)


# =========================================================
# ⚙️ Helper
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
    """Esecuzione standalone per test manuale."""
    path = train_with_cutoff(min_season=2021)
    logger.info(f"Modello completato e salvato in: {path}")


if __name__ == "__main__":
    main()
