# src/training/train_model_optimized.py
"""
Training del modello Random Forest ottimizzato e calibrato per SoS-IA.

- Usa il dataset generato da prepare_training_data_v2
- Se serve, rifiltra fino al cutoff_round (solo stagione corrente)
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

from src.core.utils import form_to_index
from src.core.config import FEATURE_CONFIG, PATHS, MODEL_CONFIG
from src.core.logger import get_logger

logger = get_logger(__name__)


def _round_to_int(round_str: str | None) -> int:
    """Converte 'Regular Season - 9' → 9"""
    if not round_str:
        return -1
    try:
        return int(round_str.split("-")[-1].strip())
    except ValueError:
        return -1


def train_with_cutoff(
    dataset_path: str | None = None,
    cutoff_round: int | None = None,
    min_season: int = 2021,
    model_version: str | None = None,
) -> tuple[str, str]:
    """
    Addestra il modello Random Forest ottimizzato fino al cutoff specificato.

    Args:
        dataset_path (str | None): path al CSV del dataset. Se None, prende l'ultimo in processed/.
        cutoff_round (int | None): round massimo incluso (solo per stagione corrente).
        min_season (int): stagione minima da includere.
        model_version (str | None): nome opzionale del modello.

    Returns:
        (model_path, metadata_path)
    """

    # -------------------------------------------------
    # 1) Carico dataset
    # -------------------------------------------------
    if dataset_path is None:
        processed_dir = PATHS["processed"]
        csv_files = [
            f for f in os.listdir(processed_dir)
            if f.startswith("training_data_") and f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError("❌ Nessun dataset disponibile in processed/.")
        csv_files.sort(reverse=True)
        dataset_path = os.path.join(processed_dir, csv_files[0])

    logger.info("📂 Carico dataset da: %s", dataset_path)
    df = pd.read_csv(dataset_path)

    # -------------------------------------------------
    # 2) Filtro per stagione e (eventuale) cutoff
    #    Nota: il daily già genera un CSV con il cutoff,
    #    ma qui lo rendiamo robusto nel caso arrivi un CSV “pieno”.
    # -------------------------------------------------
    # 2a) filtro stagione minima
    df = df[df["season"] >= min_season]

    # 2b) cutoff SOLO sulla stagione più recente
    if cutoff_round is not None:
        current_season = df["season"].max()

        if "round_number" in df.columns:
            cond_current = (df["season"] == current_season) & (df["round_number"] >= cutoff_round)
        else:
            cond_current = (df["season"] == current_season) & (df["round"].apply(_round_to_int) >= cutoff_round)

        before = len(df)
        df = df[~cond_current]
        after = len(df)
        logger.info(
            "🔪 Filtro applicato: esclusi match della stagione %s con round >= %s (%s → %s righe)",
            current_season,
            cutoff_round,
            before,
            after,
        )

    logger.info("✅ Dataset finale per training: %s righe", len(df))

    # Conversione form in indici numerici
    if "home_form" in df.columns:
        df["home_form_index"] = df["home_form"].apply(form_to_index)
    if "away_form" in df.columns:
        df["away_form_index"] = df["away_form"].apply(form_to_index)

    # -------------------------------------------------
    # 3) Definizione feature
    # -------------------------------------------------
    exclude_cols = FEATURE_CONFIG["exclude_cols"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df["label_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    # -------------------------------------------------
    # 4) GridSearchCV
    # -------------------------------------------------
    logger.info("🔍 Avvio GridSearchCV per RandomForest...")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight = "balanced")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, 16],
        "min_samples_split": [4, 6, 8],
        "min_samples_leaf": [2, 3, 4],
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="f1_macro",
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    logger.info("🏆 Migliori parametri trovati: %s", grid.best_params_)

    print("Best params:", grid.best_params_)
    print("Best score (cv mean):", grid.best_score_)

    # Se vuoi vedere il dettaglio dei punteggi per ogni combinazione
    cv_results = pd.DataFrame(grid.cv_results_)
    print(cv_results[["params", "mean_test_score", "std_test_score"]])

    # -------------------------------------------------
    # 5) Calibrazione
    # -------------------------------------------------
    logger.info("⚖️ Calibrazione del modello (sigmoid)...")
    calibrated = CalibratedClassifierCV(best_rf, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # -------------------------------------------------
    # 6) Valutazione interna
    # -------------------------------------------------
    # -------------------------------------------------
    # 6) Valutazione interna
    # -------------------------------------------------
    # 🔹 Predizioni sul TRAIN
    y_pred_train = calibrated.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="macro")

    # 🔹 Predizioni sul TEST
    y_pred_test = calibrated.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    cm = confusion_matrix(y_test, y_pred_test)

    print("\n=== Training summary ===")
    print("Best params:", grid.best_params_)
    print(f"Train accuracy: {train_acc:.3f} | F1: {train_f1:.3f}")
    print(f"Test accuracy:  {test_acc:.3f} | F1: {test_f1:.3f}")

    logger.info("📈 Accuracy holdout: %.3f | F1-macro: %.3f", test_acc, test_f1)
    logger.info("🧩 Confusion matrix:\n%s", cm)


    importances = best_rf.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(feat_imp.head(15))

    # -------------------------------------------------
    # 7) Salvataggio modello + metadati
    # -------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = model_version or f"rf_v3_{timestamp}"

    model_dir = PATHS["models"]
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_version}.pkl")
    metadata_path = os.path.join(model_dir, f"{model_version}.json")

    joblib.dump(calibrated, model_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "accuracy_holdout": float(test_acc),
        "f1_holdout": float(test_f1),
        "rows_used": int(len(df)),
        "feature_count": int(len(feature_cols)),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    # dopo aver salvato il modello
    feature_list_path = os.path.join(
        PATHS["models"],
        f"{MODEL_CONFIG['feature_list_prefix']}{model_version}{MODEL_CONFIG['feature_list_ext']}"
    )
    with open(feature_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))
    logger.info(f"📄 Feature list salvata in: {feature_list_path}")

    logger.info("💾 Modello salvato: %s", model_path)
    logger.info("🧾 Metadati salvati: %s", metadata_path)
    logger.info(f"📄 Feature list salvata in: {feature_list_path}")

    return model_path, metadata_path


def main():
    """Esecuzione standalone per test manuale."""
    model_path, meta_path = train_with_cutoff(min_season=2021)
    logger.info("Modello completato e salvato in: %s", model_path)
    logger.info("Metadati salvati in: %s", meta_path)


if __name__ == "__main__":
    main()
