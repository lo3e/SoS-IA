# src/core/utils.py
"""
Utility generali per SoS-IA.
Gestisce salvataggio/caricamento di modelli e dataset,
creazione timestamp e versioning automatico.
"""

import os
import joblib
import pandas as pd
from datetime import datetime
import csv
from pathlib import Path
from src.core.config import PATHS
from src.core.logger import get_logger

logger = get_logger(__name__)

# === Utility base ===

def timestamp() -> str:
    """Restituisce timestamp leggibile per file (YYYY-MM-DD_HH-MM-SS)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# === Gestione modelli ===

def save_model(model, name_prefix="rf_match_predictor"):
    """
    Salva il modello in PATHS["models"] con versioning automatico.
    Esempio: rf_match_predictor_2025-11-01_18-30.pkl
    """
    model_dir = PATHS["models"]
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{name_prefix}_{timestamp()}.pkl"
    file_path = model_dir / filename
    joblib.dump(model, file_path)
    logger.info(f"✅ Modello salvato in {file_path}")
    return file_path


def load_latest_model(name_prefix="rf_match_predictor"):
    """
    Carica il modello più recente disponibile in PATHS["models"].
    """
    model_dir = PATHS["models"]
    models = sorted(model_dir.glob(f"{name_prefix}_*.pkl"))
    if not models:
        raise FileNotFoundError(f"Nessun modello trovato in {model_dir}")
    latest = models[-1]
    logger.info(f"📦 Caricato modello più recente: {latest.name}")
    return joblib.load(latest)


# === Gestione dataset ===

def save_dataframe(df: pd.DataFrame, filename: str):
    """
    Salva un DataFrame come CSV in PATHS["processed"].
    """
    out_dir = PATHS["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    logger.info(f"📊 Dataset salvato in {out_path}")
    return out_path


def load_dataframe(filename: str) -> pd.DataFrame:
    """
    Carica un DataFrame CSV da PATHS["processed"].
    """
    in_path = PATHS["processed"] / filename
    if not in_path.exists():
        raise FileNotFoundError(f"File {in_path} non trovato.")
    logger.info(f"📂 Caricamento dataset da {in_path}")
    return pd.read_csv(in_path)


# === Report e metriche ===

def save_report(content: str, filename_prefix="evaluation_report", ext="txt"):
    """
    Salva un file di testo (es. report valutazione) in PATHS["reports"].
    """
    report_dir = PATHS["reports"]
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{filename_prefix}_{timestamp()}.{ext}"
    file_path = report_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"📝 Report salvato in {file_path}")
    return file_path

def form_to_index(form_str: str, max_length: int = 5) -> float:
    """
    Converte la 'form' (es. 'WDLWW') in un indice numerico medio.

    W = 3 punti, D = 1 punto, L = 0 punti
    Se mancano dati o la stringa è vuota, restituisce 0.

    Args:
        form_str (str): sequenza risultati (es. "WDLWW")
        max_length (int): numero massimo di partite recenti da considerare

    Returns:
        float: media dei punti (0–3)
    """
    if not isinstance(form_str, str) or not form_str:
        return 0.0
    mapping = {"W": 3, "D": 1, "L": 0}
    values = [mapping.get(ch, 0) for ch in form_str[-max_length:]]
    return round(sum(values) / len(values), 2) if values else 0.0

def log_model_run(report_path: str, metrics: dict, model_name: str, season: int, round_number: int, note: str = ""):
    """
    Registra le metriche di una valutazione in un CSV cumulativo.
    """
    csv_path = os.path.join(os.path.dirname(report_path), "model_runs.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "season": season,
        "round": round_number,
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "samples": metrics.get("samples", ""),
        "dataset_size": metrics.get("dataset_size", ""),
        "note": note
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)