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
