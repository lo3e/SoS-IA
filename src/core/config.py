# src/core/config.py
"""
Modulo di configurazione centrale per SoS-IA.
Organizza i percorsi a partire dal solo DATA_PATH e DB_PATH definiti nel .env.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# === Caricamento .env ===
BASE_DIR = Path(__file__).resolve().parents[2]
dotenv_path = BASE_DIR / ".env"
if not dotenv_path.exists():
    raise FileNotFoundError(f"File .env non trovato in {dotenv_path}")
load_dotenv(dotenv_path)

# === Variabili principali definite nel .env ===
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH")

if not DB_PATH or not DATA_PATH:
    raise EnvironmentError("⚠️  DB_PATH o DATA_PATH mancanti nel file .env")

# === Struttura interna di DATA_PATH ===
PATHS = {
    "root": Path(DATA_PATH),
    "raw": Path(DATA_PATH) / "raw",
    "processed": Path(DATA_PATH) / "processed",
    "models": Path(DATA_PATH) / "models",
    "logs": Path(DATA_PATH) / "logs",
    "reports": Path(DATA_PATH) / "reports",
    "teams": Path(DATA_PATH) / "fantacalcio" / "squadre",
    "suggestions": Path(DATA_PATH) / "fantacalcio" / "suggestioni",
}

# Crea automaticamente le sottocartelle se mancano
for sub in PATHS.values():
    sub.mkdir(parents=True, exist_ok=True)

# === Costanti di progetto ===
RANDOM_STATE = 42
MODEL_VERSION = "v3"

def project_info():
    """Ritorna un riepilogo dei percorsi principali (utile per debug)."""
    return {
        "DB_PATH": DB_PATH,
        **{k: str(v) for k, v in PATHS.items()},
    }

if __name__ == "__main__":
    print("✅ Config caricata con successo:")
    for k, v in project_info().items():
        print(f"{k:12s} → {v}")

# ============================================================
# FEATURE & TRAINING CONFIGURATION
# ============================================================

FEATURE_CONFIG = {
    # Colonne da escludere prima del training o della valutazione
    "exclude_cols": [
        "match_id", "home_team_id", "away_team_id", "date",
        "home_team_name", "away_team_name", "home_season",
        "away_season", "status", "home_goals", "away_goals",
        "label_result", "round", "round_number", "season",
        "home_form", "away_form", "form", "away_valid_round",
        "home_valid_round"
    ],

    # Classi possibili del modello
    "valid_classes": ["H", "D", "A"]
}

# ============================================================
# HELPER FUNCTION: compute_result
# ============================================================

def compute_result(row) -> str:
    """Calcola il risultato (label) di una partita: H, D, A."""
    try:
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"
    except Exception:
        return "D"

# --- Model training configuration ---
MODEL_CONFIG = {
    # Nome base del file usato per salvare la lista di feature del modello
    "feature_list_prefix": "features_",
    "feature_list_ext": ".txt",
    # Nome del file JSON con metadati opzionali
    "metadata_ext": ".json",
}
