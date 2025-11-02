# src/core/logger.py
"""
Logging centralizzato per SoS-IA.
Crea log sia su file (in DATA_PATH/logs) sia su console.
"""

import logging
from datetime import datetime
from pathlib import Path
from src.core.config import PATHS

def get_logger(name: str = "sosia") -> logging.Logger:
    """
    Restituisce un logger configurato per il progetto.
    Evita duplicazioni di handler se già inizializzato.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # evita di aggiungere doppi handler

    logger.setLevel(logging.INFO)

    # File di log giornaliero
    log_dir = PATHS["logs"]
    log_file = log_dir / f"sosia_{datetime.now():%Y-%m-%d}.log"

    # Handler per file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Handler per console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger inizializzato → {log_file}")
    return logger


# Se eseguito direttamente, test rapido
if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Test logger completato correttamente.")
