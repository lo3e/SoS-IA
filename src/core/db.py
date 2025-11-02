# src/core/db.py
"""
Utility di accesso al database sos_ia.db
Gestisce connessione SQLite e query → pandas.DataFrame
"""

import sqlite3
import pandas as pd
from src.core.config import DB_PATH


def fetch_df(query: str, params: tuple | None = None) -> pd.DataFrame:
    """
    Esegue una query SQL su DB_PATH e restituisce un DataFrame.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def execute_query(query: str, params: tuple | None = None) -> None:
    """
    Esegue una query di scrittura (INSERT, UPDATE, DELETE).
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(query, params or ())
        conn.commit()


def fetch_value(query: str, params: tuple | None = None):
    """
    Esegue una query e restituisce un singolo valore (prima colonna della prima riga).
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(query, params or ())
        row = c.fetchone()
    return row[0] if row else None
