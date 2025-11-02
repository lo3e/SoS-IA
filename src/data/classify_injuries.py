# src/data/classify_injuries.py
"""
Classifica gli infortuni e crea/aggiorna la vista current_unavailable_players.
Ora usa il DB_PATH del core.
"""

from src.core.config import DB_PATH
from src.core.logger import get_logger
import sqlite3

logger = get_logger(__name__)


def add_category_column(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(injuries);")
    cols = [row[1] for row in cur.fetchall()]
    if "category" not in cols:
        logger.info("🧩 Aggiungo colonna 'category' alla tabella injuries...")
        cur.execute("ALTER TABLE injuries ADD COLUMN category TEXT;")
        conn.commit()
    else:
        logger.info("✅ Colonna 'category' già presente.")


def classify_injuries(conn):
    logger.info("🔍 Classifico gli infortuni per categoria...")
    cur = conn.cursor()
    cur.execute("""
        UPDATE injuries
        SET category = CASE
            WHEN reason LIKE '%Injury%' OR reason LIKE '%Fracture%' OR reason LIKE '%Tendon%' OR
                 reason LIKE '%Surgery%' OR reason LIKE '%Illness%' OR reason LIKE '%Virus%' OR
                 reason LIKE '%Flu%' OR reason LIKE '%Health%' OR reason LIKE '%Pain%' OR
                 reason LIKE '%Infected%' OR reason LIKE '%Sprain%' OR reason LIKE '%Dislocation%' THEN 'injury'
            WHEN reason LIKE '%Card%' OR reason LIKE '%Suspens%' THEN 'suspension'
            WHEN reason LIKE '%National%' OR reason LIKE '%International%' THEN 'international'
            WHEN reason LIKE '%Coach%' OR reason LIKE '%Inactive%' OR reason LIKE '%Fitness%' OR
                 reason LIKE '%Rest%' OR reason LIKE '%Personal%' OR reason LIKE '%Unfit%' THEN 'tactical'
            ELSE 'other'
        END;
    """)
    conn.commit()
    logger.info("✅ Classificazione completata.")


def create_view(conn):
    logger.info("🧱 Creo/aggiorno la vista current_unavailable_players...")
    cur = conn.cursor()
    cur.execute("DROP VIEW IF EXISTS current_unavailable_players;")
    cur.execute("""
        CREATE VIEW current_unavailable_players AS
        WITH next_round AS (
            SELECT m.round
            FROM matches m
            WHERE m.status = 'NS'
            ORDER BY m.date
            LIMIT 1
        )
        SELECT DISTINCT
            m.season,
            i.match_id,
            t.name AS team_name,
            i.player_name,
            i.category,
            i.reason
        FROM injuries i
        JOIN matches m ON i.match_id = m.match_id
        JOIN teams t ON i.team_id = t.team_id
        WHERE m.status = 'NS'
          AND m.round = (SELECT round FROM next_round)
        ORDER BY team_name, player_name;
    """)
    conn.commit()
    logger.info("✅ Vista 'current_unavailable_players' aggiornata.")


def preview_unavailables(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT team_name, player_name, category, reason
        FROM current_unavailable_players
        LIMIT 20;
    """)
    rows = cur.fetchall()
    if not rows:
        logger.info("   Nessun giocatore indisponibile per le prossime partite.")
    else:
        for team, player, cat, reason in rows:
            logger.info(f"   ⚽ {team:<20} | {player:<25} | {cat:<12} | {reason}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        add_category_column(conn)
        classify_injuries(conn)
        create_view(conn)
        preview_unavailables(conn)
        logger.info("🎯 Operazione infortuni completata.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
