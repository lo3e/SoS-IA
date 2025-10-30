import sqlite3
from pathlib import Path

# === CONFIG ===
DB_PATH = Path(r"C:\Users\brain\Documents\ProgettiPersonali\SoS-IA\DB\sos_ia.db")

def add_category_column(conn):
    c = conn.cursor()
    c.execute("PRAGMA table_info(injuries);")
    cols = [row[1] for row in c.fetchall()]
    if "category" not in cols:
        print("🧩 Aggiungo colonna 'category' alla tabella injuries...")
        c.execute("ALTER TABLE injuries ADD COLUMN category TEXT;")
        conn.commit()
    else:
        print("✅ Colonna 'category' già presente.")

def classify_injuries(conn):
    print("🔍 Classifico gli infortuni per categoria...")
    c = conn.cursor()

    c.execute("""
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
    print("✅ Classificazione completata.")

def create_view(conn):
    print("🧱 Creo (o aggiorno) la vista current_unavailable_players basata su 'round'...")
    c = conn.cursor()
    c.execute("DROP VIEW IF EXISTS current_unavailable_players;")
    c.execute("""
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
    print("✅ Vista 'current_unavailable_players' aggiornata con successo (basata su round).")

def preview_unavailables(conn):
    print("\n📋 Esempio giocatori indisponibili (partite future):")
    c = conn.cursor()
    c.execute("""
        SELECT team_name, player_name, category, reason
        FROM current_unavailable_players
        LIMIT 20;
    """)
    rows = c.fetchall()
    if not rows:
        print("   Nessun giocatore indisponibile per le prossime partite.")
    else:
        for team, player, cat, reason in rows:
            print(f"   ⚽ {team:<20} | {player:<25} | {cat:<12} | {reason}")

def main():
    print("🚀 Classificazione e vista indisponibili giocatori\n")
    conn = sqlite3.connect(DB_PATH)
    try:
        add_category_column(conn)
        classify_injuries(conn)
        create_view(conn)
        preview_unavailables(conn)
        print("\n🎯 Operazione completata con successo.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
