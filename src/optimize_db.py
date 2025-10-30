import os
import sqlite3
import shutil
from pathlib import Path
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

BACKUP_PATH = Path(DB_PATH).with_name("sosia_backup_before_optimize.db")

# === UTILS ===
def connect_db():
    return sqlite3.connect(DB_PATH)

def backup_db():
    print(f"💾 Creo backup di sicurezza: {BACKUP_PATH}")
    shutil.copy(DB_PATH, BACKUP_PATH)
    print("✅ Backup completato.\n")

def drop_columns(conn, table, drop_cols):
    """Ricostruisce la tabella senza le colonne da rimuovere"""
    cur = conn.cursor()

    # 1. Ottieni schema originale
    cur.execute(f"PRAGMA table_info({table});")
    cols = [c[1] for c in cur.fetchall()]
    keep_cols = [c for c in cols if c not in drop_cols]

    if not drop_cols:
        print(f"ℹ️ Nessuna colonna da eliminare in {table}")
        return

    print(f"🧹 Pulizia tabella {table}: rimuovo {drop_cols}")

    # 2. Crea nuova tabella temporanea
    cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
    create_sql = cur.fetchone()[0]

    # Rimuovi le colonne non desiderate dallo schema
    for col in drop_cols:
        start = create_sql.find(col)
        if start != -1:
            # taglio grossolano, più robusto gestendo via nuovo schema
            pass

    # 3. Leggi schema con colonne da mantenere
    col_list = ", ".join(keep_cols)
    temp_table = f"{table}_new"

    cur.execute(f"CREATE TABLE {temp_table} AS SELECT {col_list} FROM {table};")
    cur.execute(f"DROP TABLE {table};")
    cur.execute(f"ALTER TABLE {temp_table} RENAME TO {table};")
    conn.commit()

    print(f"✅ Tabella {table} ricreata senza colonne: {drop_cols}\n")

def create_indexes(conn):
    print("⚙️ Creazione indici...")
    cur = conn.cursor()
    indexes = [
        ("idx_matches_match_id", "matches", "match_id"),
        ("idx_players_match_id", "players", "match_id"),
        ("idx_team_stats_match_id", "team_stats", "match_id"),
        ("idx_events_match_id", "events", "match_id"),
        ("idx_predictions_match_id", "predictions", "match_id"),
        ("idx_injuries_match_id", "injuries", "match_id"),
        ("idx_odds_match_id", "odds", "match_id"),
        ("idx_standings_season_team", "standings", "season, team_id")
    ]
    for name, table, cols in indexes:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols});")
    conn.commit()
    print("✅ Indici creati.\n")

def create_views(conn):
    print("🧩 Creazione viste logiche...")
    cur = conn.cursor()

    cur.executescript("""
    -- 1️⃣ Vista per modello AI
    CREATE VIEW IF NOT EXISTS match_features_view AS
    SELECT
        m.match_id,
        m.season,
        m.home_team_id,
        m.away_team_id,
        m.home_goals,
        m.away_goals,
        p.winner,
        p.prob_home,
        p.prob_draw,
        p.prob_away,
        sh.rank AS home_rank,
        sa.rank AS away_rank,
        sh.points AS home_points,
        sa.points AS away_points
    FROM matches m
    LEFT JOIN predictions p ON m.match_id = p.match_id
    LEFT JOIN standings sh ON sh.team_id = m.home_team_id AND sh.season = m.season
    LEFT JOIN standings sa ON sa.team_id = m.away_team_id AND sa.season = m.season;

    -- 2️⃣ Vista partite future
    -- 🔄 Ricrea la vista next_fixtures_view con quote medie per match
    DROP VIEW IF EXISTS next_fixtures_view;

    CREATE VIEW next_fixtures_view AS
    SELECT
        m.match_id,
        m.date,
        m.home_team_name,
        m.away_team_name,
        p.winner,
        p.prob_home,
        p.prob_draw,
        p.prob_away,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Home' THEN o.odd END), 2) AS avg_home_odd,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Draw' THEN o.odd END), 2) AS avg_draw_odd,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Away' THEN o.odd END), 2) AS avg_away_odd,
        COUNT(DISTINCT o.bookmaker_name) AS bookmakers_count
    FROM matches m
    LEFT JOIN predictions p ON m.match_id = p.match_id
    LEFT JOIN odds o ON m.match_id = o.match_id
    WHERE m.status = 'NS'
    GROUP BY m.match_id
    ORDER BY m.date ASC;


    -- 3️⃣ Vista riepilogo infortuni
    CREATE VIEW IF NOT EXISTS injury_summary_view AS
    SELECT
        season,
        t.name AS team_name,
        COUNT(i.id) AS total_injuries
    FROM injuries i
    JOIN matches m ON i.match_id = m.match_id
    JOIN teams t ON i.team_id = t.team_id
    GROUP BY season, team_name;

    -- 4️⃣ Vista classifica leggibile
    CREATE VIEW IF NOT EXISTS standings_view AS
    SELECT
        season,
        league_id,
        rank,
        team_name,
        points,
        goals_for,
        goals_against,
        goals_diff,
        form
    FROM standings
    ORDER BY season DESC, rank ASC;

    -- 5️⃣ Vista posizioni giocatori
    CREATE VIEW IF NOT EXISTS player_positions_view AS
    SELECT
        p.*,
        l.position
    FROM players p
    LEFT JOIN lineups l
    ON p.player_id = l.player_id AND p.match_id = l.match_id;

    -- 6️⃣ Vista statistiche giocatori pulite
    CREATE VIEW IF NOT EXISTS player_stats_clean_view AS
    SELECT
        player_id,
        match_id,
        team_id,
        player_name,
        COALESCE(minutes, 0) AS minutes,
        COALESCE(rating, 0) AS rating,
        COALESCE(shots_total, 0) AS shots_total,
        COALESCE(shots_on, 0) AS shots_on,
        COALESCE(goals_total, 0) AS goals_total,
        COALESCE(assists, 0) AS assists,
        COALESCE(passes_total, 0) AS passes_total,
        COALESCE(passes_key, 0) AS passes_key,
        COALESCE(tackles, 0) AS tackles,
        COALESCE(interceptions, 0) AS interceptions,
        COALESCE(duels_total, 0) AS duels_total,
        COALESCE(duels_won, 0) AS duels_won,
        COALESCE(yellow_cards, 0) AS yellow_cards,
        COALESCE(red_cards, 0) AS red_cards
    FROM players;
    """)

    conn.commit()
    print("✅ Viste create.\n")

# === MAIN ===
def main():
    print(f"🚀 Ottimizzazione database: {DB_PATH}\n")
    backup_db()
    conn = connect_db()

    # Rimozione colonne inutili
    #drop_columns(conn, "matches", ["last_update"])
    #drop_columns(conn, "teams", ["last_update"])
    #drop_columns(conn, "players", ["position"])
    #drop_columns(conn, "injuries", ["since", "expected_return"])
    #drop_columns(conn, "odds", ["timestamp", "source"])

    # Indici + viste
    #create_indexes(conn)
    create_views(conn)

    conn.close()
    print("🎯 Database ottimizzato e pronto all’uso!")

if __name__ == "__main__":
    main()
