# src/pipeline/optimize_db.py
"""
Ottimizzazione DB: indici + viste
"""

from src.core.config import DB_PATH
from src.core.logger import get_logger
import sqlite3

logger = get_logger(__name__)


def connect_db():
    return sqlite3.connect(DB_PATH)


def create_indexes(conn):
    logger.info("⚙️ Aggiorno indici principali...")
    cur = conn.cursor()
    indexes = [
        ("idx_matches_match_id", "matches", "match_id"),
        ("idx_players_match_id", "players", "match_id"),
        ("idx_team_stats_match_id", "team_stats", "match_id"),
        ("idx_events_match_id", "events", "match_id"),
        ("idx_predictions_match_id", "predictions", "match_id"),
        ("idx_injuries_match_id", "injuries", "match_id"),
        ("idx_odds_match_id", "odds", "match_id"),
        ("idx_standings_season_team", "standings", "season, team_id"),
    ]
    for name, table, cols in indexes:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols});")
    conn.commit()
    logger.info("✅ Indici aggiornati.")


def create_views(conn):
    logger.info("🧩 Ricreo viste logiche...")
    cur = conn.cursor()
    # qui ho incollato la tua execscript pari pari (:contentReference[oaicite:3]{index=3})
    cur.executescript("""
    DROP VIEW IF EXISTS match_features_view;
    DROP VIEW IF EXISTS next_fixtures_view;
    DROP VIEW IF EXISTS injury_summary_view;
    DROP VIEW IF EXISTS standings_view;
    DROP VIEW IF EXISTS player_positions_view;
    DROP VIEW IF EXISTS player_stats_clean_view;
    DROP VIEW IF EXISTS player_form_ranking;

    CREATE VIEW match_features_view AS
    SELECT
        m.match_id, m.season, m.round,
        m.home_team_id, m.away_team_id,
        m.home_goals, m.away_goals,
        p.winner, p.prob_home, p.prob_draw, p.prob_away,
        sh.rank AS home_rank, sa.rank AS away_rank,
        sh.points AS home_points, sa.points AS away_points
    FROM matches m
    LEFT JOIN predictions p ON m.match_id = p.match_id
    LEFT JOIN standings sh ON sh.team_id = m.home_team_id AND sh.season = m.season
    LEFT JOIN standings sa ON sa.team_id = m.away_team_id AND sa.season = m.season;

    CREATE VIEW next_fixtures_view AS
    SELECT
        m.match_id, m.date, m.round,
        m.home_team_name, m.away_team_name,
        p.winner, p.prob_home, p.prob_draw, p.prob_away,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Home' THEN o.odd END), 2) AS avg_home_odd,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Draw' THEN o.odd END), 2) AS avg_draw_odd,
        ROUND(AVG(CASE WHEN o.market = 'Match Winner' AND o.outcome = 'Away' THEN o.odd END), 2) AS avg_away_odd,
        COUNT(DISTINCT o.bookmaker_name) AS bookmakers_count
    FROM matches m
    LEFT JOIN predictions p ON m.match_id = p.match_id
    LEFT JOIN odds o ON m.match_id = o.match_id
    WHERE m.status IN ('NS', 'TBD', 'PST') AND date(m.date) <= date('now', '+14 days')
    GROUP BY m.match_id
    ORDER BY m.date ASC;

    CREATE VIEW injury_summary_view AS
    SELECT
        m.season, m.round, t.name AS team_name,
        COUNT(i.id) AS total_injuries
    FROM injuries i
    JOIN matches m ON i.match_id = m.match_id
    JOIN teams t ON i.team_id = t.team_id
    GROUP BY m.season, m.round, team_name;

    CREATE VIEW standings_view AS
    SELECT
        season, league_id, rank, team_name, points,
        goals_for, goals_against, goals_diff, form
    FROM standings
    ORDER BY season DESC, rank ASC;

    CREATE VIEW player_positions_view AS
    SELECT p.*, l.position
    FROM players p
    LEFT JOIN lineups l ON p.player_id = l.player_id AND p.match_id = l.match_id;

    CREATE VIEW player_stats_clean_view AS
    SELECT
        player_id, match_id, team_id, player_name,
        COALESCE(minutes, 0) AS minutes, COALESCE(rating, 0) AS rating,
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
                      
    CREATE VIEW player_form_ranking AS
    WITH player_stats AS (
        SELECT
            p.player_id,
            p.player_name,
            l.team_name,
            l.position,
            m.season,
            COUNT(DISTINCT p.match_id) AS matches_played,
            ROUND(AVG(p.rating), 2) AS avg_rating,
            SUM(COALESCE(p.goals_total, 0)) AS goals,
            SUM(COALESCE(p.assists, 0)) AS assists,
            SUM(COALESCE(p.yellow_cards, 0)) AS yellows,
            SUM(COALESCE(p.red_cards, 0)) AS reds
        FROM players p
        JOIN matches m ON p.match_id = m.match_id
        JOIN lineups l ON p.match_id = l.match_id AND p.player_id = l.player_id
        WHERE m.status = 'FT' AND m.season = 2025 AND p.rating IS NOT NULL
        GROUP BY p.player_id, p.player_name, l.team_name, l.position, m.season
    ),
    recent_form AS (
        SELECT
            p.player_id,
            ROUND(AVG(p.rating), 2) AS recent_avg
        FROM players p
        JOIN matches m ON p.match_id = m.match_id
        WHERE m.status = 'FT' AND m.season = 2025 AND p.rating IS NOT NULL
        GROUP BY p.player_id
        HAVING COUNT(p.match_id) >= 3
    )
    SELECT
        ps.player_id,
        ps.player_name,
        ps.team_name,
        ps.position,
        ps.season,
        ps.matches_played,
        ps.avg_rating,
        ps.goals,
        ps.assists,
        ps.yellows,
        ps.reds,
        ROUND(COALESCE(rf.recent_avg - ps.avg_rating, 0), 2) AS form_trend,
        CASE
            WHEN ps.position = 'G' THEN ROUND((ps.avg_rating * 0.8) - (ps.reds * 0.05) - (ps.yellows * 0.02), 2)
            WHEN ps.position = 'D' THEN ROUND((ps.avg_rating * 0.7) + (ps.goals * 0.1) + (ps.assists * 0.05) - (ps.reds * 0.05) - (ps.yellows * 0.02), 2)
            WHEN ps.position = 'M' THEN ROUND((ps.avg_rating * 0.6) + (ps.goals * 0.15) + (ps.assists * 0.15) - (ps.reds * 0.05) - (ps.yellows * 0.02), 2)
            WHEN ps.position = 'F' THEN ROUND((ps.avg_rating * 0.5) + (ps.goals * 0.25) + (ps.assists * 0.15) - (ps.reds * 0.05) - (ps.yellows * 0.02), 2)
            ELSE ROUND((ps.avg_rating * 0.7) + (ps.goals * 0.1) + (ps.assists * 0.1), 2)
        END AS performance_index
    FROM player_stats ps
    LEFT JOIN recent_form rf ON ps.player_id = rf.player_id
    WHERE ps.matches_played >= 2
    ORDER BY performance_index DESC;
    """)

    conn.commit()
    logger.info("✅ Viste aggiornate.")


def optimize_database():
    logger.info("🔧 Ottimizzazione leggera post-update...")
    conn = connect_db()
    create_indexes(conn)
    create_views(conn)
    conn.close()
    logger.info("🎯 Ottimizzazione completata e DB aggiornato.")


if __name__ == "__main__":
    optimize_database()
