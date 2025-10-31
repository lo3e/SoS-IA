import sqlite3
import pandas as pd
from pathlib import Path

# 📁 Percorso DB — aggiorna se necessario
DB_PATH = Path(r"C:\Users\brain\Documents\ProgettiPersonali\SoS-IA\DB\sos_ia.db")

def build_match_features(conn):
    print("🧠 Costruzione tabella match_features_train...")

    # 1️⃣ Estrai tutti i match completati (FT)
    matches = pd.read_sql_query("""
        SELECT
            m.match_id, m.season, m.round, m.date,
            m.home_team_id, m.away_team_id,
            m.home_team_name, m.away_team_name,
            m.home_goals, m.away_goals,
            m.status,
            p.prob_home, p.prob_draw, p.prob_away
        FROM matches m
        LEFT JOIN predictions p ON m.match_id = p.match_id
        WHERE m.status = 'FT'
    """, conn)

    if matches.empty:
        print("⚠️ Nessuna partita completata trovata.")
        return

    # 2️⃣ Crea etichetta risultato (1 / X / 2)
    matches["label_result"] = matches.apply(
        lambda x: 1 if x["home_goals"] > x["away_goals"]
        else (-1 if x["away_goals"] > x["home_goals"] else 0),
        axis=1
    )

    # 3️⃣ Statistiche medie per squadra
    goals_df = pd.read_sql_query("""
        SELECT
            season,
            team_id,
            AVG(CASE WHEN home_team_id = team_id THEN home_goals ELSE away_goals END) AS avg_goals_for,
            AVG(CASE WHEN home_team_id = team_id THEN away_goals ELSE home_goals END) AS avg_goals_against
        FROM (
            SELECT m.*, t.team_id
            FROM matches m
            JOIN teams t ON m.home_team_id = t.team_id OR m.away_team_id = t.team_id
            WHERE m.status = 'FT'
        )
        GROUP BY season, team_id
    """, conn)

    # 4️⃣ Infortuni per squadra (current_unavailable_players)
    injuries = pd.read_sql_query("""
        SELECT team_name, COUNT(DISTINCT player_name) AS num_injuries
        FROM current_unavailable_players
        GROUP BY team_name
    """, conn)

    # 4️⃣ BIS - Impatto infortuni ponderato (injury_impact)
    injury_impact = pd.read_sql_query("""
        SELECT
            i.team_name,
            SUM(COALESCE(p.performance_index, 0)) AS injury_impact
        FROM current_unavailable_players i
        LEFT JOIN player_form_ranking p
            ON i.player_name = p.player_name
        GROUP BY i.team_name
    """, conn)

    # 5️⃣ Quote (solo 2025)
    odds = pd.read_sql_query("""
        SELECT
            o.match_id,
            MAX(CASE WHEN o.outcome = 'Home' THEN o.odd END) AS odd_home,
            MAX(CASE WHEN o.outcome = 'Draw' THEN o.odd END) AS odd_draw,
            MAX(CASE WHEN o.outcome = 'Away' THEN o.odd END) AS odd_away
        FROM odds o
        INNER JOIN matches m ON o.match_id = m.match_id
        WHERE o.market = 'Match Winner' AND m.season = 2025
        GROUP BY o.match_id
    """, conn)

    # 6️⃣ Merge completo
    df = (matches
          .merge(odds, on="match_id", how="left")
          .merge(goals_df.rename(columns={
              "team_id": "home_team_id",
              "avg_goals_for": "home_avg_goals_for",
              "avg_goals_against": "home_avg_goals_against"
          }), on=["season", "home_team_id"], how="left")
          .merge(goals_df.rename(columns={
              "team_id": "away_team_id",
              "avg_goals_for": "away_avg_goals_for",
              "avg_goals_against": "away_avg_goals_against"
          }), on=["season", "away_team_id"], how="left")
          .merge(injuries.rename(columns={
              "team_name": "home_team_name",
              "num_injuries": "inj_home"
          }), on="home_team_name", how="left")
          .merge(injuries.rename(columns={
              "team_name": "away_team_name",
              "num_injuries": "inj_away"
          }), on="away_team_name", how="left")
          .merge(injury_impact.rename(columns={
              "team_name": "home_team_name",
              "injury_impact": "injury_impact_home"
          }), on="home_team_name", how="left")
          .merge(injury_impact.rename(columns={
              "team_name": "away_team_name",
              "injury_impact": "injury_impact_away"
          }), on="away_team_name", how="left")
    )

    # 7️⃣ Pulizia finale
    df.fillna({
        "odd_home": 0,
        "odd_draw": 0,
        "odd_away": 0,
        "inj_home": 0,
        "inj_away": 0,
        "home_avg_goals_for": 0,
        "away_avg_goals_for": 0,
        "home_avg_goals_against": 0,
        "away_avg_goals_against": 0,
        "injury_impact_home": 0,
        "injury_impact_away": 0
    }, inplace=True)

    # 8️⃣ Salva o aggiorna la tabella
    df.to_sql("match_features_train", conn, if_exists="replace", index=False)

    print(f"✅ Tabella 'match_features_train' creata con successo ({len(df)} righe).")
    print(f"   Periodo: {df['season'].min()} → {df['season'].max()}")
    print(f"   Colonne: {len(df.columns)}")

def main():
    conn = sqlite3.connect(DB_PATH)
    build_match_features(conn)
    conn.close()

if __name__ == "__main__":
    main()
