# src/model_poisson.py
import sqlite3
import pandas as pd
import numpy as np
from math import exp, factorial
from db import DB_PATH
from datetime import datetime

N_MATCHES_WINDOW = 5   # usa le ultime 20 partite per le stime (configurabile)
MAX_GOALS = 8

def poisson_pmf(lmbda, k):
    return (lmbda**k * exp(-lmbda)) / factorial(k)

def expected_goals(avg_home, avg_away, a_h, d_a, a_a, d_h):
    home_xg = avg_home * a_h * d_a
    away_xg = avg_away * a_a * d_h
    return home_xg, away_xg

def calculate_team_strengths_recent(df_matches, window=N_MATCHES_WINDOW):
    """Calcola attack/defense usando solo ultime window partite per ciascuna squadra."""
    # ordina per data
    df = df_matches.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # costruiamo serie storiche per ogni squadra (home e away) e prendiamo ultime N eventi per squadra
    teams = sorted(set(df['home']).union(set(df['away'])))
    rec = []

    # medie globali ma calcolate su tutto il set (o potresti limitarle al window globale)
    avg_home_goals = df['home_goals'].mean()
    avg_away_goals = df['away_goals'].mean()

    for t in teams:
        # estrai ultime N match in cui t è coinvolta (home o away)
        mask = (df['home'] == t) | (df['away'] == t)
        hist = df[mask].sort_values('date', ascending=False).head(window)

        # calcola medie separate home/away su quelle N
        home_hist = hist[hist['home'] == t]
        away_hist = hist[hist['away'] == t]

        home_scored = home_hist['home_goals'].mean() if len(home_hist) > 0 else np.nan
        away_scored = away_hist['away_goals'].mean() if len(away_hist) > 0 else np.nan

        home_conceded = home_hist['away_goals'].mean() if len(home_hist) > 0 else np.nan
        away_conceded = away_hist['home_goals'].mean() if len(away_hist) > 0 else np.nan

        rec.append({
            'team': t,
            'home_scored': home_scored,
            'away_scored': away_scored,
            'home_conceded': home_conceded,
            'away_conceded': away_conceded
        })

    ts = pd.DataFrame(rec).set_index('team')

    # sostituisci nan con medie di campionato per stabilità
    ts.loc[:, 'home_scored'] = ts['home_scored'].fillna(avg_home_goals)
    ts.loc[:, 'away_scored'] = ts['away_scored'].fillna(avg_away_goals)
    ts.loc[:, 'home_conceded'] = ts['home_conceded'].fillna(avg_away_goals)
    ts.loc[:, 'away_conceded'] = ts['away_conceded'].fillna(avg_home_goals)

    # normalizza in attack/defense
    ts['attack_home'] = ts['home_scored'] / avg_home_goals
    ts['attack_away'] = ts['away_scored'] / avg_away_goals
    ts['defense_home'] = ts['home_conceded'] / avg_away_goals
    ts['defense_away'] = ts['away_conceded'] / avg_home_goals

    ts.fillna(1.0, inplace=True)

    return ts, avg_home_goals, avg_away_goals

def probs_from_xg(home_xg, away_xg, max_goals=MAX_GOALS):
    p_home = p_draw = p_away = 0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            p = poisson_pmf(home_xg, i) * poisson_pmf(away_xg, j)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    # normalizza per sicurezza
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def save_model_probs_to_db(conn, match_id, p1, px, p2, home_xg, away_xg):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    # inserisce nuovo record in model_probs (non duplica: multiple timestamp ok)
    cur.execute("""
        INSERT INTO model_probs (match_id, p_home, p_draw, p_away, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (match_id, float(p1), float(px), float(p2), ts))
    conn.commit()

def run_and_store_all():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn, parse_dates=['date'])
    # consideriamo matches che hanno risultato per training
    df_hist = df[df['home_goals'].notnull() & df['away_goals'].notnull()].copy()
    team_strengths, avg_home_goals, avg_away_goals = calculate_team_strengths_recent(df_hist, window=N_MATCHES_WINDOW)

    # prendi matches futuri (senza risultato) per cui vogliamo predire
    df_future = df[df['home_goals'].isnull() | df['away_goals'].isnull()].copy()
    print(f"🔎 Calcolando prob. per {len(df_future)} match futuri")

    for _, r in df_future.iterrows():
        home = r['home']
        away = r['away']
        match_id = r['match_id']
        if home not in team_strengths.index or away not in team_strengths.index:
            continue
        h = team_strengths.loc[home]
        a = team_strengths.loc[away]
        home_xg, away_xg = expected_goals(avg_home_goals, avg_away_goals,
                                          h['attack_home'], a['defense_away'],
                                          a['attack_away'], h['defense_home'])
        p1, px, p2 = probs_from_xg(home_xg, away_xg)
        save_model_probs_to_db(conn, match_id, p1, px, p2, home_xg, away_xg)

    conn.close()
    print("✅ Modellazione completata e probabilità salvate in model_probs.")

if __name__ == "__main__":
    run_and_store_all()
