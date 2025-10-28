# src/backtest_historical.py
import sqlite3
import pandas as pd
import numpy as np
from math import exp, factorial
from db import DB_PATH

# Parametri
N_MATCHES_WINDOW = 20    # rolling window per team_strengths
CORE_BANKROLL = 450.0
FLAT_PERCENT = 0.02
FLAT_STAKE = CORE_BANKROLL * FLAT_PERCENT
MAX_GOALS = 8
VALUE_THRESHOLD = 0.03   # Value% minimo per scommettere (3%)

# --- funzioni ---
def poisson_pmf(lmbda, k):
    return (lmbda**k * exp(-lmbda)) / factorial(k)

def calc_team_strengths(df_hist):
    avg_home_goals = df_hist['home_goals'].mean()
    avg_away_goals = df_hist['away_goals'].mean()
    
    teams = sorted(set(df_hist['home']).union(set(df_hist['away'])))
    rec = []
    for t in teams:
        mask = (df_hist['home'] == t) | (df_hist['away'] == t)
        hist = df_hist[mask].sort_values('date', ascending=False).head(N_MATCHES_WINDOW)
        home_hist = hist[hist['home'] == t]
        away_hist = hist[hist['away'] == t]

        home_scored = home_hist['home_goals'].mean() if len(home_hist) > 0 else avg_home_goals
        away_scored = away_hist['away_goals'].mean() if len(away_hist) > 0 else avg_away_goals
        home_conceded = home_hist['away_goals'].mean() if len(home_hist) > 0 else avg_away_goals
        away_conceded = away_hist['home_goals'].mean() if len(away_hist) > 0 else avg_home_goals

        rec.append({
            'team': t,
            'home_scored': home_scored,
            'away_scored': away_scored,
            'home_conceded': home_conceded,
            'away_conceded': away_conceded
        })
    
    ts = pd.DataFrame(rec).set_index('team')
    ts['attack_home'] = ts['home_scored'] / avg_home_goals
    ts['attack_away'] = ts['away_scored'] / avg_away_goals
    ts['defense_home'] = ts['home_conceded'] / avg_away_goals
    ts['defense_away'] = ts['away_conceded'] / avg_home_goals
    ts.fillna(1.0, inplace=True)
    return ts, avg_home_goals, avg_away_goals

def expected_goals(avg_home, avg_away, a_h, d_a, a_a, d_h):
    home_xg = avg_home * a_h * d_a
    away_xg = avg_away * a_a * d_h
    return home_xg, away_xg

def probs_from_xg(home_xg, away_xg):
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS+1):
        for j in range(MAX_GOALS+1):
            p = poisson_pmf(home_xg, i) * poisson_pmf(away_xg, j)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def implied_prob(odds):
    return 1.0/odds if odds and odds>0 else None

# --- main backtest ---
def run_backtest():
    conn = sqlite3.connect(DB_PATH)
    
    # Carica partite
    df_matches = pd.read_sql_query(
        "SELECT * FROM matches WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL ORDER BY date",
        conn,
        parse_dates=['date']
    )
    # Carica quote storiche
    df_odds = pd.read_sql_query(
        "SELECT match_id, odds_1, odds_x, odds_2 FROM odds WHERE source='Bet365'",
        conn
    )
    conn.close()

    # Merge partite + quote
    df = pd.merge(df_matches, df_odds, on='match_id', how='left')

    bankroll = CORE_BANKROLL
    total_staked = 0.0
    total_profit = 0.0
    bets = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        if idx == 0:
            continue
        df_hist = df.iloc[:idx]

        ts, avg_h, avg_a = calc_team_strengths(df_hist)
        home = row['home']
        away = row['away']
        if home not in ts.index or away not in ts.index:
            continue
        h = ts.loc[home]
        a = ts.loc[away]

        home_xg, away_xg = expected_goals(
            avg_h, avg_a,
            h['attack_home'], a['defense_away'],
            a['attack_away'], h['defense_home']
        )
        p1, px, p2 = probs_from_xg(home_xg, away_xg)

        # quote corrette dal merge
        odds_1 = float(row.get('odds_1') or 0)
        odds_X = float(row.get('odds_x') or 0)
        odds_2 = float(row.get('odds_2') or 0)
        odds_list = [odds_1, odds_X, odds_2]

        probs = [p1, px, p2]
        picks = ['1','X','2']

        # selezione value bet
        evs = [(p*o - 1, pick, p, o) for p, o, pick in zip(probs, odds_list, picks) if p and o > 0]
        value_bets = [ev for ev in evs if ev[0] > 0 and (ev[2] - 1/ev[3]) / (1/ev[3]) > VALUE_THRESHOLD]

        if not value_bets:
            continue

        ev_val, pick, p_model, o = max(value_bets, key=lambda x: x[0])
        stake = FLAT_STAKE
        total_staked += stake

        outcome = '1' if row['home_goals'] > row['away_goals'] else 'X' if row['home_goals'] == row['away_goals'] else '2'
        profit = stake*(o-1) if outcome==pick else -stake
        bankroll += profit
        total_profit += profit

        bets.append({
            'date': row['date'], 'home': home, 'away': away,
            'pick': pick, 'odds': o, 'stake': stake,
            'profit': profit, 'bankroll': bankroll
        })

    roi = (total_profit/total_staked*100) if total_staked>0 else 0
    print("Backtest completato")
    print("Total staked:", round(total_staked,2))
    print("Total profit:", round(total_profit,2))
    print("ROI:", round(roi,2),"%")
    print("Final bankroll:", round(bankroll,2))

    pd.DataFrame(bets).to_csv("../data/backtest_historical_report.csv", index=False)
    print("Report salvato in data/backtest_historical_report.csv")

if __name__=="__main__":
    run_backtest()
