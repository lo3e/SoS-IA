import os
import sqlite3
import argparse
import statistics
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter, defaultdict

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sosia.db"

def connect_db():
    return sqlite3.connect(DB_PATH)

def fetch_df(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return cols, rows

def pct(x, total):
    if total == 0:
        return 0.0
    return round((x / total) * 100.0, 2)

def try_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except:
        return None

def summarize_players(conn, season):
    # players per match
    q = """
    SELECT p.match_id,
           COUNT(*) as n_players
    FROM players p
    JOIN matches m ON m.match_id = p.match_id
    WHERE m.season = ?
    GROUP BY p.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    players_per_match = [r[1] for r in rows]

    # % partite con dati giocatori
    total_matches = get_total_matches(conn, season)
    matches_with_players = len(rows)

    # rating coverage
    q_rating = """
    SELECT p.rating
    FROM players p
    JOIN matches m ON m.match_id = p.match_id
    WHERE m.season = ?
    """
    _, rating_rows = fetch_df(conn, q_rating, (season,))
    ratings = [try_float(r[0]) for r in rating_rows if try_float(r[0]) is not None]

    result = {
        "matches_with_players": matches_with_players,
        "players_coverage_pct": pct(matches_with_players, total_matches),
        "avg_players_per_match": round(statistics.mean(players_per_match), 2) if players_per_match else 0,
        "median_players_per_match": statistics.median(players_per_match) if players_per_match else 0,
        "player_rating_count": len(ratings),
        "avg_rating": round(statistics.mean(ratings), 2) if ratings else None,
        "median_rating": round(statistics.median(ratings), 2) if ratings else None,
    }
    return result

def summarize_team_stats(conn, season):
    # copertura stats (quante partite hanno almeno 1 stat salvata)
    q = """
    SELECT ts.match_id, COUNT(*) as n_stats
    FROM team_stats ts
    JOIN matches m ON m.match_id = ts.match_id
    WHERE m.season = ?
    GROUP BY ts.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    matches_with_stats = len(rows)
    total_matches = get_total_matches(conn, season)

    # quali tipi di stat abbiamo più spesso?
    q_types = """
    SELECT stat_type
    FROM team_stats ts
    JOIN matches m ON m.match_id = ts.match_id
    WHERE m.season = ?
    """
    _, rows2 = fetch_df(conn, q_types, (season,))
    type_counter = Counter([r[0] for r in rows2])

    # alcune metriche chiave (tiri totali, possesso, cartellini)
    # estraiamo valori numerici puliti da stat_value (che spesso è stringa tipo "62%")
    q_details = """
    SELECT ts.match_id,
           ts.team_name,
           ts.stat_type,
           ts.stat_value
    FROM team_stats ts
    JOIN matches m ON m.match_id = ts.match_id
    WHERE m.season = ?
    """
    _, rows3 = fetch_df(conn, q_details, (season,))

    shots_per_team = []
    poss_per_team = []
    yellows_per_team = []
    reds_per_team = []

    for match_id, team_name, stype, sval in rows3:
        # normalizza percentuali tipo "62%" -> 62.0
        if isinstance(sval, str) and sval.endswith("%"):
            try:
                sval_num = float(sval.strip("%"))
            except:
                sval_num = None
        else:
            try:
                sval_num = float(sval)
            except:
                sval_num = None

        if stype in ("Total Shots", "Shots total", "Total shots", "Shots Total", "Shots Total "):
            if sval_num is not None:
                shots_per_team.append(sval_num)

        if stype in ("Ball Possession", "Possession", "Possession %"):
            if sval_num is not None:
                poss_per_team.append(sval_num)

        if stype in ("Yellow Cards", "Yellow Card", "Yellow cards"):
            if sval_num is not None:
                yellows_per_team.append(sval_num)

        if stype in ("Red Cards", "Red Card", "Red cards"):
            if sval_num is not None:
                reds_per_team.append(sval_num)

    result = {
        "matches_with_team_stats": matches_with_stats,
        "team_stats_coverage_pct": pct(matches_with_stats, total_matches),
        "top_stat_types": type_counter.most_common(10),
        "avg_shots_per_team": round(statistics.mean(shots_per_team), 2) if shots_per_team else None,
        "avg_possession_pct_per_team": round(statistics.mean(poss_per_team), 2) if poss_per_team else None,
        "avg_yellows_per_team": round(statistics.mean(yellows_per_team), 2) if yellows_per_team else None,
        "avg_reds_per_team": round(statistics.mean(reds_per_team), 2) if reds_per_team else None,
    }
    return result

def summarize_lineups(conn, season):
    # partite con almeno una lineup inserita
    q = """
    SELECT l.match_id, COUNT(*) as n_rows
    FROM lineups l
    JOIN matches m ON m.match_id = l.match_id
    WHERE m.season = ?
    GROUP BY l.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    total_matches = get_total_matches(conn, season)
    matches_with_lineup = len(rows)

    # titolari medi per squadra
    q_starters = """
    SELECT l.match_id, l.team_id, COUNT(*) as n_starters
    FROM lineups l
    JOIN matches m ON m.match_id = l.match_id
    WHERE m.season = ? AND l.is_starter = 1
    GROUP BY l.match_id, l.team_id
    """
    _, rows2 = fetch_df(conn, q_starters, (season,))
    starters_per_team_match = [r[2] for r in rows2]

    result = {
        "matches_with_lineup": matches_with_lineup,
        "lineup_coverage_pct": pct(matches_with_lineup, total_matches),
        "avg_starters_per_team": round(statistics.mean(starters_per_team_match), 2) if starters_per_team_match else None,
        "median_starters_per_team": statistics.median(starters_per_team_match) if starters_per_team_match else None,
    }
    return result

def summarize_events(conn, season):
    # quante partite hanno eventi
    q = """
    SELECT e.match_id, COUNT(*) as n_events
    FROM events e
    JOIN matches m ON m.match_id = e.match_id
    WHERE m.season = ?
    GROUP BY e.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    total_matches = get_total_matches(conn, season)
    matches_with_events = len(rows)

    # tipo di eventi più comune (Goal, Card, Substitution...)
    q_types = """
    SELECT e.type, e.detail
    FROM events e
    JOIN matches m ON m.match_id = e.match_id
    WHERE m.season = ?
    """
    _, rows2 = fetch_df(conn, q_types, (season,))
    type_counter = Counter([r[0] for r in rows2])
    detail_counter = Counter([r[1] for r in rows2 if r[1] is not None])

    # media eventi / partita
    events_per_match = [r[1] for r in rows]
    result = {
        "matches_with_events": matches_with_events,
        "events_coverage_pct": pct(matches_with_events, total_matches),
        "avg_events_per_match": round(statistics.mean(events_per_match), 2) if events_per_match else None,
        "top_event_types": type_counter.most_common(10),
        "top_event_details": detail_counter.most_common(10),
    }
    return result

def summarize_predictions(conn, season):
    # quante partite della stagione hanno prediction salvata
    q = """
    SELECT match_id, prob_home, prob_draw, prob_away
    FROM predictions
    WHERE match_id IN (
        SELECT match_id FROM matches WHERE season = ?
    )
    """
    _, rows = fetch_df(conn, q, (season,))
    total_matches = get_total_matches(conn, season)

    matches_with_pred = len(rows)

    # controllo coerenza: sum(probabilities) ~ 100
    sums = []
    for _, ph, pd, pa in rows:
        # le percentuali arrivano come stringhe "45%" dall'API?
        # se sì le normalizziamo
        def norm(x):
            if x is None:
                return None
            if isinstance(x, str) and x.endswith("%"):
                try:
                    return float(x.strip("%"))
                except:
                    return None
            try:
                return float(x)
            except:
                return None

        phn = norm(ph)
        pdn = norm(pd)
        pan = norm(pa)

        if phn is not None and pdn is not None and pan is not None:
            sums.append(phn + pdn + pan)

    result = {
        "matches_with_predictions": matches_with_pred,
        "predictions_coverage_pct": pct(matches_with_pred, total_matches),
        "avg_sum_probs_pct": round(statistics.mean(sums), 2) if sums else None,
        "median_sum_probs_pct": round(statistics.median(sums), 2) if sums else None,
    }
    return result

def summarize_head2head(conn, season):
    # Quante coppie squadra-squadra hanno almeno un record h2h?
    q = """
    SELECT home_team_id, away_team_id, COUNT(*) as n_rows
    FROM head2head
    WHERE season = ?
    GROUP BY home_team_id, away_team_id
    """
    _, rows = fetch_df(conn, q, (season,))
    pairs = len(rows)
    density = [r[2] for r in rows]  # numero partite storiche viste per quella coppia

    result = {
        "distinct_pairs_with_h2h": pairs,
        "avg_h2h_matches_per_pair": round(statistics.mean(density), 2) if density else None,
        "median_h2h_matches_per_pair": statistics.median(density) if density else None,
    }
    return result

def summarize_injuries(conn, season):
    # Verifichiamo se abbiamo injury report per questa stagione
    q = """
    SELECT i.match_id, COUNT(*) as n_inj
    FROM injuries i
    JOIN matches m ON m.match_id = i.match_id
    WHERE m.season = ?
    GROUP BY i.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    total_matches = get_total_matches(conn, season)

    matches_with_inj = len(rows)
    inj_per_match = [r[1] for r in rows]

    result = {
        "matches_with_injuries": matches_with_inj,
        "injury_coverage_pct": pct(matches_with_inj, total_matches),
        "avg_injuries_per_match": round(statistics.mean(inj_per_match), 2) if inj_per_match else None,
        "median_injuries_per_match": statistics.median(inj_per_match) if inj_per_match else None,
    }
    return result

def summarize_odds(conn, season):
    # quante fixture hanno odds
    q = """
    SELECT o.match_id,
           COUNT(*) as n_odds_rows
    FROM odds o
    JOIN matches m ON m.match_id = o.match_id
    WHERE m.season = ?
    GROUP BY o.match_id
    """
    _, rows = fetch_df(conn, q, (season,))
    total_matches = get_total_matches(conn, season)

    matches_with_odds = len(rows)
    odds_rows_per_match = [r[1] for r in rows]

    # quali bookmaker vediamo più spesso
    q_bm = """
    SELECT bookmaker_name, COUNT(*) as n_rows
    FROM odds o
    JOIN matches m ON m.match_id = o.match_id
    WHERE m.season = ?
    GROUP BY bookmaker_name
    ORDER BY n_rows DESC
    LIMIT 10
    """
    _, rows2 = fetch_df(conn, q_bm, (season,))
    top_bookmakers = [(r[0], r[1]) for r in rows2]

    result = {
        "matches_with_odds": matches_with_odds,
        "odds_coverage_pct": pct(matches_with_odds, total_matches),
        "avg_odds_rows_per_match": round(statistics.mean(odds_rows_per_match), 2) if odds_rows_per_match else None,
        "median_odds_rows_per_match": statistics.median(odds_rows_per_match) if odds_rows_per_match else None,
        "top_bookmakers": top_bookmakers
    }
    return result

def get_total_matches(conn, season):
    q = "SELECT COUNT(*) FROM matches WHERE season = ?"
    _, rows = fetch_df(conn, q, (season,))
    return rows[0][0] if rows and rows[0] else 0

def summarize_matches_basic(conn, season):
    # controlliamo quante partite giocate (FT), gol medi, ecc.
    q = """
    SELECT status, home_goals, away_goals
    FROM matches
    WHERE season = ?
    """
    _, rows = fetch_df(conn, q, (season,))

    total = len(rows)
    finals = [r for r in rows if r[0] == "FT"]
    n_final = len(finals)

    goals_per_match = []
    for status, hg, ag in finals:
        if hg is not None and ag is not None:
            goals_per_match.append((hg or 0) + (ag or 0))

    result = {
        "total_matches_in_db": total,
        "final_matches": n_final,
        "pct_final_status": pct(n_final, total),
        "avg_goals_per_match": round(statistics.mean(goals_per_match), 2) if goals_per_match else None,
        "median_goals_per_match": statistics.median(goals_per_match) if goals_per_match else None,
    }
    return result

def audit_season(season: int):
    conn = connect_db()

    print(f"\n📊 ANALISI STAGIONE {season}\n" + "="*60)

    matches_info = summarize_matches_basic(conn, season)
    print("📅 Partite")
    for k,v in matches_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    players_info = summarize_players(conn, season)
    print("🧍‍♂️ Giocatori")
    for k,v in players_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    team_stats_info = summarize_team_stats(conn, season)
    print("📊 Statistiche di squadra (team_stats)")
    for k,v in team_stats_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    lineups_info = summarize_lineups(conn, season)
    print("📋 Formazioni (lineups)")
    for k,v in lineups_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    events_info = summarize_events(conn, season)
    print("⏱ Eventi partita (events)")
    for k,v in events_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    pred_info = summarize_predictions(conn, season)
    print("🔮 Predictions API")
    for k,v in pred_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    h2h_info = summarize_head2head(conn, season)
    print("⚔️ Head2Head")
    for k,v in h2h_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    inj_info = summarize_injuries(conn, season)
    print("💊 Infortuni")
    for k,v in inj_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    odds_info = summarize_odds(conn, season)
    print("💰 Quote (odds)")
    for k,v in odds_info.items():
        print(f"  {k}: {v}")
    print("-"*60)

    print("✅ Audit completato.\n")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit qualità/completessa dati per una stagione")
    parser.add_argument("--season", type=int, required=True, help="Anno stagione es. 2021")
    args = parser.parse_args()

    audit_season(args.season)
