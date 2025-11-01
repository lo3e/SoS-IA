import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv

# =============== CONFIG ===============
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH")

MODEL_PATH = Path(DATA_PATH) / "rf_match_predictor_v3.pkl"
META_PATH = Path(DATA_PATH) / "rf_match_predictor_v3.json"

# margine minimo per dire "questa è value"
VALUE_MARGIN = 0.03  # 3% sopra l'implicita
TOP_N = 10            # quante partite stampare ordinate per EV


# ---------- helper DB ----------
def get_conn():
    return sqlite3.connect(DB_PATH)


# ---------- 1) qual è il prossimo round? ----------
def get_next_round(conn):
    c = conn.cursor()
    c.execute("""
        SELECT MIN(round) 
        FROM matches
        WHERE status = 'NS'
    """)
    row = c.fetchone()
    return row[0] if row and row[0] else None


# ---------- 2) injuries agg per team (come nel training) ----------
def get_injuries_features(conn):
    # stessa logica del training: current_unavailable_players + player_form
    injuries = pd.read_sql_query("""
        SELECT
            i.team_name AS team_name,
            COUNT(DISTINCT i.player_name) AS num_injuries,
            ROUND(SUM(COALESCE(p.performance_index, 0)), 2) AS injury_impact
        FROM current_unavailable_players i
        LEFT JOIN player_form_ranking p
            ON i.player_name = p.player_name AND i.season = p.season
        GROUP BY i.team_name
    """, conn)
    return injuries


# ---------- 3) standings base ----------
def get_standings(conn):
    return pd.read_sql_query("""
        SELECT
            season,
            league_id,
            team_id,
            team_name,
            rank,
            points,
            goals_for,
            goals_against
        FROM standings
    """, conn)


# ---------- 4) rolling / recent form per team (stessa logica v2) ----------
def get_recent_stats(conn):
    # prendiamo TUTTI i match FT e calcoliamo rolling per team
    matches = pd.read_sql_query("""
        SELECT
            match_id,
            season,
            date,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
            home_goals,
            away_goals
        FROM matches
        WHERE status = 'FT'
        ORDER BY date
    """, conn)

    def build_team_side(df, side):
        # side = 'home' or 'away'
        cols = {
            "team_id": f"{side}_team_id",
            "team_name": f"{side}_team_name",
            "goals_for": f"{side}_goals_for",
            "goals_against": f"{'away' if side=='home' else 'home'}_goals"
        }
        tmp = df[[
            "match_id", "date", "season",
            cols["team_id"], cols["team_name"],
            f"{side}_goals", f"{'away' if side=='home' else 'home'}_goals"
        ]].copy()
        tmp.rename(columns={
            cols["team_id"]: "team_id",
            cols["team_name"]: "team_name",
            f"{side}_goals": "goals_for",
            f"{'away' if side=='home' else 'home'}_goals": "goals_against",
        }, inplace=True)

        # punti match
        def pts(r):
            if r["goals_for"] > r["goals_against"]:
                return 3
            if r["goals_for"] == r["goals_against"]:
                return 1
            return 0

        tmp["points_match"] = tmp.apply(pts, axis=1)

        # rolling 5
        tmp = tmp.sort_values(["team_id", "date"])
        tmp["recent_points_avg"] = (
            tmp.groupby("team_id")["points_match"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        tmp["recent_goals_for_avg"] = (
            tmp.groupby("team_id")["goals_for"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        tmp["recent_goals_against_avg"] = (
            tmp.groupby("team_id")["goals_against"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        return tmp

    home_side = build_team_side(matches, "home")
    away_side = build_team_side(matches, "away")

    team_stats = pd.concat([home_side, away_side], ignore_index=True)

    # teniamo l'ultima riga per team (la più recente)
    team_latest = (
        team_stats.sort_values(["team_id", "date"])
        .groupby("team_id")
        .tail(1)
        .reset_index(drop=True)
    )
    return team_latest[[
        "team_id", "team_name",
        "recent_points_avg",
        "recent_goals_for_avg",
        "recent_goals_against_avg",
    ]]


# ---------- 5) head to head ----------
def get_h2h_features(conn):
    # facciamo media ultimi N h2h
    h2h = pd.read_sql_query("""
        SELECT
            home_team_id,
            away_team_id,
            home_goals,
            away_goals,
            season
        FROM head2head
        ORDER BY season DESC
    """, conn)

    if h2h.empty:
        return pd.DataFrame(columns=[
            "home_team_id", "away_team_id",
            "h2h_home_wins", "h2h_away_wins", "h2h_goal_diff_avg"
        ])

    def outcome(r):
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]):
            return 0
        if r["home_goals"] > r["away_goals"]:
            return 1
        if r["home_goals"] < r["away_goals"]:
            return -1
        return 0

    h2h["outcome"] = h2h.apply(outcome, axis=1)
    h2h["goal_diff"] = (h2h["home_goals"] - h2h["away_goals"]).fillna(0)

    # tieni ultimi 5 scontri
    h2h["rn"] = (
        h2h.groupby(["home_team_id", "away_team_id"]).cumcount() + 1
    )
    h2h_recent = h2h[h2h["rn"] <= 5]

    agg = h2h_recent.groupby(["home_team_id", "away_team_id"]).agg(
        h2h_home_wins=("outcome", lambda x: int((x == 1).sum())),
        h2h_away_wins=("outcome", lambda x: int((x == -1).sum())),
        h2h_goal_diff_avg=("goal_diff", "mean"),
    ).reset_index()

    return agg


# ---------- 6) odds 1X2 per match ----------
def get_odds_for_matches(conn, match_ids):
    """
    Ritorna una tabella del tipo:
    match_id | odd_home | odd_draw | odd_away

    prendendo le quote dal mercato 'Match Winner'.
    Se ci sono più bookmaker, prova a prendere Bet365,
    altrimenti prende il primo disponibile per quel match/outcome.
    """
    if not match_ids:
        return pd.DataFrame()

    # forza a int → str per sicurezza
    match_ids_clean = [int(m) for m in match_ids if pd.notna(m)]

    placeholders = ",".join("?" for _ in match_ids_clean)

    # prendiamo SOLO il mercato giusto
    # NB: abbiamo tolto timestamp perché nel tuo DB non c’è più
    query = f"""
        WITH ranked AS (
            SELECT
                match_id,
                bookmaker_name,
                market,
                outcome,
                odd,
                /* priorità: Bet365 prima, poi gli altri */
                CASE
                    WHEN bookmaker_name = 'Bet365' THEN 1
                    ELSE 2
                END AS bk_prio
            FROM odds
            WHERE match_id IN ({placeholders})
              AND market = 'Match Winner'
        )
        SELECT
            match_id,
            bookmaker_name,
            outcome,
            odd,
            bk_prio
        FROM ranked
        ORDER BY match_id, bk_prio
    """

    raw = pd.read_sql_query(query, conn, params=match_ids_clean)

    if raw.empty:
        print("⚠️ Nessuna quota trovata in odds per Match Winner.")
        return pd.DataFrame()

    # normalizziamo gli outcome in 3 soli valori
    def norm_outcome(x: str) -> str:
        x = (x or "").strip().lower()
        if x in ("home", "1", "home team", "1 (home)"):
            return "home"
        if x in ("away", "2", "away team", "2 (away)"):
            return "away"
        if x in ("draw", "x", "tie"):
            return "draw"
        # se è strano lo butto su draw così non esplode
        return "draw"

    raw["outcome_norm"] = raw["outcome"].apply(norm_outcome)

    # teniamo per ogni match+outcome SOLO la riga col bookmaker migliore (già ordinato sopra)
    raw = (
        raw.sort_values(["match_id", "bk_prio"])
           .drop_duplicates(subset=["match_id", "outcome_norm"], keep="first")
    )

    # pivot → una riga per match
    odds_pivot = raw.pivot_table(
        index="match_id",
        columns="outcome_norm",
        values="odd",
        aggfunc="first"
    ).reset_index()

    # rinominiamo come ci serve nel resto del codice
    odds_pivot = odds_pivot.rename(
        columns={
            "home": "odd_home",
            "draw": "odd_draw",
            "away": "odd_away",
        }
    )

    # per sicurezza mettiamo le colonne che ci servono sempre
    for col in ["odd_home", "odd_draw", "odd_away"]:
        if col not in odds_pivot.columns:
            odds_pivot[col] = None

    return odds_pivot

# ---------- 7) build feature per NS ----------
def build_features_for_ns(conn):
    next_round = get_next_round(conn)
    if not next_round:
        print("⚠️ Nessun round NS trovato.")
        return pd.DataFrame()

    # 1. prendo le partite non giocate di quel round
    matches = pd.read_sql_query("""
        SELECT
            match_id,
            date,
            season,
            round,
            league_id,
            status,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name
        FROM matches
        WHERE status = 'NS'
          AND round = ?
        ORDER BY date
    """, conn, params=(next_round,))

    if matches.empty:
        print("⚠️ Nessuna partita NS per il prossimo round.")
        return matches

    # 2. prendo tutto quello che ci serve
    standings = get_standings(conn)              # season, league_id, team_id, team_name, rank, points, ...
    injuries = get_injuries_features(conn)       # team_name, num_injuries, injury_impact
    recent   = get_recent_stats(conn)            # team_id, team_name, recent_points_avg, ...
    h2h      = get_h2h_features(conn)            # home_team_id, away_team_id, h2h_...

    # 3. JOIN STANDINGS (HOME)
    matches = matches.merge(
        standings.add_prefix("home_"),
        left_on=["season", "home_team_id", "league_id"],
        right_on=["home_season", "home_team_id", "home_league_id"],
        how="left",
    ).drop(columns=["home_season", "home_league_id"])

    # 4. JOIN STANDINGS (AWAY)
    matches = matches.merge(
        standings.add_prefix("away_"),
        left_on=["season", "away_team_id", "league_id"],
        right_on=["away_season", "away_team_id", "away_league_id"],
        how="left",
    ).drop(columns=["away_season", "away_league_id"])

    # dopo i due merge qui sopra è molto probabile che i nomi squadra abbiano i suffissi
    # normalizziamo ORA così non ci incasiniamo dopo
    rename_fix = {}
    if "home_team_name_x" in matches.columns:
        rename_fix["home_team_name_x"] = "home_team_name"
    if "home_team_name_y" in matches.columns:
        # lo teniamo come info "da standings"
        rename_fix["home_team_name_y"] = "home_team_name_stand"
    if "away_team_name_x" in matches.columns:
        rename_fix["away_team_name_x"] = "away_team_name"
    if "away_team_name_y" in matches.columns:
        rename_fix["away_team_name_y"] = "away_team_name_stand"
    if rename_fix:
        matches = matches.rename(columns=rename_fix)

    # 5. INJURIES (HOME)
    matches = matches.merge(
        injuries.add_prefix("home_"),
        left_on="home_team_name",     # ora esiste sicuro
        right_on="home_team_name",
        how="left",
    )

    # 6. INJURIES (AWAY)
    matches = matches.merge(
        injuries.add_prefix("away_"),
        left_on="away_team_name",
        right_on="away_team_name",
        how="left",
    )

    matches["home_num_injuries"] = matches["home_num_injuries"].fillna(0)
    matches["away_num_injuries"] = matches["away_num_injuries"].fillna(0)
    matches["home_injury_impact"] = matches["home_injury_impact"].fillna(0)
    matches["away_injury_impact"] = matches["away_injury_impact"].fillna(0)

    # 7. RECENT FORM
    # invece di fare add_prefix PRIMA del merge (che crea conflitti),
    # rinominiamo PRIMA e poi mergiamo.
    home_recent = recent.rename(columns={
        "team_id": "home_team_id",
        "recent_points_avg": "home_recent_points_avg",
        "recent_goals_for_avg": "home_recent_goals_for_avg",
        "recent_goals_against_avg": "home_recent_goals_against_avg",
    })
    matches = matches.merge(
        home_recent[[
            "home_team_id",
            "home_recent_points_avg",
            "home_recent_goals_for_avg",
            "home_recent_goals_against_avg",
        ]],
        on="home_team_id",
        how="left",
    )

    away_recent = recent.rename(columns={
        "team_id": "away_team_id",
        "recent_points_avg": "away_recent_points_avg",
        "recent_goals_for_avg": "away_recent_goals_for_avg",
        "recent_goals_against_avg": "away_recent_goals_against_avg",
    })
    matches = matches.merge(
        away_recent[[
            "away_team_id",
            "away_recent_points_avg",
            "away_recent_goals_for_avg",
            "away_recent_goals_against_avg",
        ]],
        on="away_team_id",
        how="left",
    )

    # 8. HEAD2HEAD
    matches = matches.merge(
        h2h,
        on=["home_team_id", "away_team_id"],
        how="left",
    )

    # 9. FEATURE DI DIFFERENZA
    matches["rank_diff"] = matches["home_rank"].fillna(0) - matches["away_rank"].fillna(0)
    matches["points_diff"] = matches["home_points"].fillna(0) - matches["away_points"].fillna(0)
    matches["form_diff"] = matches["home_recent_points_avg"].fillna(0) - matches["away_recent_points_avg"].fillna(0)

    # 10. ODDS (se ci sono)
    odds = get_odds_for_matches(conn, matches["match_id"].tolist())
    matches = matches.merge(odds, on="match_id", how="left")

    return matches


# ---------- 8) main ----------
def main():
    conn = get_conn()
    df = build_features_for_ns(conn)
    conn.close()

    # se non ci sono partite NS, stop
    if df.empty:
        print("⚠️ Nessuna partita NS trovata.")
        return

    # il prossimo round è quello (unico) che abbiamo appena estratto
    # la colonna si chiama proprio "round" nel DB
    if "round" in df.columns:
        next_round = df["round"].iloc[0]
    else:
        # fallback super difensivo
        next_round = "sconosciuto"

    print(f"🗓️ Prossimo round: {next_round} (partite: {len(df)})")

    # --- CARICO MODELLO ---
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_features = meta["features"]
    print(f"✅ Modello caricato con {len(train_features)} feature di training")

    # --- PREPARO X PER IL MODELLO ---
    drop_cols = [
        "match_id", "date", "season", "league_id",
        "home_team_id", "away_team_id",
        "home_team_name", "away_team_name",
        "status", "round",
        "target_1x2", "target_num",
        "home_goals", "away_goals",
        "home_points_match", "away_points_match",
        "home_goals_diff", "away_goals_diff",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # allineo le colonne a quelle del training
    for col in train_features:
        if col not in X.columns:
            X[col] = 0
    X = X[train_features]

    # --- PREDIZIONI ---
    proba = model.predict_proba(X)
    classes_ = list(model.classes_)   # es. [-1, 0, 1]

    df["p_home"] = 0.0
    df["p_draw"] = 0.0
    df["p_away"] = 0.0

    for i, cls_probs in enumerate(proba):
        prob_map = {int(c): float(cls_probs[j]) for j, c in enumerate(classes_)}
        df.at[i, "p_home"] = prob_map.get(1, 0.0)
        df.at[i, "p_draw"] = prob_map.get(0, 0.0)
        df.at[i, "p_away"] = prob_map.get(-1, 0.0)

    # --- ODDS → EV + filtri “anti-bombe” ---
    def implied(o):
        return 1.0 / o if o and o > 1e-6 else np.nan

    df["imp_home"] = df["odd_home"].apply(implied)
    df["imp_draw"] = df["odd_draw"].apply(implied)
    df["imp_away"] = df["odd_away"].apply(implied)

    bets = []

    for _, row in df.iterrows():
        candidati = []

        outcomes = [
            ("1", row.get("p_home"), row.get("odd_home")),
            ("X", row.get("p_draw"), row.get("odd_draw")),
            ("2", row.get("p_away"), row.get("odd_away")),
        ]

        for label, p_mod, odd in outcomes:
            if p_mod is None or pd.isna(p_mod):
                continue
            if odd is None or pd.isna(odd) or odd <= 1.01:
                continue

            p_imp = 1.0 / odd
            ev = p_mod * odd

            # FILTRI:
            if p_mod < 0.45:     # modello deve essere un minimo convinto
                continue
            if odd > 5.0:        # niente bombe a 6.50
                continue
            if ev < 1.02:        # EV davvero > 1
                continue
            if (p_mod - p_imp) < 0.00:  # modello deve battere il book di almeno 3 punti
                continue

            candidati.append({
                "label": label,
                "p_mod": round(p_mod, 3),
                "p_imp": round(p_imp, 3),
                "odd": odd,
                "ev": round(ev, 3),
            })

        if candidati:
            candidati.sort(key=lambda x: x["ev"], reverse=True)
            best = candidati[0]
            bets.append({
                "match_id": row["match_id"],
                "match": f"{row['home_team_name']} - {row['away_team_name']}",
                "pick": best["label"],
                "p_mod": best["p_mod"],
                "p_imp": best["p_imp"],
                "odd": best["odd"],
                "ev": best["ev"],
            })

    bets = sorted(bets, key=lambda x: x["ev"], reverse=True)

    if not bets:
        print("⚠️ Nessuna value bet con i criteri attuali.")
    else:
        print("\n💡 Proposte per la prossima giornata:")
        for b in bets:
            print(
                f"✅ {b['match']}: {b['pick']} | "
                f"p_mod={b['p_mod']}  p_imp={b['p_imp']}  odd={b['odd']}  EV={b['ev']}"
            )
    
    # --- SEZIONE 2: Previsioni pure (sempre mostrata) ---
    print("\n🔮 Previsioni del modello (schedina):")

    for _, row in df.iterrows():
        probs = {
            "1": row.get("p_home", 0.0),
            "X": row.get("p_draw", 0.0),
            "2": row.get("p_away", 0.0),
        }
        best_outcome = max(probs, key=probs.get)
        best_prob = probs[best_outcome]

        # mostra solo se la confidenza è abbastanza alta
        if best_prob < 0.30:
            continue

        print(
            f"🧩 {row['home_team_name']} - {row['away_team_name']} → "
            f"{best_outcome}  (p={best_prob:.2f})"
        )


if __name__ == "__main__":
    main()
