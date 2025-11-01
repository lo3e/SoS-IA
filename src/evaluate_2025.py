# src/evaluate_2025.py
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import json
from dotenv import load_dotenv
from pathlib import Path
import pickle

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH", "./data")
MODEL_PATH = os.path.join(DATA_PATH, "xgb_match_predictor.pkl")
META_PATH = Path(MODEL_PATH).with_suffix(".json")

# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================
def build_features_2025(conn):
    """
    Ricrea le stesse feature del training set v2
    per le partite concluse (FT) del 2025.
    """
    print("🧠 Genero feature per partite concluse 2025...")

    # === 1️⃣ Matches + standings
    matches = pd.read_sql_query(
        """
        SELECT
            m.match_id,
            m.date,
            m.season,
            m.league_id,
            m.round AS round_number,
            m.status,
            m.home_team_id,
            m.away_team_id,
            m.home_team_name,
            m.away_team_name,
            m.home_goals,
            m.away_goals,
            s_home.rank AS home_rank,
            s_away.rank AS away_rank,
            s_home.points AS home_points,
            s_away.points AS away_points,
            s_home.goals_for AS home_goals_for,
            s_home.goals_against AS home_goals_against,
            s_away.goals_for AS away_goals_for,
            s_away.goals_against AS away_goals_against
        FROM matches m
        LEFT JOIN standings s_home
            ON m.home_team_id = s_home.team_id AND m.season = s_home.season
        LEFT JOIN standings s_away
            ON m.away_team_id = s_away.team_id AND m.season = s_away.season
        WHERE m.status = 'FT' AND m.season = 2025
        ORDER BY m.date
        """,
        conn,
    )

    if matches.empty:
        raise ValueError("❌ Nessuna partita FT trovata per il 2025 nel database!")

    # === 2️⃣ Rolling performance (ultime 5 partite)
    matches = matches.sort_values("date")
    for side in ["home", "away"]:
        matches[f"{side}_points_gained"] = matches.apply(
            lambda r: 3 if (r["home_goals"] > r["away_goals"] and side == "home") or
                             (r["away_goals"] > r["home_goals"] and side == "away")
            else (1 if r["home_goals"] == r["away_goals"] else 0),
            axis=1
        )
        for metric, col in zip(
            ["points", "goals_for", "goals_against"],
            [f"{side}_points_gained", f"{side}_goals_for", f"{side}_goals_against"]
        ):
            matches[f"{side}_recent_{metric}_avg"] = (
                matches.groupby(f"{side}_team_id")[col]
                .rolling(5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # === 3️⃣ Head-to-Head (ultimi 5 match)
    print("⚔️  Calcolo feature Head-to-Head...")
    h2h_data = []
    for i, row in matches.iterrows():
        home, away, date = row["home_team_id"], row["away_team_id"], row["date"]
        past = matches[
            (
                ((matches["home_team_id"] == home) & (matches["away_team_id"] == away)) |
                ((matches["home_team_id"] == away) & (matches["away_team_id"] == home))
            ) & (matches["date"] < date)
        ].tail(5)

        if past.empty:
            h2h_data.append((0, 0, 0, 0, 0))
            continue

        home_wins = ((past["home_team_id"] == home) & (past["home_goals"] > past["away_goals"])).sum() + \
                    ((past["away_team_id"] == home) & (past["away_goals"] > past["home_goals"])).sum()
        away_wins = ((past["home_team_id"] == away) & (past["home_goals"] > past["away_goals"])).sum() + \
                    ((past["away_team_id"] == away) & (past["away_goals"] > past["home_goals"])).sum()
        draws = (past["home_goals"] == past["away_goals"]).sum()
        goal_diff_avg = (
            ((past["home_team_id"] == home) * (past["home_goals"] - past["away_goals"])) +
            ((past["away_team_id"] == home) * (past["away_goals"] - past["home_goals"]))
        ).mean()
        total = len(past)
        h2h_data.append((home_wins, away_wins, draws, goal_diff_avg, total))

    matches[["h2h_home_wins", "h2h_away_wins", "h2h_draws",
             "h2h_goal_diff_avg", "h2h_total_matches"]] = pd.DataFrame(h2h_data, index=matches.index)

    # === 4️⃣ Injuries
    injuries = pd.read_sql_query(
        """
        SELECT
            i.team_name AS team_name,
            COUNT(DISTINCT i.player_name) AS num_injuries,
            ROUND(SUM(COALESCE(p.performance_index, 0)), 2) AS injury_impact
        FROM current_unavailable_players i
        LEFT JOIN player_form_ranking p
            ON i.player_name = p.player_name
        GROUP BY i.team_name
        """,
        conn,
    )
    for side in ["home", "away"]:
        matches = matches.merge(
            injuries,
            how="left",
            left_on=f"{side}_team_name",
            right_on="team_name",
            suffixes=("", f"_{side}")
        )
        matches.rename(columns={
            "num_injuries": f"{side}_num_injuries",
            "injury_impact": f"{side}_injury_impact"
        }, inplace=True)
        matches.drop(columns=["team_name"], inplace=True, errors="ignore")

    matches.fillna(0, inplace=True)

    # === 5️⃣ Target vero (solo per valutazione)
    def outcome(row):
        if row["home_goals"] > row["away_goals"]:
            return 1
        elif row["home_goals"] < row["away_goals"]:
            return -1
        else:
            return 0
    matches["target_num"] = matches.apply(outcome, axis=1)

    # --- 6️⃣ FEATURE DI EQUILIBRIO MATCH (per migliorare la predizione dei pareggi) ---

    print("⚖️  Calcolo feature di equilibrio...")

    # Differenza tra rank
    if "home_rank" in matches.columns and "away_rank" in matches.columns:
        matches["rank_diff"] = (matches["home_rank"] - matches["away_rank"]).abs()
    else:
        matches["rank_diff"] = 0

    # Differenza tra punti in classifica
    if "home_points" in matches.columns and "away_points" in matches.columns:
        matches["points_diff"] = (matches["home_points"] - matches["away_points"]).abs()
    else:
        matches["points_diff"] = 0

    # Differenza media punti recenti (forma)
    if "home_recent_points_avg" in matches.columns and "away_recent_points_avg" in matches.columns:
        matches["form_diff"] = (matches["home_recent_points_avg"] - matches["away_recent_points_avg"]).abs()
    else:
        matches["form_diff"] = 0

    # Bonus: indice sintetico di equilibrio (0=molto sbilanciato, 1=molto equilibrato)
    max_rank_diff = matches["rank_diff"].max() if matches["rank_diff"].max() != 0 else 1
    matches["match_balance_index"] = 1 - (matches["rank_diff"] / max_rank_diff)

    print(f"✅ Feature set 2025 pronto ({matches.shape[0]} partite, {matches.shape[1]} colonne)")
    return matches

# ============================================================
# MAIN
# ============================================================
'''def main():
    print("🔍 Valutazione modello su partite 2025 concluse...")
    conn = sqlite3.connect(DB_PATH)
    df = build_features_2025(conn)
    conn.close()

    print(f"📦 Partite trovate: {len(df)}")

    print(f"📂 Carico modello da: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Carica lista feature del training
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_features = meta["features"]
    print(f"✅ Caricate {len(train_features)} feature dal modello addestrato.")

    # pulizia colonne
    drop_cols = [
        # identificativi
        "match_id", "date", "season", "league_id",
        "home_team_id", "away_team_id",
        "home_team_name", "away_team_name",
        "status",
        # target
        "target_1x2", "target_num",
        # roba del risultato
        "home_goals", "away_goals",
        "home_points_match", "away_points_match",
        "home_goals_diff", "away_goals_diff",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y_true = df["target_num"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Allineamento feature set
    for col in train_features:
        if col not in X.columns:
            X[col] = 0  # Aggiunge colonne mancanti (es. round_number_Regular Season - 10)

    # Rimuove eventuali colonne in eccesso
    X = X[train_features]

    # predizione
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n🎯 Accuracy su partite FT 2025: {acc:.3f}\n")
    print("📊 Report di classificazione:")
    print(classification_report(y_true, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0, -1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Home", "Draw", "Away"],
        yticklabels=["Home", "Draw", "Away"],
    )
    plt.title("Matrice di Confusione 2025 (1=Home, 0=Draw, -1=Away)")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.tight_layout()
    plt.show()

    # Analisi errori principali
    wrong = df[y_true != y_pred][
        ["home_team_name", "away_team_name", "target_num"]
    ].copy()
    print(f"\n❌ Predizioni errate: {len(wrong)} / {len(df)}")
    print("Esempi:")
    print(wrong.head(10))'''

def main():
    print("🔍 Valutazione modello XGBoost su partite 2025 concluse...")

    # 1) build delle feature come abbiamo fatto per RF
    conn = sqlite3.connect(DB_PATH)
    df = build_features_2025(conn)
    conn.close()

    print(f"📦 Partite trovate: {len(df)}")

    # 2) carico il modello (SOLO il modello, non un dict)
    print(f"📂 Carico modello da: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)  # CalibratedClassifierCV

    # 3) carico le feature usate in training
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_features = meta["features"]
    print(f"✅ Caricate {len(train_features)} feature dal modello addestrato.")

    # 4) preparo X e y
    drop_cols = [
        # identificativi
        "match_id", "date", "season", "league_id",
        "home_team_id", "away_team_id",
        "home_team_name", "away_team_name",
        "status",
        # target
        "target_1x2", "target_num",
        # roba del risultato (non vanno date al modello)
        "home_goals", "away_goals",
        "home_points_match", "away_points_match",
        "home_goals_diff", "away_goals_diff",
    ]

    # y vero (nel db è -1 / 0 / 1)
    y_true = df["target_num"].astype(int)

    # X grezzo
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # numerico + NaN -> 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 5) allineo le colonne con quelle del training
    #    (se nel 2025 manca una colonna la aggiungo a 0)
    for col in train_features:
        if col not in X.columns:
            X[col] = 0

    #    (se nel 2025 c’è una colonna in più, la tolgo)
    X = X[train_features]

    # 6) predizione
    # il modello è multiclasse e in training avevamo:
    #   -1 -> 0
    #    0 -> 1
    #    1 -> 2
    # quindi ORA dobbiamo fare l’inverso
    y_pred_encoded = model.predict(X)  # 0/1/2
    inv_map = {0: -1, 1: 0, 2: 1}
    y_pred = pd.Series(y_pred_encoded).map(inv_map).astype(int)

    # 7) metriche
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Accuracy su partite FT 2025: {acc:.3f}\n")
    print("📊 Report di classificazione:")
    print(classification_report(y_true, y_pred, digits=3))

    # 8) confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0, -1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Home", "Draw", "Away"],
        yticklabels=["Home", "Draw", "Away"],
    )
    plt.title("Matrice di Confusione 2025 (XGBoost — 1=Home, 0=Draw, -1=Away)")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.tight_layout()
    plt.show()

    # 9) errori
    wrong = df[y_true != y_pred][
        ["home_team_name", "away_team_name", "target_num"]
    ].copy()
    print(f"\n❌ Predizioni errate: {len(wrong)} / {len(df)}")
    print("Esempi:")
    print(wrong.head(10))

if __name__ == "__main__":
    main()
