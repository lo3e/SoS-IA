# src/evaluate_2025.py
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH", "./data")
MODEL_PATH = os.path.join(DATA_PATH, "rf_match_predictor.pkl")

# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================
def build_features_2025(conn):
    """
    Ricrea le stesse feature del training set,
    ma solo per le partite FT del 2025.
    """
    print("🧠 Genero feature per partite concluse 2025...")

    matches = pd.read_sql_query(
        """
        SELECT
            m.match_id,
            m.date,
            m.season,
            m.league_id,
            m.home_team_id,
            m.away_team_id,
            m.home_team_name,
            m.away_team_name,
            m.home_goals,
            m.away_goals,
            m.status
        FROM matches m
        WHERE m.status = 'FT' AND m.season = 2025
        ORDER BY m.date
        """,
        conn,
    )

    if matches.empty:
        raise ValueError("❌ Nessuna partita FT trovata per il 2025 nel database!")

    # standings
    standings = pd.read_sql_query(
        """
        SELECT
            season,
            team_id,
            rank,
            points,
            goals_diff,
            goals_for,
            goals_against
        FROM standings
        """,
        conn,
    )

    # merge home / away
    matches = matches.merge(
        standings.add_prefix("home_"),
        left_on=["season", "home_team_id"],
        right_on=["home_season", "home_team_id"],
        how="left",
    ).merge(
        standings.add_prefix("away_"),
        left_on=["season", "away_team_id"],
        right_on=["away_season", "away_team_id"],
        how="left",
    )

    matches.drop(columns=["home_season", "away_season"], inplace=True, errors="ignore")

    # injuries
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
    injuries_home = injuries.add_prefix("home_")
    injuries_away = injuries.add_prefix("away_")

    matches = matches.merge(
        injuries_home,
        left_on="home_team_name",
        right_on="home_team_name",
        how="left",
    ).merge(
        injuries_away,
        left_on="away_team_name",
        right_on="away_team_name",
        how="left",
    )

    # predictions (per 2025 ci sono)
    preds = pd.read_sql_query(
        """
        SELECT match_id, prob_home, prob_draw, prob_away
        FROM predictions
        """,
        conn,
    )

    df = matches.merge(preds, on="match_id", how="left")

    # pulizia prob
    def clean_prob(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            x = x.replace("%", "")
        try:
            val = float(x)
        except:
            return 0.0
        return val / 100 if val > 1 else val

    for c in ["prob_home", "prob_draw", "prob_away"]:
        df[c] = df[c].apply(clean_prob)

    # target vero
    def outcome(row):
        if row["home_goals"] > row["away_goals"]:
            return 1
        elif row["home_goals"] < row["away_goals"]:
            return -1
        else:
            return 0

    df["target_num"] = df.apply(outcome, axis=1)

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("🔍 Valutazione modello su partite 2025 concluse...")
    conn = sqlite3.connect(DB_PATH)
    df = build_features_2025(conn)
    conn.close()

    print(f"📦 Partite trovate: {len(df)}")

    # carica modello
    model = joblib.load(MODEL_PATH)

    # pulizia colonne
    drop_cols = [
        "match_id", "date", "season", "league_id", "home_team_id",
        "away_team_id", "home_team_name", "away_team_name", "status",
        "target_num", "home_goals", "away_goals"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y_true = df["target_num"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

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
    print(wrong.head(10))


if __name__ == "__main__":
    main()
