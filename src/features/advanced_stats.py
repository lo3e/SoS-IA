import pandas as pd
import numpy as np
from src.core.db import fetch_df

def build_advanced_stats(season: int, cutoff_round: int) -> pd.DataFrame:
    """
    Costruisce statistiche avanzate di squadra (xG, tiri, passaggi, possesso, ecc.)
    a partire dalla tabella `team_stats`, aggregando i dati per match_id e team_id.
    """

    print(f"[DEBUG] Caricamento team_stats per season={season}, cutoff={cutoff_round}")

    query = f"""
        SELECT
            ts.match_id,
            ts.team_id,
            ts.team_name,
            ts.stat_type,
            ts.stat_value,
            m.season,
            m.round
        FROM team_stats ts
        JOIN matches m ON ts.match_id = m.match_id
        WHERE m.season = {season}
    """

    if cutoff_round is not None:
        query += f"""
            AND CAST(REPLACE(m.round, 'Regular Season - ', '') AS INTEGER) <= {cutoff_round}
        """

    df = fetch_df(query)

    if df.empty:
        print("⚠️ Nessuna statistica disponibile per advanced stats.")
        return pd.DataFrame()

    for col in ["stat_value"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace(["None", "nan", "None%", ""], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Pivot long → wide ---
    pivot = df.pivot_table(
        index=["match_id", "team_id", "team_name"],
        columns="stat_type",
        values="stat_value",
        aggfunc="first"
    ).reset_index()

    # --- Rinomina colonne ---
    rename_map = {
        "Shots on Goal": "shots_on_goal",
        "Shots off Goal": "shots_off_goal",
        "Total Shots": "total_shots",
        "Blocked Shots": "blocked_shots",
        "Shots insidebox": "shots_inside_box",
        "Shots outsidebox": "shots_outside_box",
        "Fouls": "fouls",
        "Corner Kicks": "corner_kicks",
        "Offsides": "offsides",
        "Ball Possession": "possession",
        "Yellow Cards": "yellow_cards",
        "Red Cards": "red_cards",
        "Goalkeeper Saves": "goalkeeper_saves",
        "Total passes": "passes_total",
        "Passes accurate": "passes_accurate",
        "Passes %": "passes_accuracy",
        "Expected Goals": "xg",  # solo se presente
    }
    pivot.rename(columns=rename_map, inplace=True)

    # --- Feature derivate ---
    pivot["shot_accuracy"] = pivot["shots_on_goal"] / pivot["total_shots"]
    pivot["passes_ratio"] = pivot["passes_accurate"] / pivot["passes_total"]
    pivot["possession"] = pivot["possession"] / 100.0

    # xG per tiro (se xG disponibile)
    if "xg" in pivot.columns:
        pivot["xg_per_shot"] = pivot["xg"] / pivot["total_shots"]
    else:
        pivot["xg_per_shot"] = 0.0

    # --- Riempi NaN con 0 ---
    pivot.fillna(0, inplace=True)

    print(f"[DEBUG] Advanced stats generate: {len(pivot)} record, {len(pivot.columns)} colonne")
    print(f"[DEBUG] Colonne disponibili: {list(pivot.columns)}")

    return pivot

def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola feature derivate basate sulle advanced stats già integrate.
    Usa solo colonne effettivamente presenti nel dataset finale.
    """

    # Safety: evita errori su colonne mancanti
    cols = df.columns

    # --- Conversione in numerico con gestione percentuali ---
    for c in ["home_possession", "away_possession",
              "home_passes_accuracy", "away_passes_accuracy"]:
        if c in cols:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("None", np.nan)
                .replace("", np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Derivate base ---

    # Shot conversion rate (goals / total shots)
    if "home_total_shots" in cols and "home_goals" in cols:
        df["home_shot_conversion"] = df["home_goals"] / df["home_total_shots"].replace(0, np.nan)
    if "away_total_shots" in cols and "away_goals" in cols:
        df["away_shot_conversion"] = df["away_goals"] / df["away_total_shots"].replace(0, np.nan)

    # Shot accuracy (on goal / total)
    if "home_shots_on_goal" in cols and "home_total_shots" in cols:
        df["home_shot_accuracy_ratio"] = df["home_shots_on_goal"] / df["home_total_shots"].replace(0, np.nan)
    if "away_shots_on_goal" in cols and "away_total_shots" in cols:
        df["away_shot_accuracy_ratio"] = df["away_shots_on_goal"] / df["away_total_shots"].replace(0, np.nan)

    # Possession difference
    if "home_possession" in cols and "away_possession" in cols:
        df["possession_diff"] = df["home_possession"] - df["away_possession"]

    # Pass accuracy difference
    if "home_passes_accuracy" in cols and "away_passes_accuracy" in cols:
        df["passes_accuracy_diff"] = df["home_passes_accuracy"] - df["away_passes_accuracy"]

    # Total shots difference
    if "home_total_shots" in cols and "away_total_shots" in cols:
        df["shots_total_diff"] = df["home_total_shots"] - df["away_total_shots"]

    # XG per shot difference
    if "home_xg_per_shot" in cols and "away_xg_per_shot" in cols:
        df["xg_per_shot_diff"] = df["home_xg_per_shot"] - df["away_xg_per_shot"]

    # Defensive stability (saves per shots on goal conceded)
    if "home_saves" in cols and "away_shots_on_goal" in cols:
        df["home_defensive_stability"] = df["home_saves"] / df["away_shots_on_goal"].replace(0, np.nan)
    if "away_saves" in cols and "home_shots_on_goal" in cols:
        df["away_defensive_stability"] = df["away_saves"] / df["home_shots_on_goal"].replace(0, np.nan)

    # Discipline ratio (fouls per card)
    if all(c in cols for c in ["home_fouls", "home_yellow_cards"]):
        df["home_discipline_ratio"] = df["home_fouls"] / (df["home_yellow_cards"].fillna(0) + 1)
    if all(c in cols for c in ["away_fouls", "away_yellow_cards"]):
        df["away_discipline_ratio"] = df["away_fouls"] / (df["away_yellow_cards"].fillna(0) + 1)

    # --- Cleanup numerico ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[DEBUG] 🧮 Derived stats calcolate: {len(set(df.columns) - set(cols))} nuove colonne")
    return df