# ============================================================
# src/features/feature_engineering.py
# Centralizza tutte le trasformazioni e feature derivate
# ============================================================

import pandas as pd
import numpy as np
from typing import Literal
from src.core.logger import get_logger
from src.core.utils import form_to_index
from src.features.advanced_stats import build_advanced_stats, compute_derived_stats

logger = get_logger(__name__)


# ---------------------------
# 🔧 Utility Functions
# ---------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    """Converte in float qualsiasi colonna numerica o percentuale."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def _percent_to_float(series: pd.Series) -> pd.Series:
    """Converte '65%' → 0.65."""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
        .div(100)
        .fillna(0)
    )


def _encode_form(form: str) -> float:
    """Codifica una stringa forma tipo 'WWDLW' in un indice numerico medio."""
    if not isinstance(form, str) or form.strip() == "":
        return 0.5
    mapping = {"W": 1, "D": 0.5, "L": 0}
    values = [mapping.get(ch, 0.5) for ch in form.strip()]
    return np.mean(values) if values else 0.5


# ---------------------------
# 🧠 Main Feature Function
# ---------------------------

def compute_features(df: pd.DataFrame, mode: Literal["train", "eval", "predict"], cutoff_round=None, season=None) -> pd.DataFrame:
    """
    Calcola tutte le feature numeriche e derivate in modo coerente
    per training, evaluation e betting.
    """

    df = df.copy()

    # 🔢 Conversione form → indici numerici
    if "home_form" in df.columns and "away_form" in df.columns:
        df["home_form_index"] = df["home_form"].apply(form_to_index)
        df["away_form_index"] = df["away_form"].apply(form_to_index)
        logger.info("🧠 Form convertita automaticamente in indici numerici (%s).", mode)
        
    # 1️⃣ Conversioni di tipo
    for col in ["prob_home", "prob_draw", "prob_away"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = _percent_to_float(df[col])

    for col in ["home_rank", "away_rank", "home_points", "away_points"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col]).fillna(0)

    # 2️⃣ Encoding della forma
    if "form_home" in df.columns and "form_away" in df.columns:
        df["home_form_index"] = df["form_home"].apply(_encode_form)
        df["away_form_index"] = df["form_away"].apply(_encode_form)
        logger.info("🧠 Form dei team convertita in indici numerici.")
    else:
        df["home_form_index"] = 0.5
        df["away_form_index"] = 0.5

    # 3️⃣ Feature derivate principali
    base_features = [
        ("points_diff", "home_points", "away_points"),
        ("rank_diff", "home_rank", "away_rank"),
        ("form_diff", "home_form_index", "away_form_index"),
    ]
    for new_col, home_col, away_col in base_features:
        if home_col in df.columns and away_col in df.columns:
            df[new_col] = df[home_col] - df[away_col]
        else:
            df[new_col] = 0

    # 4️⃣ Feature di equilibrio (draw tendency, bilanciamento)
    if all(col in df.columns for col in ["prob_home", "prob_away"]):
        df["expected_draw_tendency"] = 1 - abs(df["prob_home"] - df["prob_away"])
    else:
        df["expected_draw_tendency"] = 0.5

    # Bilanciamenti normalizzati (versione classica)
    df["rank_balance"] = 1 / (1 + abs(df["rank_diff"]))
    df["points_balance"] = 1 / (1 + abs(df["points_diff"]))
    df["form_balance"] = 1 / (1 + abs(df["form_diff"]))

    # 🔍 Nuove feature di simmetria (versione relativa)
    # più interpretabili e con range [0,1] più lineare
    df["rank_similarity"] = (
        1 - abs(df["home_rank"] - df["away_rank"]) / (df[["home_rank", "away_rank"]].max(axis=1) + 1)
    )
    df["points_similarity"] = (
        1 - abs(df["home_points"] - df["away_points"]) / (df["home_points"] + df["away_points"] + 1)
    )
    df["form_similarity"] = (
        1 - abs(df["home_form_index"] - df["away_form_index"]) / 5
    )

    for col in ["rank_similarity", "points_similarity", "form_similarity"]:
        df[col] = df[col].clip(0, 1)
        df[col] = df[col].fillna(0.5)


    # 4️⃣.bis Rolling stats (ultimi 5 match)
    if mode in ["train", "eval"]:
        df = compute_rolling_stats(df)
    
    # 4️⃣.tris Advanced stats (xG, shots, actions)
    try:
        # Calcola solo se non siamo in "predict"
        if mode in ["train", "eval"]:
            # Se non passato, estrai dai dati
            if cutoff_round is None:
                if "round" in df.columns:
                    import re
                    round_nums = df["round"].astype(str).str.extract(r"(\d+)").dropna()
                    if not round_nums.empty:
                        cutoff_round = int(round_nums[0].astype(int).max()) - 1

            if season is None and "season" in df.columns:
                season = int(df["season"].max())

            adv_stats = build_advanced_stats(season=season, cutoff_round=cutoff_round)
        else:
            # in modalità predict NON filtriamo per round
            if season is None and "season" in df.columns:
                season = int(df["season"].max())
            adv_stats = build_advanced_stats(season=season, cutoff_round=None)

        if not adv_stats.empty:
            home_adv = adv_stats.rename(columns={
                "team_id": "home_team_id",
                "possession": "home_possession",
                "corner_kicks": "home_corners",
                "fouls": "home_fouls",
                "goalkeeper_saves": "home_saves",
                "passes_accuracy": "home_passes_accuracy",
                "shots_on_goal": "home_shots_on_goal",
                "shots_off_goal": "home_shots_off_goal",
                "shots_inside_box": "home_shots_in_box",
                "shots_outside_box": "home_shots_out_box",
                "total_shots": "home_total_shots",
                "xg_per_shot": "home_xg_per_shot"
            }).drop(columns=["team_name"], errors="ignore")

            away_adv = adv_stats.rename(columns={
                "team_id": "away_team_id",
                "possession": "away_possession",
                "corner_kicks": "away_corners",
                "fouls": "away_fouls",
                "goalkeeper_saves": "away_saves",
                "passes_accuracy": "away_passes_accuracy",
                "shots_on_goal": "away_shots_on_goal",
                "shots_off_goal": "away_shots_off_goal",
                "shots_inside_box": "away_shots_in_box",
                "shots_outside_box": "away_shots_out_box",
                "total_shots": "away_total_shots",
                "xg_per_shot": "away_xg_per_shot"
            }).drop(columns=["team_name"], errors="ignore")

            df = df.merge(home_adv.drop_duplicates("match_id"), on="match_id", how="left")
            df = df.merge(away_adv.drop_duplicates("match_id"), on="match_id", how="left")

            drop_cols = [c for c in df.columns if c.endswith("_homeadv") or c.endswith("_awayadv") or c.endswith("_y") 
                         or c.endswith("_x")]
            df = df.drop(columns=drop_cols, errors="ignore")

            print(f"[DEBUG] ✅ Advanced stats integrate automaticamente in mode={mode} ({len(adv_stats)} righe)")

    except Exception as e:
        print(f"[WARN] Advanced stats non integrate in mode={mode}: {e}")
    
    # 4️⃣.quater Altre feature derivate
    df = compute_derived_stats(df)

    # 5️⃣ Pulizia finale
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    logger.info(f"🔁 Feature numeriche e derivate calcolate correttamente per la modalità: {mode}")
    return df

def compute_rolling_stats(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola rolling features (ultimi 5 match) per ogni squadra.
    Usa shift(1) per evitare data leakage e considera solo partite concluse.
    """
    df = dataset.copy()

    if "status" in df.columns:
        df = df[df["status"] == "FT"].copy()

    df = df.sort_values(["date"])

    # punti per match (3/1/0)
    df["points_per_match"] = np.select(
        [df["label_result"] == "H", df["label_result"] == "D", df["label_result"] == "A"],
        [3, 1, 0],
        default=0,
    )

    rolling_features = []

    for team_col, prefix in [("home_team_id", "home"), ("away_team_id", "away")]:
        temp = df.rename(columns={
            team_col: "team_id",
            f"{prefix}_goals": "goals_scored",
            f"{'away' if prefix == 'home' else 'home'}_goals": "goals_conceded",
        })[
            ["team_id", "date", "goals_scored", "goals_conceded", "points_per_match"]
        ].copy()

        temp = temp.sort_values(["team_id", "date"])
        group = temp.groupby("team_id", group_keys=False)

        temp[f"{prefix}_avg_points_last5"] = group["points_per_match"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        temp[f"{prefix}_avg_goals_scored_last5"] = group["goals_scored"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        temp[f"{prefix}_avg_goals_conceded_last5"] = group["goals_conceded"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        temp[f"{prefix}_goal_diff_avg_last5"] = (
            temp[f"{prefix}_avg_goals_scored_last5"] - temp[f"{prefix}_avg_goals_conceded_last5"]
        )
        temp[f"{prefix}_win_rate_last5"] = group["points_per_match"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=1).apply(lambda r: np.mean(r == 3))
        )

        temp = temp.drop(columns=["goals_scored", "goals_conceded", "points_per_match"])
        rolling_features.append(temp)

    # merge con dataset principale
    merged = dataset.merge(
        rolling_features[0], left_on=["home_team_id", "date"], right_on=["team_id", "date"], how="left"
    ).drop(columns="team_id")

    merged = merged.merge(
        rolling_features[1], left_on=["away_team_id", "date"], right_on=["team_id", "date"], how="left"
    ).drop(columns="team_id")

    return merged