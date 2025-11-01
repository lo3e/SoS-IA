# src/train_model_optimized.py
import os
import json
import joblib
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
if not DATA_PATH:
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"

CSV_PATH = Path(DATA_PATH) / "training_data_v2.csv"
MODEL_PATH = Path(DATA_PATH) / "rf_match_predictor_v3.pkl"

def main():
    print(f"📂 Carico dataset da: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # target
    y = df["target_num"]

    # colonne da droppare perché identitarie / di target / inutili
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
    # lascia standings, injuries e probs
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # alcune colonne potrebbero essere bool/string a 0/1 -> converti
    for c in X.columns:
        if X[c].dtype == "object":
            # prova a convertire, altrimenti one-hot
            try:
                X[c] = pd.to_numeric(X[c], errors="raise")
            except Exception:
                X = pd.get_dummies(X, columns=[c], drop_first=True)

    print(f"✅ Feature matrix shape: {X.shape}, Target length: {len(y)}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 1) Randomized search sul RandomForest grezzo ---
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [None, 6, 8, 10, 12, 15],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
        "class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=40,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    print("🌲 Addestro RandomForest con ricerca iperparametri...")
    search.fit(X_train, y_train)

    best_rf: RandomForestClassifier = search.best_estimator_
    print("🏅 Best params:", search.best_params_)

    # --- 2) calibrazione del *miglior* RF ---
    # (la usiamo solo per predire con proba "pulite")
    calibrated_rf = CalibratedClassifierCV(best_rf, cv=3, method="isotonic")
    calibrated_rf.fit(X_train, y_train)

    # --- 3) valutazione sul test set ---
    y_pred = calibrated_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy complessiva: {acc:.3f}\n")

    print("📊 Report di classificazione:")
    print(classification_report(y_test, y_pred, digits=3))

    # matrice di confusione
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Home", "Draw", "Away"],
        yticklabels=["Home", "Draw", "Away"],
    )
    plt.title("Matrice di Confusione (1=Home, 0=Draw, -1=Away)")
    plt.ylabel("Reale")
    plt.xlabel("Predetto")
    plt.tight_layout()
    plt.show()
    plt.close()

    # --- 4) feature importance: PRENDILA DAL RANDOMFOREST, NON dal calibrato ---
    importances = pd.Series(
        best_rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    top_n = 15 if len(importances) > 15 else len(importances)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.iloc[:top_n].values, y=importances.iloc[:top_n].index, palette="viridis")
    plt.title("Top Feature Importance (Random Forest ottimizzato)")
    plt.xlabel("Importanza")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Salva SOLO il modello (senza dizionario)
    joblib.dump(calibrated_rf, MODEL_PATH)
    print(f"💾 Modello calibrato salvato in: {MODEL_PATH}")

    # Salva metadati a parte per sicurezza (parametri + feature list)
    meta_path = Path(MODEL_PATH).with_suffix(".json")
    metadata = {
        "features": list(X.columns),
        "best_params": search.best_params_,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"🧾 Metadati salvati in: {meta_path}")

    # --- 6) salviamo anche un mini-report JSON ---
    report = {
        "accuracy": float(acc),
        "best_params": search.best_params_,
        "top_features": importances.head(15).to_dict(),
    }
    with open(Path(DATA_PATH) / "rf_training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("📝 Report salvato.")

if __name__ == "__main__":
    main()
