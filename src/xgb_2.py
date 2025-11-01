# src/train_xgb.py

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

# ============ CONFIG ============
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
if not DATA_PATH:
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"
else:
    DATA_PATH = Path(DATA_PATH)

CSV_PATH = DATA_PATH / "training_data_v2.csv"
MODEL_PATH = DATA_PATH / "xgb_match_predictor.pkl"
META_PATH = DATA_PATH / "xgb_match_predictor.json"

# ============ SCRIPT ============

print(f"📂 Carico dataset da: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset shape: {df.shape}")

# target in input è -1 / 0 / 1
y_raw = df["target_num"]

# mappiamo a 0/1/2 per XGB
label_map = {-1: 0, 0: 1, 1: 2}
y = y_raw.map(label_map).astype(int)

# stesse colonne da droppare del RF
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

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"📦 Dopo il drop: {X.shape[1]} colonne")

# qui facciamo ESATTAMENTE come nel RF:
# - se è object proviamo a convertirla
# - se non si converte, facciamo get_dummies
obj_cols = [c for c in X.columns if X[c].dtype == "object"]

for col in obj_cols:
    try:
        X[col] = pd.to_numeric(X[col], errors="raise")
    except Exception:
        # one-hot solo per questa colonna
        X = pd.get_dummies(X, columns=[col], drop_first=True)

# a questo punto tutte dovrebbero essere numeriche
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"✅ Feature matrix shape: {X.shape}, Target length: {len(y)}")

# train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGB base
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",  # più veloce
    random_state=42,
)

# grid piccola ma sensata
param_grid = {
    "n_estimators": [150, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

print("🚀 Inizio GridSearch per XGBoost...")
search = GridSearchCV(
    xgb,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

best_xgb = search.best_estimator_
print("🏅 Best params:", search.best_params_)

# calibrazione (come per RF)
calibrated_xgb = CalibratedClassifierCV(best_xgb, cv=3, method="isotonic")
calibrated_xgb.fit(X_train, y_train)

# valutazione
y_pred = calibrated_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy complessiva: {acc:.3f}\n")
print("📊 Report di classificazione (classi: 0=away, 1=draw, 2=home):")
print(classification_report(y_test, y_pred, digits=3))

# matrice di confusione (la rigiriamo come fai tu: 1,0,-1)
# cioè 2 -> home, 1 -> draw, 0 -> away
cm = confusion_matrix(y_test, y_pred, labels=[2, 1, 0])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Home (2)", "Draw (1)", "Away (0)"],
    yticklabels=["Home (2)", "Draw (1)", "Away (0)"],
)
plt.title("Matrice di Confusione (XGBoost)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()
plt.close()

# curva di calibrazione SOLO per la classe "home" (2)
prob_pos = calibrated_xgb.predict_proba(X_test)[:, 2]
true_pos = (y_test == 2).astype(int)
frac_pos, mean_pred = calibration_curve(true_pos, prob_pos, n_bins=5)

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, "o-", label="Calibrato")
plt.plot([0, 1], [0, 1], "k--", label="Ideale")
plt.title("Curva di calibrazione (classe 2 = Vittoria Casa)")
plt.xlabel("Probabilità predetta")
plt.ylabel("Probabilità reale")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# feature importance (dal best_xgb, NON dal calibrato)
importances = pd.Series(
    best_xgb.feature_importances_, index=X.columns
).sort_values(ascending=False)

top_n = 15 if len(importances) > 15 else len(importances)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.iloc[:top_n].values, y=importances.iloc[:top_n].index, palette="viridis")
plt.title("Top 15 Feature Importance (XGBoost ottimizzato)")
plt.tight_layout()
plt.show()
plt.close()

# salvataggio modello e metadati
joblib.dump(calibrated_xgb, MODEL_PATH)
print(f"💾 Modello XGBoost calibrato salvato in: {MODEL_PATH}")

meta = {
    "features": list(X.columns),
    "best_params": search.best_params_,
    "label_map": label_map,  # importantissimo per l'evaluate
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=4)
print(f"🧾 Metadati salvati in: {META_PATH}")

print("✅ Fine training.")
