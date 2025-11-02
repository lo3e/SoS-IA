import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# === CONFIG ===
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
if not DATA_PATH:
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"

CSV_PATH = Path(DATA_PATH) / "training_data_v2.csv"
MODEL_PATH = Path(DATA_PATH) / "xgb_match_predictor.pkl"
META_PATH = Path(DATA_PATH) / "xgb_match_predictor.json"

# === Colonne da escludere ===
DROP_COLS = [
    "match_id", "date", "season", "league_id",
    "home_team_id", "away_team_id",
    "home_team_name", "away_team_name",
    "status",
    "target_1x2", "target_num",
    "home_goals", "away_goals",
    "home_points_match", "away_points_match",
    "home_goals_diff", "away_goals_diff",
]

# === 1️⃣ Caricamento dati ===
print(f"📂 Carico dataset da: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset shape: {df.shape}")

X = df.drop(columns=DROP_COLS, errors="ignore")
X["round_number"] = X["round_number"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
y = df["winner"] if "winner" in df.columns else df["target_num"]

print(f"✅ Feature matrix shape: {X.shape}, Target length: {len(y)}")

# XGBoost non accetta target negativi
y = y.map({-1: 0, 0: 1, 1: 2})

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 2️⃣ Definizione modello XGBoost ===
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)

# GridSearch per ottimizzare
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

print("🚀 Inizio GridSearch per XGBoost...")
search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print(f"🏅 Best params: {search.best_params_}")

# === 3️⃣ Calibrazione delle probabilità ===
print("⚖️  Calibrazione delle probabilità...")
calibrated_xgb = CalibratedClassifierCV(best_xgb, method="sigmoid", cv=3)
calibrated_xgb.fit(X_train, y_train)

# === 4️⃣ Valutazione ===
# Predizioni e rimappatura
y_pred = pd.Series(calibrated_xgb.predict(X_test))
# Rimappa indietro per valutazione
y_test_mapped = y_test.map({0: -1, 1: 0, 2: 1})
y_pred_mapped = y_pred.map({0: -1, 1: 0, 2: 1})

acc = accuracy_score(y_test_mapped, y_pred_mapped)
print(f"\n🎯 Accuracy complessiva: {acc:.3f}\n")
print("📊 Report di classificazione:")
print(classification_report(y_test_mapped, y_pred_mapped))

# Confusion matrix
cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=[1, 0, -1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Home (1)", "Draw (0)", "Away (-1)"])
disp.plot(cmap="Blues")
plt.title("Matrice di Confusione (XGBoost)")
plt.tight_layout()
plt.show()

# === 5️⃣ Feature importance ===
importances = pd.Series(
    getattr(calibrated_xgb, "estimator_", calibrated_xgb.estimator).feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances.iloc[:15].values,
    y=importances.iloc[:15].index,
    palette="viridis"
)
plt.title("Top 15 Feature Importance (XGBoost ottimizzato)")
plt.xlabel("Importanza")
plt.tight_layout()
plt.show()

# === 6️⃣ Curva di calibrazione ===
print("📈 Genero curva di calibrazione...")
prob_true, prob_pred = calibration_curve(
    (y_test == 1).astype(int),
    calibrated_xgb.predict_proba(X_test)[:, list(calibrated_xgb.classes_).index(1)],
    n_bins=10
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Calibrato")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideale")
plt.title("Curva di calibrazione (classe 1 = Vittoria Casa)")
plt.xlabel("Probabilità predetta")
plt.ylabel("Probabilità reale")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 7️⃣ Salvataggio ===
joblib.dump(calibrated_xgb, MODEL_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_type": "xgboost",
            "best_params": search.best_params_,
            "accuracy": acc,
            "features": list(X.columns)
        },
        f,
        indent=4
    )

print(f"💾 Modello XGBoost calibrato salvato in: {MODEL_PATH}")
print(f"🧾 Metadati salvati in: {META_PATH}")
print("✅ Fine training.")
