import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Carica path dai .env
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "./data")
CSV_PATH = os.path.join(DATA_PATH, "training_data.csv")
MODEL_PATH = os.path.join(DATA_PATH, "rf_match_predictor_optimized.pkl")

print(f"📂 Carico dataset da: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Target e feature
y = df["target_num"]
drop_cols = [
    "match_id", "date", "league_id",
    "home_team_id", "away_team_id",
    "home_team_name", "away_team_name",
    "status", "target_1x2", "target_num",
    "home_goals", "away_goals", "season"
]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Pulizia feature
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].astype("category").cat.codes

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print(f"✅ Feature matrix shape: {X.shape}, Target length: {len(y)}")

# Parametri per tuning
param_dist = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

print("\n🎯 Eseguo RandomizedSearchCV per ottimizzare il Random Forest...")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=30, cv=3, verbose=1, n_jobs=-1, scoring="accuracy", random_state=42
)
search.fit(X_train, y_train)

print(f"✅ Migliori parametri trovati: {search.best_params_}")
best_rf = search.best_estimator_

# Calibrazione probabilità
print("\n⚙️ Calibro le probabilità (CalibratedClassifierCV)...")
calibrated_rf = CalibratedClassifierCV(best_rf, cv=3)
calibrated_rf.fit(X_train, y_train)

# Valutazione finale
y_pred = calibrated_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy ottimizzata: {acc:.3f}\n")
print(classification_report(y_test, y_pred))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home", "Draw", "Away"],
            yticklabels=["Home", "Draw", "Away"])
plt.title("Matrice di Confusione (Random Forest Ottimizzato)")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()

# Feature importance
importances = pd.Series(calibrated_rf.base_estimator_.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Feature Importance (Random Forest Ottimizzato)")
plt.xlabel("Importanza")
plt.show()

# Salva modello
joblib.dump(calibrated_rf, MODEL_PATH)
print(f"💾 Modello salvato in: {MODEL_PATH}")
