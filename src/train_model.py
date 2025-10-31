import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from dotenv import load_dotenv

# =============================================================
# 1️⃣ Setup e caricamento dati
# =============================================================
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "./data")
CSV_PATH = os.path.join(DATA_PATH, "training_data.csv")

print(f"📂 Carico dataset da: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# =============================================================
# 2️⃣ Preparazione feature e target
# =============================================================
# =============================================================
# Target
# =============================================================
if "target_num" in df.columns:
    y = df["target_num"]
else:
    y = df["target_1x2"].map({"H": 1, "D": 0, "A": -1}).fillna(0)

# Rimuoviamo colonne non numeriche o identificative
drop_cols = [
    "match_id", "date", "season", "league_id", "home_team_id",
    "away_team_id", "home_team_name", "away_team_name", "status",
    "target_1x2", "target_num", "home_goals", "away_goals"
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Conversione a numerico (forza eventuali errori in float)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"✅ Feature matrix shape: {X.shape}, Target length: {len(y)}")

# =============================================================
# 3️⃣ Split Train/Test
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================
# 4️⃣ Addestramento modello Random Forest
# =============================================================
print("🌲 Addestro modello Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# =============================================================
# 5️⃣ Valutazione modello
# =============================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy complessiva: {acc:.3f}")
print("\n📊 Report di classificazione:")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home", "Draw", "Away"],
            yticklabels=["Home", "Draw", "Away"])
plt.title("Matrice di Confusione (1=Home, 0=Draw, -1=Away)")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()

# =============================================================
# 📆 5bis️⃣ Analisi accuratezza per stagione
# =============================================================
if "season" in df.columns:
    print("\n📅 Analisi performance per stagione:")

    season_scores = []
    for season in sorted(df["season"].unique()):
        season_mask = df["season"] == season
        X_season = X.loc[season_mask]
        y_season = y.loc[season_mask]

        # Se ci sono abbastanza partite, valutiamo
        if len(X_season) > 30:
            y_pred_season = model.predict(X_season)
            acc_season = accuracy_score(y_season, y_pred_season)
            season_scores.append((season, acc_season))
            print(f"  🏟️ {season}: {acc_season:.3f}")

    # Grafico
    if season_scores:
        seasons, scores = zip(*season_scores)
        plt.figure(figsize=(8, 4))
        plt.plot(seasons, scores, marker="o", linewidth=2, color="teal")
        plt.title("Accuratezza del modello per stagione")
        plt.xlabel("Stagione")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

# =============================================================
# 6️⃣ Analisi Feature Importance
# =============================================================
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Feature Importance (Random Forest)")
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# =============================================================
# 7️⃣ Salvataggio modello (facoltativo per step 3)
# =============================================================
import joblib
MODEL_PATH = os.path.join(DATA_PATH, "rf_match_predictor.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\n💾 Modello salvato in: {MODEL_PATH}")
