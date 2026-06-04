"""
match_winner.py
===============
Predicts IPL match winners using a Random Forest Classifier.

Features (auto-detected subset present in the CSV):
    season, team1, team2, city, venue, toss_winner, toss_decision
Target:
    winner

BUG FIXES:
    1. Actual champion sourced from VERIFIED_SEASONS (config.py) instead of
       computing "most-wins team" from the CSV — these differ in ~32% of seasons
       (e.g. 2009, 2010, 2014, 2023, 2025). Final-match-winner detection was the
       previous partial fix; using hardcoded verified records is fully accurate.

Public API:
    train_match_winner_model(data_path, model_save_path)
        -> (model, encoders, accuracy, season_champions_df)
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Verified ground-truth import (works whether run directly or via main.py)
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ    = os.path.dirname(_SRC_DIR)
for _p in (_PROJ, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.config import get_verified_champion
except ImportError:
    from config import get_verified_champion

# ---------------------------------------------------------------------------
PREFERRED_FEATURES = [
    "season", "team1", "team2", "city", "venue", "toss_winner", "toss_decision"
]
TARGET_COLUMN  = "winner"
MODEL_FILENAME = "winner_model.pkl"


# ---------------------------------------------------------------------------
def _detect_features(df):
    available = [c for c in PREFERRED_FEATURES if c in df.columns]
    skipped   = [c for c in PREFERRED_FEATURES if c not in df.columns]
    if skipped:
        print(f"[WARN] Feature columns not in CSV (skipped): {skipped}")
    print(f"[INFO] Using features : {available}")
    return available


def _label_encode(X, y):
    X_enc    = X.copy()
    encoders = {}
    for col in X_enc.columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
        print(f"[INFO]   '{col}' -> {len(le.classes_)} unique classes")
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))
    encoders[TARGET_COLUMN] = target_le
    print(f"[INFO]   '{TARGET_COLUMN}' (target) -> {len(target_le.classes_)} classes")
    return X_enc, y_enc, encoders


def _predict_season_champions(model, encoders, df, feature_cols):
    """
    Run ALL matches through the trained model.

    Predicted champion  = team predicted to win the most matches that season
                          (league-stage proxy — we cannot predict the final bracket).
    Actual champion     = from VERIFIED_SEASONS (official IPL records).
                          Falls back to final-match winner from the CSV when a
                          season is not yet in VERIFIED_SEASONS.
    """
    df = df.copy()
    X_all = df[feature_cols].copy()

    for col in feature_cols:
        le = encoders[col]
        class_map = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_all[col] = X_all[col].astype(str).map(class_map).fillna(0).astype(int)

    target_le    = encoders[TARGET_COLUMN]
    y_pred_enc   = model.predict(X_all)
    df["predicted_winner"] = target_le.inverse_transform(y_pred_enc)

    pred_champs = (
        df.groupby("season")["predicted_winner"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"predicted_winner": "predicted_champion"})
    )

    # Actual champion: VERIFIED_SEASONS first, CSV final-match fallback
    actual_rows = []
    for season in pred_champs["season"].tolist():
        verified = get_verified_champion(int(season))
        if verified:
            actual_rows.append({"season": season, "actual_champion": verified})
        else:
            # Fallback: winner of last match (highest id) in the season
            s_df   = df[df["season"] == season]
            winner = s_df.loc[s_df.index[-1], TARGET_COLUMN] if len(s_df) else "Unknown"
            actual_rows.append({"season": season, "actual_champion": winner})
    actual_champs = pd.DataFrame(actual_rows)

    comparison = pred_champs.merge(actual_champs, on="season").sort_values("season")

    # Pretty-print
    print("\n" + "=" * 80)
    print("Match Winner — Predicted Champion vs Actual IPL Champion per Season")
    print("=" * 80)
    print(f"{'Season':<10}{'Predicted Champion':<35}{'Actual IPL Champion'}")
    print("-" * 80)
    for _, row in comparison.iterrows():
        tick = "✓" if row["predicted_champion"] == row["actual_champion"] else " "
        print(f"  {int(row['season']):<8}{row['predicted_champion']:<35}{row['actual_champion']}  {tick}")
    print("=" * 80 + "\n")

    return comparison


def _save_feature_importance_chart(model, feature_cols, save_dir):
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue", edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Match Winner — Feature Importance")
    fig.tight_layout()
    out = os.path.join(save_dir, "match_winner_feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def train_match_winner_model(data_path, model_save_path):
    """
    Returns: model, encoders, accuracy, season_champions_df
    """
    # 1. Load
    csv_path = os.path.join(data_path, "cleaned_matches.csv")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows")

    # 2. Drop no-result matches
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].astype(str).str.strip() != ""]
    print(f"[INFO] {len(df):,} rows after dropping no-result matches")

    # 3. Features
    feature_cols = _detect_features(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # 4. Encode
    print("[INFO] Encoding categorical columns ...")
    X_enc, y_enc, encoders = _label_encode(X, y)

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.20, random_state=42
    )
    print(f"[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 6. Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[INFO] Training complete")

    # 7. Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Season-wise champion predictions vs VERIFIED_SEASONS
    season_champs = _predict_season_champions(model, encoders, df, feature_cols)

    # 9. Chart
    os.makedirs(model_save_path, exist_ok=True)
    chart = _save_feature_importance_chart(model, feature_cols, model_save_path)
    print(f"[INFO] Feature importance chart saved -> {chart}")

    # 10. Save
    pkl_path = os.path.join(model_save_path, MODEL_FILENAME)
    with open(pkl_path, "wb") as fh:
        pickle.dump({"model": model, "encoders": encoders}, fh)
    print(f"[INFO] Model saved -> {pkl_path}")

    return model, encoders, accuracy, season_champs


if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m, enc, acc, champs = train_match_winner_model(
        data_path       = os.path.join(_BASE, "data"),
        model_save_path = os.path.join(_BASE, "models"),
    )
    print(f"\n[DONE] Accuracy: {acc:.4f}")
