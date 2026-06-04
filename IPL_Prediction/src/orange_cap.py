"""
orange_cap.py
=============
Predicts the Orange Cap winner (top run-scorer) for each IPL season
using a RandomForestRegressor trained on per-season batting statistics.

Features: matches_played, balls_faced, strike_rate, fours, sixes
Target  : total_runs

BUG FIXES:
    1. CRITICAL — batsman_runs column is all zeros in cleaned_deliveries.csv.
       FIX: reconstruct as total_runs − (wide + bye + legbye + noball + penalty
       runs), clipped at 0.

    2. Actual Orange Cap winner now sourced from VERIFIED_SEASONS (config.py)
       instead of being computed from the CSV data (which has mis-attributed
       runs, e.g. GC Smith shown as 2008 leader instead of Shaun Marsh).

    3. balls_faced: exclude only wide deliveries (wide_runs > 0). No-balls are
       legal deliveries and count toward balls faced.

    4. Season merge: validates 'id' key exists before merging.

Public API:
    train_orange_cap_model(data_path, model_save_path)
        -> (model, r2_score, orange_cap_df)
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ    = os.path.dirname(_SRC_DIR)
for _p in (_PROJ, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.config import get_verified_orange, normalize_player_name
except ImportError:
    from config import get_verified_orange, normalize_player_name

# ---------------------------------------------------------------------------
FEATURE_COLS = ["matches_played", "balls_faced", "strike_rate", "fours", "sixes"]
TARGET_COL   = "total_runs"


# ---------------------------------------------------------------------------
def _get_batter_col(df):
    for name in ("batsman", "batter"):
        if name in df.columns:
            return name
    raise KeyError(
        f"No batter column found in deliveries. Available: {list(df.columns)}"
    )


def _fix_batsman_runs(deliveries):
    """
    Reconstruct batsman_runs when the column is zeroed out (data-cleaning bug).
    Formula: total_runs − (wide_runs + bye_runs + legbye_runs + noball_runs + penalty_runs)
    Clipped at 0 for edge-case extras.
    """
    EXTRA_COLS = ["wide_runs", "bye_runs", "legbye_runs", "noball_runs", "penalty_runs"]
    available  = [c for c in EXTRA_COLS if c in deliveries.columns]

    if deliveries["batsman_runs"].sum() == 0 and available:
        print("[WARN] batsman_runs is all zeros — reconstructing from total_runs - extras")
        deliveries = deliveries.copy()
        deliveries["batsman_runs"] = (
            deliveries["total_runs"] - deliveries[available].sum(axis=1)
        ).clip(lower=0)
        print(f"[INFO] Reconstructed batsman_runs total : {int(deliveries['batsman_runs'].sum()):,}")
    return deliveries


def _build_batting_stats(deliveries, batter_col):
    """Aggregate delivery-level data into per-season batting stats."""
    deliveries = _fix_batsman_runs(deliveries)

    grouped = deliveries.groupby(["season", batter_col]).agg(
        total_runs    = ("batsman_runs", "sum"),
        matches_played= ("match_id",     "nunique"),
    ).reset_index().rename(columns={batter_col: "batter"})

    # Balls faced = legal deliveries (exclude wides; no-balls are legal)
    legal = deliveries[deliveries["wide_runs"] == 0]
    balls = (
        legal.groupby(["season", batter_col])
        .size()
        .reset_index(name="balls_faced")
        .rename(columns={batter_col: "batter"})
    )

    fours = (
        deliveries[deliveries["batsman_runs"] == 4]
        .groupby(["season", batter_col]).size()
        .reset_index(name="fours")
        .rename(columns={batter_col: "batter"})
    )
    sixes = (
        deliveries[deliveries["batsman_runs"] == 6]
        .groupby(["season", batter_col]).size()
        .reset_index(name="sixes")
        .rename(columns={batter_col: "batter"})
    )

    stats = (
        grouped
        .merge(balls, on=["season", "batter"], how="left")
        .merge(fours, on=["season", "batter"], how="left")
        .merge(sixes, on=["season", "batter"], how="left")
    )
    stats[["balls_faced", "fours", "sixes"]] = (
        stats[["balls_faced", "fours", "sixes"]].fillna(0).astype(int)
    )
    stats["strike_rate"] = np.where(
        stats["balls_faced"] > 0,
        (stats["total_runs"] / stats["balls_faced"]) * 100,
        0.0,
    )
    return stats


def _build_orange_cap_table(stats, model):
    """
    Predict runs for every batter-season row.

    Returns a DataFrame with columns:
        season | predicted_orange_cap | predicted_runs | actual_orange_cap | actual_runs
    where actual_* comes from VERIFIED_SEASONS (official IPL records).
    """
    stats = stats.copy()
    stats["predicted_runs"] = model.predict(stats[FEATURE_COLS])

    # Predicted top scorer per season (from model output)
    pred_idx  = stats.groupby("season")["predicted_runs"].idxmax()
    predicted = stats.loc[pred_idx, ["season", "batter", "predicted_runs"]].rename(
        columns={"batter": "predicted_orange_cap"}
    )
    # Normalise CSV abbreviations → full names for fair comparison
    predicted["predicted_orange_cap"] = predicted["predicted_orange_cap"].apply(normalize_player_name)

    # Actual top scorer per season from VERIFIED_SEASONS
    seasons      = sorted(stats["season"].unique().astype(int))
    actual_rows  = []
    for season in seasons:
        player, runs = get_verified_orange(season)
        if player is None:
            idx    = stats[stats["season"] == season]["total_runs"].idxmax()
            player = normalize_player_name(stats.loc[idx, "batter"])
            runs   = int(stats.loc[idx, "total_runs"])
        actual_rows.append({
            "season":            season,
            "actual_orange_cap": player,
            "actual_runs":       runs,
        })
    actual = pd.DataFrame(actual_rows)

    comparison = predicted.merge(actual, on="season").sort_values("season")

    # Pretty-print
    hits = (comparison["predicted_orange_cap"] == comparison["actual_orange_cap"]).sum()
    total = len(comparison)
    print("\n" + "=" * 102)
    print(f"Orange Cap — Predicted vs Actual Top Run-Scorer per Season  "
          f"[Actual = official IPL records]  [{hits}/{total} correct]")
    print("=" * 102)
    print(
        f"{'Season':<10}"
        f"{'Predicted Orange Cap':<30}"
        f"{'Pred Runs':>12}"
        f"{'Actual Orange Cap (Verified)':<32}"
        f"{'Actual Runs':>12}"
    )
    print("-" * 102)
    for _, row in comparison.iterrows():
        tick = "  ✓" if row["predicted_orange_cap"] == row["actual_orange_cap"] else "  ✗"
        print(
            f"  {int(row['season']):<8}"
            f"{row['predicted_orange_cap']:<30}"
            f"{row['predicted_runs']:>12.0f}"
            f"{row['actual_orange_cap']:<32}"
            f"{int(row['actual_runs']):>12}"
            f"{tick}"
        )
    print("=" * 102 + "\n")
    return comparison


def _plot_feature_importance(model, save_dir):
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_COLS[i] for i in indices],
        importances[indices],
        color="darkorange", edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Orange Cap — Feature Importance")
    fig.tight_layout()
    out = os.path.join(save_dir, "orange_cap_feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def train_orange_cap_model(data_path, model_save_path):
    """
    Returns: model, r2_score, orange_cap_comparison_df
    """
    # 1. Load
    deliveries = pd.read_csv(os.path.join(data_path, "cleaned_deliveries.csv"))
    matches    = pd.read_csv(os.path.join(data_path, "cleaned_matches.csv"))
    print(f"[INFO] Loaded deliveries : {deliveries.shape[0]:,} rows")
    print(f"[INFO] Loaded matches    : {matches.shape[0]:,} rows")

    # 2. Auto-detect batter column
    batter_col = _get_batter_col(deliveries)
    print(f"[INFO] Batter column     : '{batter_col}'")

    # 3. Merge season
    if "id" not in matches.columns:
        raise KeyError(f"'id' column not found in matches. Got: {list(matches.columns)}")
    deliveries = deliveries.merge(
        matches[["id", "season"]], left_on="match_id", right_on="id", how="left"
    ).drop(columns=["id"])
    print(f"[INFO] Seasons in data   : {sorted(deliveries['season'].dropna().unique().astype(int))}")

    # 4. Build stats
    stats = _build_batting_stats(deliveries, batter_col)
    print(f"[INFO] Batting stat rows : {stats.shape[0]:,}  (season x batter)")

    # 5. Features / target
    X = stats[FEATURE_COLS]
    y = stats[TARGET_COL]

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 7. Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[INFO] Training complete")

    # 8. Evaluate
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{'='*45}")
    print(f"  R² Score            : {r2:.4f}")
    print(f"  Mean Absolute Error : {mae:.2f} runs")
    print(f"{'='*45}")

    # 9. Orange Cap comparison table
    orange_cap_df = _build_orange_cap_table(stats, model)

    # 10. Chart
    os.makedirs(model_save_path, exist_ok=True)
    chart = _plot_feature_importance(model, model_save_path)
    print(f"[INFO] Feature importance chart saved -> {chart}")

    # 11. Save
    model_file = os.path.join(model_save_path, "orange_cap_model.pkl")
    with open(model_file, "wb") as fh:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, fh)
    print(f"[INFO] Model saved -> {model_file}")

    return model, r2, orange_cap_df


if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m, score, df = train_orange_cap_model(
        data_path       = os.path.join(_BASE, "data"),
        model_save_path = os.path.join(_BASE, "models"),
    )
    print(f"\n[DONE] Orange Cap model R² = {score:.4f}")
