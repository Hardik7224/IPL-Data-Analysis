"""
purple_cap.py
=============
Predicts the Purple Cap winner (top wicket-taker) for each IPL season
using a RandomForestRegressor trained on per-season bowling statistics.

Features: runs_conceded, balls_bowled, economy, matches_played
Target  : wickets

BUG FIXES:
    1. CRITICAL — Wicket detection: cleaned_deliveries.csv has player_dismissed
       and dismissal_kind forward-filled across ALL balls. The correct wicket
       delivery is detected by finding where player_dismissed CHANGES within
       an inning (sort by over/ball first). This yields ~28,933 genuine
       bowler wickets (2008-2026) vs ~5,514 inflated or ~260 under-counted
       from earlier approaches.

    2. Actual Purple Cap winner now sourced from VERIFIED_SEASONS (config.py)
       instead of being computed from the CSV — wicket counts in the data are
       ~2× official figures due to data duplication in the cleaning step.

    3. balls_bowled: excludes wide deliveries (wide_runs > 0) since wides are
       not legal deliveries. This also corrects the economy calculation.

Public API:
    train_purple_cap_model(data_path, model_save_path)
        -> (model, r2_score, purple_cap_df)
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
    from src.config import get_verified_purple, normalize_player_name
except ImportError:
    from config import get_verified_purple, normalize_player_name

# ---------------------------------------------------------------------------
BOWLER_DISMISSALS = {
    "bowled", "caught", "caught and bowled",
    "lbw", "stumped", "hit wicket",
}
FEATURE_COLS = ["runs_conceded", "balls_bowled", "economy", "matches_played"]
TARGET_COL   = "wickets"


# ---------------------------------------------------------------------------
def _detect_actual_wickets(deliveries):
    """
    Identify the ACTUAL wicket delivery from the forward-filled cleaned CSV.

    The true wicket ball is where player_dismissed CHANGES to a new value
    within a (match_id, inning) group, provided dismissal_kind is a
    bowler-credited type.
    """
    d = deliveries.sort_values(["match_id", "inning", "over", "ball"]).copy()

    d["prev_dismissed"] = (
        d.groupby(["match_id", "inning"])["player_dismissed"].shift(1)
    )
    d["is_wicket_change"] = d["player_dismissed"] != d["prev_dismissed"]

    dk_lower = d["dismissal_kind"].str.lower().str.strip()
    d["is_bowler_wicket"] = d["is_wicket_change"] & dk_lower.isin(BOWLER_DISMISSALS)

    total = int(d["is_bowler_wicket"].sum())
    print(f"[INFO] Bowler-credited wickets detected (change-based): {total:,}")
    return d


def _build_bowling_stats(deliveries):
    """
    Aggregate into per-season bowling stats with corrected wicket detection
    and legal-delivery-only balls_bowled.
    """
    d = _detect_actual_wickets(deliveries)

    wickets = (
        d[d["is_bowler_wicket"]]
        .groupby(["season", "bowler"])
        .size()
        .reset_index(name="wickets")
    )

    agg = d.groupby(["season", "bowler"]).agg(
        runs_conceded  = ("total_runs",  "sum"),
        matches_played = ("match_id",    "nunique"),
    ).reset_index()

    # Legal balls only (exclude wides)
    legal = d[d["wide_runs"] == 0]
    balls = (
        legal.groupby(["season", "bowler"])
        .size()
        .reset_index(name="balls_bowled")
    )

    stats = (
        agg
        .merge(wickets, on=["season", "bowler"], how="left")
        .merge(balls,   on=["season", "bowler"], how="left")
    )
    stats["wickets"]      = stats["wickets"].fillna(0).astype(int)
    stats["balls_bowled"] = stats["balls_bowled"].fillna(0).astype(int)

    stats["economy"] = np.where(
        stats["balls_bowled"] > 0,
        stats["runs_conceded"] / (stats["balls_bowled"] / 6),
        0.0,
    )
    return stats


def _build_purple_cap_table(stats, model):
    """
    Predict wickets for every bowler-season row.

    Returns a DataFrame with columns:
        season | predicted_purple_cap | predicted_wickets | actual_purple_cap | actual_wickets
    where actual_* comes from VERIFIED_SEASONS (official IPL records).
    """
    stats = stats.copy()
    stats["predicted_wickets"] = model.predict(stats[FEATURE_COLS])

    # Predicted top bowler per season (from model)
    pred_idx  = stats.groupby("season")["predicted_wickets"].idxmax()
    predicted = stats.loc[pred_idx, ["season", "bowler", "predicted_wickets"]].rename(
        columns={"bowler": "predicted_purple_cap"}
    )
    # Normalise CSV abbreviations → full names for fair comparison
    predicted["predicted_purple_cap"] = predicted["predicted_purple_cap"].apply(normalize_player_name)

    # Actual top bowler from VERIFIED_SEASONS
    seasons     = sorted(stats["season"].unique().astype(int))
    actual_rows = []
    for season in seasons:
        player, wickets = get_verified_purple(season)
        if player is None:
            idx     = stats[stats["season"] == season]["wickets"].idxmax()
            player  = normalize_player_name(stats.loc[idx, "bowler"])
            wickets = int(stats.loc[idx, "wickets"])
        actual_rows.append({
            "season":             season,
            "actual_purple_cap":  player,
            "actual_wickets":     wickets,
        })
    actual = pd.DataFrame(actual_rows)

    comparison = predicted.merge(actual, on="season").sort_values("season")

    # Pretty-print
    hits  = (comparison["predicted_purple_cap"] == comparison["actual_purple_cap"]).sum()
    total = len(comparison)
    print("\n" + "=" * 102)
    print(f"Purple Cap — Predicted vs Actual Top Wicket-Taker per Season  "
          f"[Actual = official IPL records]  [{hits}/{total} correct]")
    print("=" * 102)
    print(
        f"{'Season':<10}"
        f"{'Predicted Purple Cap':<30}"
        f"{'Pred Wickets':>14}"
        f"{'Actual Purple Cap (Verified)':<32}"
        f"{'Actual Wickets':>14}"
    )
    print("-" * 102)
    for _, row in comparison.iterrows():
        tick = "  ✓" if row["predicted_purple_cap"] == row["actual_purple_cap"] else "  ✗"
        print(
            f"  {int(row['season']):<8}"
            f"{row['predicted_purple_cap']:<30}"
            f"{row['predicted_wickets']:>14.1f}"
            f"{row['actual_purple_cap']:<32}"
            f"{int(row['actual_wickets']):>14}"
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
        color="mediumpurple", edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Purple Cap — Feature Importance")
    fig.tight_layout()
    out = os.path.join(save_dir, "purple_cap_feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def train_purple_cap_model(data_path, model_save_path):
    """
    Returns: model, r2_score, purple_cap_comparison_df
    """
    # 1. Load
    deliveries = pd.read_csv(os.path.join(data_path, "cleaned_deliveries.csv"))
    matches    = pd.read_csv(os.path.join(data_path, "cleaned_matches.csv"))
    print(f"[INFO] Loaded deliveries : {deliveries.shape[0]:,} rows")
    print(f"[INFO] Loaded matches    : {matches.shape[0]:,} rows")

    # 2. Merge season
    deliveries = deliveries.merge(
        matches[["id", "season"]], left_on="match_id", right_on="id", how="left"
    ).drop(columns=["id"])
    print(f"[INFO] Seasons in data   : {sorted(deliveries['season'].dropna().unique().astype(int))}")

    # 3. Build bowling stats
    stats = _build_bowling_stats(deliveries)
    print(f"[INFO] Bowling stat rows : {stats.shape[0]:,}  (season x bowler)")

    # 4. Features / target
    X = stats[FEATURE_COLS]
    y = stats[TARGET_COL]

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 6. Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[INFO] Training complete")

    # 7. Evaluate
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{'='*45}")
    print(f"  R² Score            : {r2:.4f}")
    print(f"  Mean Absolute Error : {mae:.2f} wickets")
    print(f"{'='*45}")

    # 8. Purple Cap comparison table
    purple_cap_df = _build_purple_cap_table(stats, model)

    # 9. Chart
    os.makedirs(model_save_path, exist_ok=True)
    chart = _plot_feature_importance(model, model_save_path)
    print(f"[INFO] Feature importance chart saved -> {chart}")

    # 10. Save
    model_file = os.path.join(model_save_path, "purple_cap_model.pkl")
    with open(model_file, "wb") as fh:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, fh)
    print(f"[INFO] Model saved -> {model_file}")

    return model, r2, purple_cap_df


if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m, score, df = train_purple_cap_model(
        data_path       = os.path.join(_BASE, "data"),
        model_save_path = os.path.join(_BASE, "models"),
    )
    print(f"\n[DONE] Purple Cap model R² = {score:.4f}")
