"""
=============================================================================
IPL Data Analysis — Machine Learning Prediction Pipeline
=============================================================================
Usage:
    python main.py

Datasets  -> data/cleaned_matches.csv  &  data/cleaned_deliveries.csv
Models    -> models/*.pkl
Charts    -> models/*_feature_importance.png

Ground-truth "actual" values for all three models come from VERIFIED_SEASONS
in src/config.py (official IPL records, 2008-2026). This decouples evaluation
accuracy from data-quality issues in the cleaned CSV files.
=============================================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.match_winner import train_match_winner_model
from src.orange_cap   import train_orange_cap_model
from src.purple_cap   import train_purple_cap_model
from src.config       import VERIFIED_SEASONS, get_verified_orange, get_verified_purple, get_verified_champion, normalize_player_name


def _check_datasets():
    missing = [
        f for f in ["cleaned_matches.csv", "cleaned_deliveries.csv"]
        if not os.path.isfile(os.path.join(DATA_DIR, f))
    ]
    if missing:
        print(f"\n[ERROR] Missing datasets in data/: {missing}")
        sys.exit(1)


def _print_final_summary(winner_acc, winner_champs, oc_r2, oc_df, pc_r2, pc_df):
    """Print a clean consolidated results table for all three models."""

    latest_season = int(oc_df["season"].max())

    # Latest-season rows
    lw = winner_champs[winner_champs["season"] == latest_season]
    lo = oc_df[oc_df["season"] == latest_season]
    lp = pc_df[pc_df["season"] == latest_season]

    pred_champion   = normalize_player_name(lw["predicted_champion"].values[0]) if len(lw) else "N/A"
    actual_champion = get_verified_champion(latest_season) or "N/A"

    pred_oc        = normalize_player_name(lo["predicted_orange_cap"].values[0]) if len(lo) else "N/A"
    pred_oc_runs   = int(lo["predicted_runs"].values[0])  if len(lo) else 0
    v_oc, v_oc_r   = get_verified_orange(latest_season)
    actual_oc      = v_oc  or "N/A"
    actual_oc_runs = v_oc_r or 0

    pred_pc        = normalize_player_name(lp["predicted_purple_cap"].values[0])  if len(lp) else "N/A"
    pred_pc_wk     = lp["predicted_wickets"].values[0]     if len(lp) else 0
    v_pc, v_pc_w   = get_verified_purple(latest_season)
    actual_pc      = v_pc  or "N/A"
    actual_pc_wk   = v_pc_w or 0

    W = 76

    print("\n" + "=" * W)
    print("  FINAL RESULTS SUMMARY")
    print("=" * W)

    # ── Model scores ──────────────────────────────────────────────────────────
    print(f"\n  MODEL PERFORMANCE")
    print(f"  {'-'*52}")
    print(f"  {'Match Winner':<24} Accuracy  : {winner_acc:.4f}  ({winner_acc*100:.2f}%)")
    print(f"  {'Orange Cap':<24} R² Score  : {oc_r2:.4f}")
    print(f"  {'Purple Cap':<24} R² Score  : {pc_r2:.4f}")

    # ── Latest season spotlight ───────────────────────────────────────────────
    print(f"\n  LATEST SEASON — {latest_season}")
    print(f"  {'-'*52}")

    w_tick = "✓" if pred_champion   == actual_champion else "✗"
    o_tick = "✓" if pred_oc         == actual_oc       else "✗"
    p_tick = "✓" if pred_pc         == actual_pc       else "✗"

    print(f"\n  [MATCH WINNER]")
    print(f"    Predicted Champion          : {pred_champion}")
    print(f"    Actual IPL Champion         : {actual_champion}  {w_tick}")

    print(f"\n  [ORANGE CAP — Top Run Scorer]")
    print(f"    Predicted Player (runs)     : {pred_oc}  ({pred_oc_runs})")
    print(f"    Actual Player (Verified)    : {actual_oc}  ({actual_oc_runs})  {o_tick}")

    print(f"\n  [PURPLE CAP — Top Wicket Taker]")
    print(f"    Predicted Player (wickets)  : {pred_pc}  ({pred_pc_wk:.0f})")
    print(f"    Actual Player (Verified)    : {actual_pc}  ({actual_pc_wk})  {p_tick}")

    # ── All seasons — Orange + Purple ─────────────────────────────────────────
    print(f"\n  ALL SEASONS — PREDICTED vs VERIFIED ORANGE & PURPLE CAP")
    print(f"  {'-'*72}")
    print(f"  {'Season':<8}{'Pred OC':<26}{'Actual OC (Verified)':<26}{'Pred PC':<26}{'Actual PC (Verified)'}")
    print(f"  {'-'*72}")
    seasons = sorted(oc_df["season"].unique().astype(int))
    for s in seasons:
        or_ = oc_df[oc_df["season"] == s]
        pc_ = pc_df[pc_df["season"] == s]
        p_oc = or_["predicted_orange_cap"].values[0] if len(or_) else "N/A"
        a_oc, _ = get_verified_orange(s)
        a_oc = a_oc or (or_["actual_orange_cap"].values[0] if len(or_) else "N/A")
        p_pc = pc_["predicted_purple_cap"].values[0] if len(pc_) else "N/A"
        a_pc, _ = get_verified_purple(s)
        a_pc = a_pc or (pc_["actual_purple_cap"].values[0] if len(pc_) else "N/A")
        o_t = "✓" if p_oc == a_oc else " "
        p_t = "✓" if p_pc == a_pc else " "
        print(f"  {s:<8}{p_oc:<26}{a_oc:<26}{p_pc:<26}{a_pc} {p_t}")

    # ── All seasons — Match Winner ─────────────────────────────────────────────
    print(f"\n  ALL SEASONS — PREDICTED vs VERIFIED IPL CHAMPION")
    print(f"  {'-'*72}")
    print(f"  {'Season':<8}{'Predicted Champion':<36}{'Actual Champion (Verified)'}")
    print(f"  {'-'*72}")
    for _, row in winner_champs.sort_values("season").iterrows():
        verified = get_verified_champion(int(row["season"])) or row["actual_champion"]
        tick = " ✓" if row["predicted_champion"] == verified else ""
        print(f"  {int(row['season']):<8}{row['predicted_champion']:<36}{verified}{tick}")

    print("\n" + "=" * W)
    print(f"  Models    : {MODEL_DIR}")
    print(f"  Data      : {DATA_DIR}")
    print(f"  Verified  : src/config.py  (VERIFIED_SEASONS)")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
def main():
    _check_datasets()

    # ── Model 1: Match Winner ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL 1 — IPL MATCH WINNER PREDICTION")
    print("=" * 70)
    try:
        _, _, winner_acc, winner_champs = train_match_winner_model(DATA_DIR, MODEL_DIR)
        print(f"\n[✓] Match Winner complete  |  Accuracy: {winner_acc:.4f}")
    except Exception as e:
        print(f"\n[✗] Match Winner FAILED: {e}")
        raise

    # ── Model 2: Orange Cap ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL 2 — ORANGE CAP PREDICTION  (Top Run Scorer)")
    print("=" * 70)
    try:
        _, oc_r2, oc_df = train_orange_cap_model(DATA_DIR, MODEL_DIR)
        print(f"\n[✓] Orange Cap complete  |  R²: {oc_r2:.4f}")
    except Exception as e:
        print(f"\n[✗] Orange Cap FAILED: {e}")
        raise

    # ── Model 3: Purple Cap ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL 3 — PURPLE CAP PREDICTION  (Top Wicket Taker)")
    print("=" * 70)
    try:
        _, pc_r2, pc_df = train_purple_cap_model(DATA_DIR, MODEL_DIR)
        print(f"\n[✓] Purple Cap complete  |  R²: {pc_r2:.4f}")
    except Exception as e:
        print(f"\n[✗] Purple Cap FAILED: {e}")
        raise

    _print_final_summary(winner_acc, winner_champs, oc_r2, oc_df, pc_r2, pc_df)


if __name__ == "__main__":
    main()
