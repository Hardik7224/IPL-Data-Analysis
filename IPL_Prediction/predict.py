"""
predict.py
==========
Interactive IPL Match Winner Predictor.

Loads the trained RandomForest model and encoders from models/winner_model.pkl,
asks the user to enter two teams (and optional details), then predicts the
winner along with win-probability for each team.

Usage:
    python predict.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "winner_model.pkl")

# ---------------------------------------------------------------------------
# Team → default home city & venue (auto-fallback when user skips)
# ---------------------------------------------------------------------------
TEAM_HOME_DEFAULTS = {
    "Chennai Super Kings"       : ("Chennai",    "MA Chidambaram Stadium, Chepauk"),
    "Deccan Chargers"           : ("Hyderabad",  "Rajiv Gandhi International Stadium, Uppal"),
    "Delhi Capitals"            : ("Delhi",      "Feroz Shah Kotla Ground"),
    "Delhi Daredevils"          : ("Delhi",      "Feroz Shah Kotla"),
    "Gujarat Lions"             : ("Rajkot",     "Saurashtra Cricket Association Stadium"),
    "Kings XI Punjab"           : ("Chandigarh", "Punjab Cricket Association Stadium, Mohali"),
    "Kochi Tuskers Kerala"      : ("Kochi",      "Nehru Stadium"),
    "Kolkata Knight Riders"     : ("Kolkata",    "Eden Gardens"),
    "Mumbai Indians"            : ("Mumbai",     "Wankhede Stadium"),
    "Pune Warriors"             : ("Mumbai",     "Subrata Roy Sahara Stadium"),
    "Rajasthan Royals"          : ("Jaipur",     "Sawai Mansingh Stadium"),
    "Rising Pune Supergiant"    : ("Mumbai",     "Wankhede Stadium"),
    "Rising Pune Supergiants"   : ("Pune",       "Maharashtra Cricket Association Stadium"),
    "Royal Challengers Bangalore": ("Bangalore", "M Chinnaswamy Stadium"),
    "Sunrisers Hyderabad"       : ("Hyderabad",  "Rajiv Gandhi International Stadium, Uppal"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model():
    """Load model bundle from disk."""
    if not os.path.isfile(MODEL_PATH):
        print(f"\n[ERROR] Model file not found: {MODEL_PATH}")
        print("        Please run  python main.py  first to train the model.")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)

    model    = bundle["model"]
    encoders = bundle["encoders"]
    return model, encoders


def _known_teams(encoders):
    """Return sorted list of teams the model knows about."""
    return sorted(encoders["team1"].classes_)


def _pick_team(prompt, known_teams, exclude=None):
    """
    Prompt the user to type a team name with fuzzy-match fallback.
    Re-prompts until a valid team is chosen.
    """
    # Build display list (exclude already-chosen team if given)
    display = [t for t in known_teams if t != exclude]

    while True:
        print(f"\n  Available teams:")
        for i, t in enumerate(display, 1):
            print(f"    {i:>2}. {t}")

        raw = input(f"\n  {prompt} (name or number): ").strip()

        # Numeric selection
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(display):
                return display[idx]
            print("  [!] Invalid number. Try again.")
            continue

        # Exact match (case-insensitive)
        lower_map = {t.lower(): t for t in display}
        if raw.lower() in lower_map:
            return lower_map[raw.lower()]

        # Partial / fuzzy match
        matches = [t for t in display if raw.lower() in t.lower()]
        if len(matches) == 1:
            print(f"  [✓] Matched: {matches[0]}")
            return matches[0]
        if len(matches) > 1:
            print(f"  [!] Multiple matches: {matches}. Be more specific.")
            continue

        print(f"  [!] '{raw}' not recognised. Try typing part of the name.")


def _encode_value(col, value, encoders):
    """
    Encode a single value using the fitted LabelEncoder.
    If the value is unseen, use the closest known class (index 0 as fallback).
    """
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    # Fallback: use first class (model will still run, just less accurate)
    return 0


def _predict_winner(model, encoders, team1, team2, city, venue, toss_winner, toss_decision, season):
    """
    Build the feature row, run the model, return predicted winner + probabilities.
    """
    feature_row = {
        "season"       : _encode_value("season",        str(season),       encoders),
        "team1"        : _encode_value("team1",         team1,             encoders),
        "team2"        : _encode_value("team2",         team2,             encoders),
        "city"         : _encode_value("city",          city,              encoders),
        "venue"        : _encode_value("venue",         venue,             encoders),
        "toss_winner"  : _encode_value("toss_winner",   toss_winner,       encoders),
        "toss_decision": _encode_value("toss_decision", toss_decision,     encoders),
    }

    X = pd.DataFrame([feature_row])
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    # Decode prediction
    winner = encoders["winner"].inverse_transform([pred_enc])[0]

    # Build probability map for teams that appear in model classes
    target_classes = encoders["winner"].classes_
    proba_map      = dict(zip(target_classes, pred_proba))

    team1_prob = proba_map.get(team1, 0.0) * 100
    team2_prob = proba_map.get(team2, 0.0) * 100

    # Normalise the two teams' probabilities so they sum to 100 %
    total = team1_prob + team2_prob
    if total > 0:
        team1_prob = (team1_prob / total) * 100
        team2_prob = (team2_prob / total) * 100
    else:
        team1_prob = team2_prob = 50.0

    return winner, team1_prob, team2_prob


def _bar(prob, width=30):
    """Return a simple ASCII progress bar for a probability (0–100)."""
    filled = int(round(prob / 100 * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def run_predictor():
    print("\n" + "=" * 60)
    print("   IPL MATCH WINNER PREDICTOR")
    print("=" * 60)

    # Load model once
    model, encoders = _load_model()
    known_teams      = _known_teams(encoders)
    latest_season    = max(int(s) for s in encoders["season"].classes_)

    print("\n  Model loaded successfully.")
    print(f"  Trained on seasons : 2008 – {latest_season}")
    print(f"  Teams in model     : {len(known_teams)}")

    while True:
        print("\n" + "-" * 60)
        print("  ENTER MATCH DETAILS")
        print("-" * 60)

        # ── Step 1: Teams ─────────────────────────────────────────────────────
        team1 = _pick_team("Enter Team 1", known_teams)
        team2 = _pick_team("Enter Team 2", known_teams, exclude=team1)

        # ── Step 2: Toss winner ───────────────────────────────────────────────
        print(f"\n  Who won the toss?")
        print(f"    1. {team1}")
        print(f"    2. {team2}")
        while True:
            toss_choice = input("  Enter 1 or 2: ").strip()
            if toss_choice == "1":
                toss_winner = team1; break
            elif toss_choice == "2":
                toss_winner = team2; break
            print("  [!] Please enter 1 or 2.")

        # ── Step 3: Toss decision ─────────────────────────────────────────────
        print(f"\n  Toss decision?")
        print(f"    1. Bat")
        print(f"    2. Field")
        while True:
            td_choice = input("  Enter 1 or 2: ").strip()
            if td_choice == "1":
                toss_decision = "bat";   break
            elif td_choice == "2":
                toss_decision = "field"; break
            print("  [!] Please enter 1 or 2.")

        # ── Step 4: Season (optional) ─────────────────────────────────────────
        season_input = input(f"\n  Season year [press Enter for {latest_season}]: ").strip()
        season = int(season_input) if season_input.isdigit() else latest_season

        # ── Step 5: City & Venue (auto-default to team1's home ground) ────────
        default_city, default_venue = TEAM_HOME_DEFAULTS.get(
            team1, ("Mumbai", "Wankhede Stadium")
        )
        city_input  = input(f"  City   [press Enter for '{default_city}']: ").strip()
        venue_input = input(f"  Venue  [press Enter for '{default_venue}']: ").strip()

        city  = city_input  if city_input  else default_city
        venue = venue_input if venue_input else default_venue

        # ── Predict ───────────────────────────────────────────────────────────
        winner, t1_prob, t2_prob = _predict_winner(
            model, encoders,
            team1, team2, city, venue,
            toss_winner, toss_decision, season
        )

        # ── Display result ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  PREDICTION RESULT")
        print("=" * 60)
        print(f"\n  {team1}")
        print(f"  {_bar(t1_prob)}  {t1_prob:.1f} %")
        print()
        print(f"  {team2}")
        print(f"  {_bar(t2_prob)}  {t2_prob:.1f} %")
        print()
        print(f"  {'─'*56}")
        print(f"  🏆  Predicted Winner : {winner}")
        print(f"  {'─'*56}")
        print(f"\n  Match details used:")
        print(f"    Season        : {season}")
        print(f"    City          : {city}")
        print(f"    Venue         : {venue}")
        print(f"    Toss Winner   : {toss_winner}")
        print(f"    Toss Decision : {toss_decision}")
        print("=" * 60)

        # ── Ask to predict again ──────────────────────────────────────────────
        again = input("\n  Predict another match? (y / n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye! 🏏\n")
            break


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_predictor()
