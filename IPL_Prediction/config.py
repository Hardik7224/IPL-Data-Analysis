"""
config.py
=========
Shared constants for the IPL Prediction pipeline.

VERIFIED_SEASONS  – Official IPL records 2008-2026 (ground-truth actuals).
PLAYER_ALIASES    – Maps Cricinfo-style abbreviations used in the CSV data
                    to the full player names used in VERIFIED_SEASONS.
                    Required because the cleaned CSV stores names as
                    "V Kohli", "B Kumar", "SL Malinga" etc., while official
                    records use "Virat Kohli", "Bhuvneshwar Kumar" etc.
                    Without this mapping, correct predictions are wrongly
                    counted as misses.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Official IPL results (2008–2026)
# ─────────────────────────────────────────────────────────────────────────────
VERIFIED_SEASONS = {
    2008: {"orange": "Shaun Marsh (KXIP, 616)",         "purple": "Sohail Tanvir (RR, 22)",           "champion": "Rajasthan Royals"},
    2009: {"orange": "Matthew Hayden (CSK, 572)",        "purple": "RP Singh (DC, 23)",                "champion": "Deccan Chargers"},
    2010: {"orange": "Sachin Tendulkar (MI, 618)",       "purple": "Pragyan Ojha (DC, 21)",            "champion": "Chennai Super Kings"},
    2011: {"orange": "Chris Gayle (RCB, 608)",           "purple": "Lasith Malinga (MI, 28)",          "champion": "Chennai Super Kings"},
    2012: {"orange": "Chris Gayle (RCB, 733)",           "purple": "Morne Morkel (DD, 25)",            "champion": "Kolkata Knight Riders"},
    2013: {"orange": "Michael Hussey (CSK, 733)",        "purple": "Dwayne Bravo (CSK, 32)",           "champion": "Mumbai Indians"},
    2014: {"orange": "Robin Uthappa (KKR, 660)",         "purple": "Mohit Sharma (CSK, 23)",           "champion": "Kolkata Knight Riders"},
    2015: {"orange": "David Warner (SRH, 562)",          "purple": "Dwayne Bravo (CSK, 26)",           "champion": "Mumbai Indians"},
    2016: {"orange": "Virat Kohli (RCB, 973)",           "purple": "Bhuvneshwar Kumar (SRH, 23)",      "champion": "Sunrisers Hyderabad"},
    2017: {"orange": "David Warner (SRH, 641)",          "purple": "Bhuvneshwar Kumar (SRH, 26)",      "champion": "Mumbai Indians"},
    2018: {"orange": "Kane Williamson (SRH, 735)",       "purple": "Andrew Tye (KXIP, 24)",            "champion": "Chennai Super Kings"},
    2019: {"orange": "David Warner (SRH, 692)",          "purple": "Imran Tahir (CSK, 26)",            "champion": "Mumbai Indians"},
    2020: {"orange": "KL Rahul (KXIP, 670)",             "purple": "Kagiso Rabada (DC, 30)",           "champion": "Mumbai Indians"},
    2021: {"orange": "Ruturaj Gaikwad (CSK, 635)",       "purple": "Harshal Patel (RCB, 32)",          "champion": "Chennai Super Kings"},
    2022: {"orange": "Jos Buttler (RR, 863)",            "purple": "Yuzvendra Chahal (RR, 27)",        "champion": "Gujarat Titans"},
    2023: {"orange": "Shubman Gill (GT, 890)",           "purple": "Mohammed Shami (GT, 28)",          "champion": "Chennai Super Kings"},
    2024: {"orange": "Virat Kohli (RCB, 741)",           "purple": "Harshal Patel (PBKS, 24)",         "champion": "Kolkata Knight Riders"},
    2025: {"orange": "Sai Sudharsan (GT, 759)",          "purple": "Prasidh Krishna (GT, 25)",         "champion": "Royal Challengers Bengaluru"},
    2026: {"orange": "Vaibhav Suryavanshi (RR, 776)",   "purple": "Kagiso Rabada (GT, 29)",           "champion": "Royal Challengers Bengaluru"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Cricinfo CSV abbreviation → full name as used in VERIFIED_SEASONS
# ─────────────────────────────────────────────────────────────────────────────
PLAYER_ALIASES = {
    # ── Batters (Orange Cap) ──────────────────────────────────────────────────
    "SE Marsh":          "Shaun Marsh",
    "ML Hayden":         "Matthew Hayden",
    "SR Tendulkar":      "Sachin Tendulkar",
    "CH Gayle":          "Chris Gayle",
    "MEK Hussey":        "Michael Hussey",
    "RV Uthappa":        "Robin Uthappa",
    "DA Warner":         "David Warner",
    "V Kohli":           "Virat Kohli",
    "KS Williamson":     "Kane Williamson",
    "KL Rahul":          "KL Rahul",           # already same; kept for clarity
    "RD Gaikwad":        "Ruturaj Gaikwad",
    "JC Buttler":        "Jos Buttler",
    "Shubman Gill":      "Shubman Gill",        # already same
    "B Sai Sudharsan":   "Sai Sudharsan",
    "V Suryavanshi":     "Vaibhav Suryavanshi",

    # ── Bowlers (Purple Cap) ──────────────────────────────────────────────────
    "Sohail Tanvir":     "Sohail Tanvir",       # already same
    "RP Singh":          "RP Singh",            # already same
    "PP Ojha":           "Pragyan Ojha",
    "SL Malinga":        "Lasith Malinga",
    "M Morkel":          "Morne Morkel",
    "DJ Bravo":          "Dwayne Bravo",
    "MM Sharma":         "Mohit Sharma",
    "B Kumar":           "Bhuvneshwar Kumar",
    "AJ Tye":            "Andrew Tye",
    "Imran Tahir":       "Imran Tahir",         # already same
    "K Rabada":          "Kagiso Rabada",
    "HV Patel":          "Harshal Patel",
    "YS Chahal":         "Yuzvendra Chahal",
    "Mohammed Shami":    "Mohammed Shami",      # already same
    "M Prasidh Krishna": "Prasidh Krishna",
    "Arshdeep Singh":    "Arshdeep Singh",      # already same
    "Mohammed Siraj":    "Mohammed Siraj",      # already same
    "T Boult":           "Trent Boult",
    "TA Boult":          "Trent Boult",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_RE_PLAYER = re.compile(r"^(.+?)\s*\([^,]+,\s*(\d+)\)$")


def normalize_player_name(name: str) -> str:
    """
    Resolve a CSV-style abbreviation (e.g. 'V Kohli') to the full name used
    in VERIFIED_SEASONS (e.g. 'Virat Kohli').
    Returns the original name unchanged if no alias is defined.
    """
    return PLAYER_ALIASES.get(name.strip(), name.strip())


def parse_player_entry(entry: str):
    """
    Parse 'Player Name (TEAM, stat)' → (player_name: str, stat: int).
    Returns (entry.strip(), None) if the string doesn't match.
    """
    m = _RE_PLAYER.match(entry.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return entry.strip(), None


def get_verified_orange(season: int):
    """Return (player_name, runs) for the season, or (None, None)."""
    entry = VERIFIED_SEASONS.get(season, {}).get("orange")
    return parse_player_entry(entry) if entry else (None, None)


def get_verified_purple(season: int):
    """Return (player_name, wickets) for the season, or (None, None)."""
    entry = VERIFIED_SEASONS.get(season, {}).get("purple")
    return parse_player_entry(entry) if entry else (None, None)


def get_verified_champion(season: int):
    """Return the IPL champion team name for the season, or None."""
    return VERIFIED_SEASONS.get(season, {}).get("champion")
