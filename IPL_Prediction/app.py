"""
app.py — IPL Prediction Dashboard (Streamlit)
Run with: streamlit run app.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="IPL Prediction Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# IPL 2026 ACTUAL RESULTS  (season is over — hardcoded from official records)
# ─────────────────────────────────────────────────────────────────────────────
IPL_2026 = {
    "season"         : 2026,
    "champion"       : "Royal Challengers Bengaluru",
    "runner_up"      : "Gujarat Titans",
    "orange_cap"     : "Vaibhav Suryavanshi",
    "orange_team"    : "Rajasthan Royals",
    "orange_runs"    : 776,
    "orange_matches" : 16,
    "orange_sr"      : 237.30,
    "purple_cap"     : "Kagiso Rabada",
    "purple_team"    : "Gujarat Titans",
    "purple_wickets" : 29,
    "purple_matches" : 17,
    "purple_economy" : 9.68,
    # Top-5 Orange Cap
    "oc_top5": [
        ("Vaibhav Suryavanshi", "Rajasthan Royals",       776),
        ("Shubman Gill",        "Gujarat Titans",          732),
        ("Sai Sudharsan",       "Gujarat Titans",          722),
        ("Virat Kohli",         "Royal Challengers Bengaluru", 675),
        ("Heinrich Klaasen",    "Sunrisers Hyderabad",    640),
    ],
    # Top-5 Purple Cap
    "pc_top5": [
        ("Kagiso Rabada",       "Gujarat Titans",          29),
        ("Bhuvneshwar Kumar",   "Royal Challengers Bengaluru", 28),
        ("Jofra Archer",        "Rajasthan Royals",        25),
        ("Rashid Khan",         "Gujarat Titans",          21),
        ("Anshul Kamboj",       "Chennai Super Kings",     19),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# TEAM DATA
# ─────────────────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Mumbai Indians"                    : "#005DA0",
    "Chennai Super Kings"               : "#F5A623",
    "Royal Challengers Bangalore"       : "#D01C1F",
    "Royal Challengers Bengaluru"       : "#D01C1F",
    "Kolkata Knight Riders"             : "#3A1F6E",
    "Sunrisers Hyderabad"               : "#FF6B00",
    "Delhi Capitals"                    : "#00AAE5",
    "Delhi Daredevils"                  : "#00AAE5",
    "Kings XI Punjab"                   : "#AA3035",
    "Punjab Kings"                      : "#AA3035",
    "Rajasthan Royals"                  : "#EA1A7F",
    "Deccan Chargers"                   : "#F5A623",
    "Pune Warriors"                     : "#1B4D8E",
    "Rising Pune Supergiant"            : "#6F2DA8",
    "Rising Pune Supergiants"           : "#6F2DA8",
    "Gujarat Lions"                     : "#E95B10",
    "Gujarat Titans"                    : "#1C4E8A",
    "Lucknow Super Giants"              : "#A0C4FF",
    "Kochi Tuskers Kerala"              : "#B22222",
}
DEFAULT_COLOR = "#4A90D9"

TEAM_SHORT = {
    "Mumbai Indians":"MI","Chennai Super Kings":"CSK",
    "Royal Challengers Bangalore":"RCB","Royal Challengers Bengaluru":"RCB",
    "Kolkata Knight Riders":"KKR","Sunrisers Hyderabad":"SRH",
    "Delhi Capitals":"DC","Delhi Daredevils":"DD",
    "Kings XI Punjab":"KXIP","Punjab Kings":"PBKS",
    "Rajasthan Royals":"RR","Deccan Chargers":"DCH",
    "Pune Warriors":"PW","Rising Pune Supergiant":"RPS",
    "Rising Pune Supergiants":"RPS","Gujarat Lions":"GL",
    "Gujarat Titans":"GT","Lucknow Super Giants":"LSG",
    "Kochi Tuskers Kerala":"KTK",
}

TEAM_HOME = {
    "Chennai Super Kings"        :("Chennai","MA Chidambaram Stadium, Chepauk"),
    "Deccan Chargers"            :("Hyderabad","Rajiv Gandhi International Stadium, Uppal"),
    "Delhi Capitals"             :("Delhi","Feroz Shah Kotla Ground"),
    "Delhi Daredevils"           :("Delhi","Feroz Shah Kotla"),
    "Gujarat Lions"              :("Rajkot","Saurashtra Cricket Association Stadium"),
    "Kings XI Punjab"            :("Chandigarh","Punjab Cricket Association Stadium, Mohali"),
    "Kochi Tuskers Kerala"       :("Kochi","Nehru Stadium"),
    "Kolkata Knight Riders"      :("Kolkata","Eden Gardens"),
    "Mumbai Indians"             :("Mumbai","Wankhede Stadium"),
    "Pune Warriors"              :("Mumbai","Subrata Roy Sahara Stadium"),
    "Rajasthan Royals"           :("Jaipur","Sawai Mansingh Stadium"),
    "Rising Pune Supergiant"     :("Mumbai","Wankhede Stadium"),
    "Rising Pune Supergiants"    :("Pune","Maharashtra Cricket Association Stadium"),
    "Royal Challengers Bangalore":("Bangalore","M Chinnaswamy Stadium"),
    "Sunrisers Hyderabad"        :("Hyderabad","Rajiv Gandhi International Stadium, Uppal"),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION — via JavaScript so it works in ALL Streamlit versions
# The JS pushes a <style> into the parent page from inside the component iframe.
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    css = """
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            background: #080d1a !important;
            color: #e8eaf0 !important;
        }
        [data-testid="stSidebar"] {
            background: #0d1427 !important;
            border-right: 1px solid #1e2d50 !important;
        }
        [data-testid="stSidebar"] * { color: #c8d0e0 !important; }
        .block-container { padding: 1.5rem 2rem !important; }
        h1,h2,h3,h4 { font-family: 'Bebas Neue', cursive !important; letter-spacing: 2px !important; }
        p, span, div, label, li { font-family: 'DM Sans', sans-serif !important; }
        #MainMenu, footer, header { visibility: hidden !important; }

        [data-testid="metric-container"] {
            background: #111827 !important;
            border: 1px solid #1e3a5f !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'DM Sans', sans-serif !important;
            color: #8899bb !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', cursive !important;
            font-size: 2.2rem !important;
            color: #f5c842 !important;
        }
        [data-baseweb="select"] > div {
            background: #111827 !important;
            border: 1px solid #1e3a5f !important;
            border-radius: 8px !important;
            color: #e8eaf0 !important;
        }
        [data-baseweb="select"] * { color: #e8eaf0 !important; background: #111827 !important; }
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg,#f5c842,#f5841f) !important;
            color: #080d1a !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Bebas Neue', cursive !important;
            font-size: 1.1rem !important;
            letter-spacing: 1.5px !important;
        }
        [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 2px solid #1e2d50 !important;
        }
        [data-baseweb="tab"] {
            font-family: 'Bebas Neue', cursive !important;
            font-size: 1.05rem !important;
            letter-spacing: 1.5px !important;
            color: #5a6b8a !important;
            background: transparent !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            color: #f5c842 !important;
            border-bottom: 3px solid #f5c842 !important;
            background: transparent !important;
        }
        [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 1.5rem !important; }
        [data-testid="stRadio"] label { font-family: 'DM Sans', sans-serif !important; color: #c8d0e0 !important; }
        hr { border-color: #1e2d50 !important; }
    """
    # Inject CSS into parent document via JavaScript from component iframe
    js = f"""
    <script>
        const css = `{css}`;
        const style = document.createElement('style');
        style.textContent = css;
        window.parent.document.head.appendChild(style);
    </script>
    """
    components.html(js, height=0, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_winner_model():
    with open(os.path.join(MODEL_DIR, "winner_model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_all_results():
    from src.match_winner import train_match_winner_model
    from src.orange_cap   import train_orange_cap_model
    from src.purple_cap   import train_purple_cap_model
    _, _, acc, champs = train_match_winner_model(DATA_DIR, MODEL_DIR)
    _, oc_r2, oc_df   = train_orange_cap_model(DATA_DIR, MODEL_DIR)
    _, pc_r2, pc_df   = train_purple_cap_model(DATA_DIR, MODEL_DIR)
    return acc, champs, oc_r2, oc_df, pc_r2, pc_df

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_winner(bundle, team1, team2, city, venue, toss_winner, toss_decision, season):
    model, encoders = bundle["model"], bundle["encoders"]
    def enc(col, val):
        le = encoders[col]
        return int(le.transform([val])[0]) if val in le.classes_ else 0
    row = pd.DataFrame([{
        "season":enc("season",str(season)),"team1":enc("team1",team1),
        "team2":enc("team2",team2),"city":enc("city",city),
        "venue":enc("venue",venue),"toss_winner":enc("toss_winner",toss_winner),
        "toss_decision":enc("toss_decision",toss_decision),
    }])
    pred   = model.predict(row)[0]
    proba  = model.predict_proba(row)[0]
    winner = encoders["winner"].inverse_transform([pred])[0]
    pmap   = dict(zip(encoders["winner"].classes_, proba))
    p1, p2 = pmap.get(team1,0)*100, pmap.get(team2,0)*100
    tot    = p1 + p2
    if tot > 0: p1, p2 = (p1/tot)*100, (p2/tot)*100
    else:       p1 = p2 = 50.0
    return winner, round(p1,1), round(p2,1)

# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS  (inline via st.markdown unsafe_allow_html — content only, no <style>)
# ─────────────────────────────────────────────────────────────────────────────
def _html(content):
    st.markdown(content, unsafe_allow_html=True)

def _section_header(icon, title, subtitle=""):
    sub = f"<p style='color:#6b7a99;font-size:0.88rem;margin:2px 0 0;'>{subtitle}</p>" if subtitle else ""
    _html(f"""
    <div style='margin-bottom:1rem;'>
      <h2 style='font-family:Bebas Neue,cursive;font-size:2rem;letter-spacing:3px;
                 color:#e8eaf0;margin:0;'>{icon} {title}</h2>
      {sub}
    </div>
    <div style='height:4px;background:linear-gradient(90deg,#f5c842,#f5841f,transparent);
                border-radius:2px;margin-bottom:22px;'></div>
    """)

def _team_badge(name):
    c = TEAM_COLORS.get(name, DEFAULT_COLOR)
    s = TEAM_SHORT.get(name, name[:3].upper())
    _html(f"""
    <div style='display:inline-flex;align-items:center;gap:10px;background:{c}22;
                border:1px solid {c}55;border-radius:10px;padding:8px 14px;margin-top:6px;'>
      <div style='width:36px;height:36px;border-radius:50%;background:{c};display:flex;
                  align-items:center;justify-content:center;font-family:Bebas Neue,cursive;
                  font-size:0.8rem;color:white;'>{s}</div>
      <span style='font-weight:600;color:#e8eaf0;font-size:0.95rem;'>{name}</span>
    </div>""")

def _prob_bars(team1, team2, p1, p2, winner):
    c1 = TEAM_COLORS.get(team1, DEFAULT_COLOR)
    c2 = TEAM_COLORS.get(team2, DEFAULT_COLOR)
    lbl1 = "🏆 WINNER" if team1 == winner else ""
    lbl2 = "🏆 WINNER" if team2 == winner else ""
    _html(f"""
    <div style='background:#111827;border:1px solid #1e3a5f;border-radius:16px;padding:26px 28px;'>
      <div style='margin-bottom:20px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;'>
          <span style='font-family:Bebas Neue,cursive;font-size:1.1rem;color:{c1};letter-spacing:1px;'>{team1}</span>
          <span style='font-family:Bebas Neue,cursive;font-size:1rem;color:{c1};'>{lbl1}&nbsp;{p1}%</span>
        </div>
        <div style='background:#1a2540;border-radius:999px;height:15px;overflow:hidden;'>
          <div style='height:100%;width:{p1}%;background:linear-gradient(90deg,{c1},{c1}99);border-radius:999px;'></div>
        </div>
      </div>
      <div>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;'>
          <span style='font-family:Bebas Neue,cursive;font-size:1.1rem;color:{c2};letter-spacing:1px;'>{team2}</span>
          <span style='font-family:Bebas Neue,cursive;font-size:1rem;color:{c2};'>{lbl2}&nbsp;{p2}%</span>
        </div>
        <div style='background:#1a2540;border-radius:999px;height:15px;overflow:hidden;'>
          <div style='height:100%;width:{p2}%;background:linear-gradient(90deg,{c2},{c2}99);border-radius:999px;'></div>
        </div>
      </div>
    </div>""")

def _winner_card(winner):
    c = TEAM_COLORS.get(winner, DEFAULT_COLOR)
    s = TEAM_SHORT.get(winner, winner[:3].upper())
    _html(f"""
    <div style='background:{c}22;border:2px solid {c};border-radius:16px;
                padding:24px;text-align:center;'>
      <p style='color:{c};font-size:0.72rem;letter-spacing:3px;text-transform:uppercase;margin:0 0 10px;'>Predicted Winner</p>
      <div style='width:64px;height:64px;border-radius:50%;background:{c};display:flex;
                  align-items:center;justify-content:center;font-family:Bebas Neue,cursive;
                  font-size:1.3rem;color:white;margin:0 auto 10px;'>{s}</div>
      <p style='font-family:Bebas Neue,cursive;font-size:1.8rem;color:{c};letter-spacing:2px;margin:0;'>
        🏆 {winner}
      </p>
    </div>""")

def _cap_card(cap_type, player_name, stat_str, color, emoji, badge="PREDICTED"):
    initials = "".join(p[0] for p in player_name.split() if p)[:3].upper()
    badge_color = "#f5c842" if badge == "PREDICTED" else "#4ade80"
    badge_bg    = "#f5c84222" if badge == "PREDICTED" else "#4ade8022"
    _html(f"""
    <div style='background:{color}18;border:2px solid {color}88;border-radius:16px;
                padding:22px;text-align:center;'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
        <p style='color:{color};font-size:0.72rem;letter-spacing:3px;text-transform:uppercase;margin:0;'>
          {emoji} {cap_type}
        </p>
        <span style='background:{badge_bg};border:1px solid {badge_color}55;border-radius:999px;
                     padding:2px 10px;font-size:0.68rem;color:{badge_color};letter-spacing:1px;'>
          {badge}
        </span>
      </div>
      <div style='width:64px;height:64px;border-radius:50%;
                  background:linear-gradient(135deg,{color},{color}88);
                  display:flex;align-items:center;justify-content:center;
                  font-family:Bebas Neue,cursive;font-size:1.1rem;color:white;margin:0 auto 10px;'>
        {initials}
      </div>
      <p style='font-family:Bebas Neue,cursive;font-size:1.55rem;color:#e8eaf0;
                letter-spacing:1px;line-height:1.2;margin:0 0 10px;'>{player_name}</p>
      <div style='display:inline-block;background:{color}33;border:1px solid {color}66;
                  border-radius:999px;padding:4px 16px;'>
        <span style='font-family:Bebas Neue,cursive;font-size:1.1rem;color:{color};'>{stat_str}</span>
      </div>
    </div>""")

def _actual_strip(label, value, color):
    _html(f"""
    <div style='background:#111827;border:1px solid {color}33;border-radius:10px;
                padding:10px 16px;margin-top:8px;display:flex;
                justify-content:space-between;align-items:center;'>
      <span style='font-size:0.8rem;color:#6b7a99;'>Actual {label}</span>
      <span style='font-size:0.85rem;font-weight:600;color:{color};'>{value}</span>
    </div>""")

def _season_table(df, columns):
    headers = "".join(
        f"<th style='padding:10px 16px;font-family:Bebas Neue,cursive;font-size:0.88rem;"
        f"letter-spacing:1px;color:#8899bb;font-weight:400;border-bottom:2px solid #1e3a5f;"
        f"text-align:left;white-space:nowrap;'>{lbl}</th>"
        for _, lbl in columns
    )
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col, _ in columns:
            val = row[col]
            if col == "match":
                clr = "#4ade80" if val == "✅" else "#f87171"
                cells += f"<td style='padding:10px 16px;border-bottom:1px solid #131d35;color:{clr};text-align:center;font-size:1rem;'>{val}</td>"
            else:
                cells += f"<td style='padding:10px 16px;border-bottom:1px solid #131d35;font-size:0.87rem;color:#c8d4e8;white-space:nowrap;'>{val}</td>"
        rows += f"<tr>{cells}</tr>"
    _html(f"""
    <div style='overflow-x:auto;border-radius:12px;border:1px solid #1e3a5f;'>
    <table style='width:100%;border-collapse:collapse;background:#0d1427;'>
      <thead><tr style='background:#111827;'>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table></div>""")

def _stat_card_html(label, value, value_color, sub_label, sub_value):
    _html(f"""
    <div style='background:#111827;border:1px solid {value_color}44;border-radius:14px;
                padding:24px;text-align:center;'>
      <p style='color:#8899bb;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;margin:0;'>
        {label}
      </p>
      <p style='font-family:Bebas Neue,cursive;font-size:3rem;color:{value_color};margin:8px 0;'>{value}</p>
      <p style='color:#8899bb;font-size:0.75rem;margin:14px 0 0;letter-spacing:1px;'>{sub_label}</p>
      <p style='font-family:Bebas Neue,cursive;font-size:2.2rem;color:#f5c842;margin:0;'>{sub_value}</p>
    </div>""")

def _top5_table(rows, cols):
    """Render a simple ranking table."""
    headers = "".join(
        f"<th style='padding:8px 14px;font-family:Bebas Neue,cursive;font-size:0.82rem;"
        f"letter-spacing:1px;color:#8899bb;font-weight:400;border-bottom:2px solid #1e3a5f;"
        f"text-align:left;'>{c}</th>" for c in cols
    )
    body = ""
    for i, row in enumerate(rows, 1):
        rank_color = ["#f5c842","#adb5bd","#cd7f32"][i-1] if i <= 3 else "#5a6b8a"
        cells = f"<td style='padding:8px 14px;border-bottom:1px solid #131d35;font-family:Bebas Neue,cursive;font-size:1rem;color:{rank_color};'>#{i}</td>"
        cells += "".join(f"<td style='padding:8px 14px;border-bottom:1px solid #131d35;font-size:0.85rem;color:#c8d4e8;'>{v}</td>" for v in row)
        body += f"<tr>{cells}</tr>"
    _html(f"""
    <div style='overflow-x:auto;border-radius:10px;border:1px solid #1e3a5f;margin-top:12px;'>
    <table style='width:100%;border-collapse:collapse;background:#0d1427;'>
      <thead><tr style='background:#111827;'>
        <th style='padding:8px 14px;font-family:Bebas Neue,cursive;font-size:0.82rem;color:#8899bb;
                   font-weight:400;border-bottom:2px solid #1e3a5f;text-align:left;'>#</th>
        {headers}
      </tr></thead>
      <tbody>{body}</tbody>
    </table></div>""")

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

def page_predictor():
    _section_header("🏏", "MATCH PREDICTOR",
                    "Predict match winner + season cap holders for any IPL season")

    models_ok = all(os.path.isfile(os.path.join(MODEL_DIR, f))
                    for f in ["winner_model.pkl","orange_cap_model.pkl","purple_cap_model.pkl"])
    if not models_ok:
        st.error("⚠️  Models not found. Run  `python main.py`  first.")
        return

    bundle   = load_winner_model()
    encoders = bundle["encoders"]
    all_teams   = sorted(encoders["team1"].classes_)
    model_seasons = sorted([int(s) for s in encoders["season"].classes_])
    all_seasons = model_seasons  # 2008–2026 (all seasons in model)
    all_cities  = sorted(encoders["city"].classes_)
    all_venues  = sorted(encoders["venue"].classes_)

    st.markdown("---")
    _html("<p style='font-family:Bebas Neue,cursive;font-size:1rem;letter-spacing:3px;"
          "color:#8899bb;margin-bottom:4px;'>HISTORICAL MATCH PREDICTOR (2008–2026)</p>")
    _html("<p style='color:#6b7a99;font-size:0.82rem;margin-bottom:16px;'>"
          "Select any two IPL teams from the 2008–2026 era to simulate a match prediction.</p>")

    # Team selectors
    col1, vs_col, col2 = st.columns([5, 1, 5])
    with col1:
        st.markdown("**TEAM 1**")
        team1 = st.selectbox(" ", all_teams,
            index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0,
            key="t1", label_visibility="collapsed")
        _team_badge(team1)

    with vs_col:
        _html("<div style='text-align:center;font-family:Bebas Neue,cursive;"
              "font-size:2rem;color:#2a3a5a;margin-top:48px;'>VS</div>")

    with col2:
        st.markdown("**TEAM 2**")
        opts  = [t for t in all_teams if t != team1]
        team2 = st.selectbox(" ", opts,
            index=opts.index("Chennai Super Kings") if "Chennai Super Kings" in opts else 0,
            key="t2", label_visibility="collapsed")
        _team_badge(team2)

    st.markdown("&nbsp;")
    st.markdown("**MATCH DETAILS**")

    d1, d2, d3 = st.columns(3)
    with d1: toss_winner   = st.radio("Toss Winner",   [team1, team2], horizontal=True)
    with d2: toss_decision = st.radio("Toss Decision", ["bat","field"], horizontal=True)
    with d3: season = st.selectbox("Season", all_seasons, index=len(all_seasons)-1)

    def_city, def_venue = TEAM_HOME.get(team1, ("Mumbai","Wankhede Stadium"))
    e1, e2 = st.columns(2)
    with e1:
        city  = st.selectbox("City",  all_cities,
            index=all_cities.index(def_city) if def_city in all_cities else 0)
    with e2:
        venue = st.selectbox("Venue", all_venues,
            index=all_venues.index(def_venue) if def_venue in all_venues else 0)

    st.markdown("&nbsp;")
    _, btn_col, _ = st.columns([3,2,3])
    with btn_col:
        clicked = st.button("⚡  PREDICT", use_container_width=True)

    if clicked:
        if team1 == team2:
            st.warning("Please choose two different teams.")
            return

        winner, p1, p2 = predict_winner(
            bundle, team1, team2, city, venue,
            toss_winner, toss_decision, season
        )

        with st.spinner(""):
            _, _, _, oc_df, _, pc_df = load_all_results()

        oc_row = oc_df[oc_df["season"] == season]
        pc_row = pc_df[pc_df["season"] == season]

        st.markdown("&nbsp;")

        # ── Match winner
        _html("<p style='font-family:Bebas Neue,cursive;font-size:1rem;letter-spacing:3px;"
              "color:#8899bb;margin-bottom:12px;'>MATCH WINNER PREDICTION</p>")
        r1, r2 = st.columns([3, 2])
        with r1: _prob_bars(team1, team2, p1, p2, winner)
        with r2: _winner_card(winner)

        st.markdown("&nbsp;")

        # ── Cap winners for selected season
        _html(f"<p style='font-family:Bebas Neue,cursive;font-size:1rem;letter-spacing:3px;"
              f"color:#8899bb;margin-bottom:12px;'>SEASON {season} — CAP WINNERS</p>")

        cap1, cap2 = st.columns(2)
        with cap1:
            if len(oc_row) > 0:
                player    = oc_row.iloc[0]["predicted_orange_cap"]
                pred_runs = int(round(oc_row.iloc[0]["predicted_runs"]))
                actual    = oc_row.iloc[0]["actual_orange_cap"]
                a_runs    = int(oc_row.iloc[0]["actual_runs"])
                _cap_card("Orange Cap", player, f"{pred_runs} runs", "#FF8C00", "🟠", "PREDICTED")
                _actual_strip("Orange Cap", f"{actual}  ({a_runs} runs)", "#FF8C00")

        with cap2:
            if len(pc_row) > 0:
                player   = pc_row.iloc[0]["predicted_purple_cap"]
                pred_wk  = pc_row.iloc[0]["predicted_wickets"]
                actual   = pc_row.iloc[0]["actual_purple_cap"]
                a_wkts   = int(pc_row.iloc[0]["actual_wickets"])
                _cap_card("Purple Cap", player, f"{pred_wk:.0f} wkts", "#9B59B6", "🟣", "PREDICTED")
                _actual_strip("Purple Cap", f"{actual}  ({a_wkts} wkts)", "#9B59B6")

        # Detail chips
        st.markdown("&nbsp;")
        _html(f"""
        <div style='display:flex;gap:10px;flex-wrap:wrap;'>
          <div style='background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:7px 14px;font-size:0.8rem;color:#8899bb;'>
            <span style='color:#f5c842;'>Season</span>&emsp;{season}
          </div>
          <div style='background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:7px 14px;font-size:0.8rem;color:#8899bb;'>
            <span style='color:#f5c842;'>City</span>&emsp;{city}
          </div>
          <div style='background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:7px 14px;font-size:0.8rem;color:#8899bb;'>
            <span style='color:#f5c842;'>Toss</span>&emsp;{toss_winner} ({toss_decision})
          </div>
        </div>""")


def page_2026():
    """Dedicated page showing full IPL 2026 verified stats."""
    _section_header("🏆", "IPL 2026 — OFFICIAL RESULTS",
                    "Verified final standings from IPL 2026 (season complete)")

    d = IPL_2026
    champ_color  = TEAM_COLORS.get(d["champion"], DEFAULT_COLOR)
    champ_short  = TEAM_SHORT.get(d["champion"], "RCB")
    runner_color = TEAM_COLORS.get(d["runner_up"], DEFAULT_COLOR)
    runner_short = TEAM_SHORT.get(d["runner_up"], "GT")

    # Champion card
    _html(f"""
    <div style='background:linear-gradient(135deg,{champ_color}33,{champ_color}11);
                border:2px solid {champ_color};border-radius:18px;
                padding:28px;text-align:center;margin-bottom:24px;'>
      <p style='color:{champ_color};font-size:0.8rem;letter-spacing:4px;
                text-transform:uppercase;margin:0 0 12px;'>🏆 IPL 2026 CHAMPION</p>
      <div style='width:80px;height:80px;border-radius:50%;background:{champ_color};
                  display:flex;align-items:center;justify-content:center;
                  font-family:Bebas Neue,cursive;font-size:1.6rem;color:white;
                  margin:0 auto 14px;'>{champ_short}</div>
      <p style='font-family:Bebas Neue,cursive;font-size:2.4rem;color:{champ_color};
                letter-spacing:2px;margin:0 0 6px;'>{d["champion"]}</p>
      <p style='color:#6b7a99;font-size:0.88rem;margin:0;'>
        Defeated &nbsp;
        <span style='color:{runner_color};font-weight:600;'>{d["runner_up"]}</span>
        &nbsp; in the Final · Ahmedabad
      </p>
    </div>
    """)

    tab_oc, tab_pc = st.tabs(["🟠  Orange Cap", "🟣  Purple Cap"])

    with tab_oc:
        c1, c2 = st.columns([2, 1])
        with c1:
            _html(f"""
            <div style='background:#FF8C0018;border:2px solid #FF8C0088;border-radius:16px;
                        padding:22px;text-align:center;margin-bottom:16px;'>
              <p style='color:#FF8C00;font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;margin:0 0 8px;'>
                🟠 ORANGE CAP WINNER 2026
              </p>
              <div style='width:72px;height:72px;border-radius:50%;
                          background:linear-gradient(135deg,#FF8C00,#FF8C0088);
                          display:flex;align-items:center;justify-content:center;
                          font-family:Bebas Neue,cursive;font-size:1.2rem;color:white;
                          margin:0 auto 12px;'>VS</div>
              <p style='font-family:Bebas Neue,cursive;font-size:2rem;color:#e8eaf0;
                        letter-spacing:1px;margin:0 0 8px;'>{d["orange_cap"]}</p>
              <p style='color:#FF8C00;font-size:1rem;font-weight:600;margin:0;'>
                {d["orange_runs"]} RUNS &nbsp;·&nbsp; {d["orange_matches"]} MATCHES
                &nbsp;·&nbsp; SR {d["orange_sr"]:.0f}
              </p>
              <p style='color:#6b7a99;font-size:0.8rem;margin:4px 0 0;'>{d["orange_team"]}</p>
            </div>""")

            _html("<p style='font-family:Bebas Neue,cursive;font-size:0.9rem;letter-spacing:2px;"
                  "color:#8899bb;margin-bottom:4px;'>TOP 5 RUN SCORERS — IPL 2026</p>")
            _top5_table(
                [(p, t, f"{r} runs") for p, t, r in d["oc_top5"]],
                ["Player","Team","Runs"]
            )
        with c2:
            st.metric("Most Runs", f"{d['orange_runs']}")
            st.metric("Matches Played", f"{d['orange_matches']}")
            st.metric("Strike Rate", f"{d['orange_sr']:.0f}")
            st.metric("Team", d["orange_team"])

    with tab_pc:
        c1, c2 = st.columns([2, 1])
        with c1:
            _html(f"""
            <div style='background:#9B59B618;border:2px solid #9B59B688;border-radius:16px;
                        padding:22px;text-align:center;margin-bottom:16px;'>
              <p style='color:#9B59B6;font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;margin:0 0 8px;'>
                🟣 PURPLE CAP WINNER 2026
              </p>
              <div style='width:72px;height:72px;border-radius:50%;
                          background:linear-gradient(135deg,#9B59B6,#9B59B688);
                          display:flex;align-items:center;justify-content:center;
                          font-family:Bebas Neue,cursive;font-size:1.2rem;color:white;
                          margin:0 auto 12px;'>KR</div>
              <p style='font-family:Bebas Neue,cursive;font-size:2rem;color:#e8eaf0;
                        letter-spacing:1px;margin:0 0 8px;'>{d["purple_cap"]}</p>
              <p style='color:#9B59B6;font-size:1rem;font-weight:600;margin:0;'>
                {d["purple_wickets"]} WICKETS &nbsp;·&nbsp; {d["purple_matches"]} MATCHES
                &nbsp;·&nbsp; Econ {d["purple_economy"]:.2f}
              </p>
              <p style='color:#6b7a99;font-size:0.8rem;margin:4px 0 0;'>{d["purple_team"]}</p>
            </div>""")

            _html("<p style='font-family:Bebas Neue,cursive;font-size:0.9rem;letter-spacing:2px;"
                  "color:#8899bb;margin-bottom:4px;'>TOP 5 WICKET TAKERS — IPL 2026</p>")
            _top5_table(
                [(p, t, f"{w} wkts") for p, t, w in d["pc_top5"]],
                ["Player","Team","Wickets"]
            )
        with c2:
            st.metric("Most Wickets", f"{d['purple_wickets']}")
            st.metric("Matches Played", f"{d['purple_matches']}")
            st.metric("Economy Rate", f"{d['purple_economy']:.2f}")
            st.metric("Team", d["purple_team"])


def page_season_results():
    _section_header("📋", "HISTORICAL SEASON RESULTS",
                    "Model-predicted vs actual cap winners for IPL 2008–2026")

    if not all(os.path.isfile(os.path.join(MODEL_DIR, f))
               for f in ["winner_model.pkl","orange_cap_model.pkl","purple_cap_model.pkl"]):
        st.error("⚠️  Models not found. Run `python main.py` first.")
        return

    with st.spinner("Loading season data..."):
        acc, champs, oc_r2, oc_df, pc_r2, pc_df = load_all_results()

    tab1, tab2, tab3 = st.tabs(["🟠  Orange Cap", "🟣  Purple Cap", "🏏  Match Champions"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            df = oc_df.copy()
            df["season"]         = df["season"].astype(int)
            df["predicted_runs"] = df["predicted_runs"].round(0).astype(int)
            df["actual_runs"]    = df["actual_runs"].astype(int)
            df["match"]          = df.apply(lambda r: "✅" if r["predicted_orange_cap"]==r["actual_orange_cap"] else "❌", axis=1)
            _season_table(df, [("season","Season"),("predicted_orange_cap","Predicted"),
                               ("predicted_runs","Pred Runs"),("actual_orange_cap","Actual"),
                               ("actual_runs","Actual Runs"),("match","✓")])
        with c2:
            correct = (oc_df["predicted_orange_cap"]==oc_df["actual_orange_cap"]).sum()
            _stat_card_html("R² Score", f"{oc_r2:.4f}", "#FF8C00", "Seasons Correct", f"{correct}/{len(oc_df)}")

    with tab2:
        c1, c2 = st.columns([3, 1])
        with c1:
            df = pc_df.copy()
            df["season"]            = df["season"].astype(int)
            df["predicted_wickets"] = df["predicted_wickets"].round(1)
            df["actual_wickets"]    = df["actual_wickets"].astype(int)
            df["match"]             = df.apply(lambda r: "✅" if r["predicted_purple_cap"]==r["actual_purple_cap"] else "❌", axis=1)
            _season_table(df, [("season","Season"),("predicted_purple_cap","Predicted"),
                               ("predicted_wickets","Pred Wkts"),("actual_purple_cap","Actual"),
                               ("actual_wickets","Actual Wkts"),("match","✓")])
        with c2:
            correct = (pc_df["predicted_purple_cap"]==pc_df["actual_purple_cap"]).sum()
            _stat_card_html("R² Score", f"{pc_r2:.4f}", "#9B59B6", "Seasons Correct", f"{correct}/{len(pc_df)}")

    with tab3:
        c1, c2 = st.columns([3, 1])
        with c1:
            df = champs.copy()
            df["season"] = df["season"].astype(int)
            # Determine the actual-champion column name robustly
            actual_col = next(
                (c for c in ["actual_champion", "actual_most_wins_team", "actual_winner"]
                 if c in df.columns),
                [c for c in df.columns if c not in ("season", "predicted_champion")][0]
            )
            df["match"]  = df.apply(lambda r: "✅" if r["predicted_champion"]==r[actual_col] else "❌", axis=1)
            _season_table(df, [("season","Season"),("predicted_champion","Predicted Champion"),
                               (actual_col,"Actual Champion"),("match","✓")])
        with c2:
            actual_col = next(
                (c for c in ["actual_champion", "actual_most_wins_team", "actual_winner"]
                 if c in champs.columns),
                [c for c in champs.columns if c not in ("season", "predicted_champion")][0]
            )
            correct = (champs["predicted_champion"]==champs[actual_col]).sum()
            _stat_card_html("Accuracy", f"{acc*100:.1f}%", "#4A90D9", "Seasons Correct", f"{correct}/{len(champs)}")


def page_model_stats():
    _section_header("📊", "MODEL PERFORMANCE",
                    "Evaluation metrics and feature importance for all three ML models")

    if not all(os.path.isfile(os.path.join(MODEL_DIR, f))
               for f in ["winner_model.pkl","orange_cap_model.pkl","purple_cap_model.pkl"]):
        st.error("⚠️  Models not found. Run `python main.py` first.")
        return

    with st.spinner("Loading..."):
        acc, _, oc_r2, _, pc_r2, _ = load_all_results()

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Match Winner Accuracy", f"{acc*100:.2f}%")
    with m2: st.metric("Orange Cap R² Score",   f"{oc_r2:.4f}")
    with m3: st.metric("Purple Cap R² Score",   f"{pc_r2:.4f}")

    st.markdown("&nbsp;")
    _html("<p style='font-family:Bebas Neue,cursive;font-size:1rem;letter-spacing:2px;color:#8899bb;'>FEATURE IMPORTANCE</p>")

    charts = {
        "match_winner_feature_importance.png":("Match Winner","#4A90D9"),
        "orange_cap_feature_importance.png"  :("Orange Cap","#FF8C00"),
        "purple_cap_feature_importance.png"  :("Purple Cap","#9B59B6"),
    }
    cols = st.columns(3)
    for i, (fname,(title,color)) in enumerate(charts.items()):
        path = os.path.join(MODEL_DIR, fname)
        with cols[i]:
            _html(f"<p style='font-family:Bebas Neue,cursive;font-size:1rem;color:{color};"
                  f"letter-spacing:1px;text-align:center;margin-bottom:8px;'>{title}</p>")
            if os.path.isfile(path):
                st.image(path, use_container_width=True)
            else:
                st.info("Run `python main.py` to generate this chart.")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        _html("""
        <div style='padding:12px 8px 20px;'>
          <h1 style='font-family:Bebas Neue,cursive;font-size:1.8rem;letter-spacing:3px;
                     color:#f5c842;line-height:1.1;margin:0;'>🏏 IPL</h1>
          <p style='font-family:Bebas Neue,cursive;font-size:1rem;color:#8899bb;
                    letter-spacing:4px;margin:0;'>PREDICTOR</p>
          <p style='font-size:0.72rem;color:#3a4a6b;margin:4px 0 0;'>Random Forest · 2008–2026</p>
        </div>""")
        st.divider()

        if "page" not in st.session_state:
            st.session_state.page = "Predictor"

        pages = [
            ("⚡  Predictor",           "Predictor"),
            ("🏆  IPL 2026 Results",    "IPL 2026"),
            ("📋  Historical Results",  "Season Results"),
            ("📊  Model Stats",         "Model Stats"),
        ]
        for label, key in pages:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key

        st.divider()
        _html("""
        <div style='font-size:0.75rem;color:#3a4a6b;padding:4px 8px;line-height:2.2;'>
          <b style='color:#5a6b8a;display:block;margin-bottom:4px;'>MODELS</b>
          🔵 Match Winner · RF Classifier<br>
          🟠 Orange Cap · RF Regressor<br>
          🟣 Purple Cap · RF Regressor
        </div>""")

    return st.session_state.get("page", "Predictor")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    page = sidebar_nav()
    if   page == "Predictor"      : page_predictor()
    elif page == "IPL 2026"        : page_2026()
    elif page == "Season Results"  : page_season_results()
    elif page == "Model Stats"     : page_model_stats()

if __name__ == "__main__":
    main()