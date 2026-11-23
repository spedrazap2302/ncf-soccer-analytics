# Streamlit app: scouting the Sun Conference for New College men's soccer
# Run with: streamlit run app.py
import importlib.util  # built-in: checks if a package is installed without loading it
import os  # built-in: used to check if a logo file exists

# Altair is the chart library that comes with Streamlit. It's like matplotlib, but the
# charts are interactive: you can hover a bar or a dot to see the exact number.
import altair as alt
import pandas as pd
import streamlit as st

import eda
import scouting as sc

st.set_page_config(page_title="NCF Soccer Scouting", page_icon="⚽", layout="wide")

US_COLOR = "#0b7a75"      # New College teal
THEM_COLOR = "#e4572e"    # opponent orange
OTHER_COLOR = "#c9ced6"   # everyone else gray
RESULT_COLORS = alt.Scale(domain=["Win", "Tie", "Loss"], range=["#2e9e5b", "#9aa3ad", "#d64545"])

# readable names for the raw game columns
GAME_LABELS = {
    "date": "Date", "season": "Season", "home_away": "Where", "result": "Result",
    "opponent": "Opponent", "team_name": "Team",
    "goals_for": "Goals scored", "goals_against": "Goals allowed",
    "shots_for": "Shots", "shots_against": "Shots allowed",
    "sog_for": "Shots on target", "sog_against": "Shots on target allowed",
    "corners_for": "Corners", "corners_against": "Corners allowed",
    "yellows_for": "Yellow cards", "yellows_against": "Opponent yellow cards",
    "conference_game": "Conference game", "goal_diff": "Goal difference",
}


# @st.cache_data = "remember the result": load the CSVs once instead of on every click
@st.cache_data
def get_data():
    return eda.load_games(), eda.load_players(), eda.load_goals()


# @st.cache_resource = same idea for big objects like the AI model (load it only once)
@st.cache_resource
def get_llm():
    # load the language model once and keep it in memory
    import llm
    return llm.load_model()


# remember this one too, since it takes a few seconds
@st.cache_data
def get_backtest(games, half_life):
    # slow (it rebuilds the model for every game), so only run it once
    return sc.backtest(games, half_life=half_life)


try:
    games, players, goals = get_data()
except FileNotFoundError:
    st.error("Data files not found. Run `python scraper.py`, `python boxscores.py`, `python weather.py` "
             "and then `python clean.py` first.")
    st.stop()

name_to_key = {v: k for k, v in eda.TEAM_NAMES.items()}

# ---------- sidebar ----------
st.sidebar.title("⚙️ Settings")
our_name = st.sidebar.selectbox("Our team", list(name_to_key), index=list(name_to_key).index("New College"))
us = name_to_key[our_name]
seasons = sorted(games["season"].unique())
picked = st.sidebar.multiselect("Seasons to learn from", seasons, default=[2025, 2026],
                                help="Recent seasons show current form. Add more seasons for a bigger sample.")
conf_only = st.sidebar.toggle("Conference games only", value=True,
                              help="Conference games are the fairest comparison: every team plays the same opponents.")
recent = st.sidebar.toggle("Count recent games more", value=True,
                           help="A game from a year ago counts half as much as one from this week.")
half_life = 365 if recent else None
st.sidebar.caption("New College joined the Sun Conference in 2024 (2023 was friendlies). "
                   "2026 is the current season.")

if not picked:
    st.warning("Pick at least one season in the sidebar.")
    st.stop()

g = sc.pick_games(games, picked, conf_only)
prof = sc.team_profile(g, half_life)
if us not in prof.index:
    st.warning(f"{our_name} has no games for these settings. Try other seasons or turn off 'Conference games only'.")
    st.stop()

# ---------- header ----------
st.title("⚽ Sun Conference Scouting")
st.markdown(f"Pick an opponent and see **how likely {our_name} is to win, what the opponent is good at, "
            "and how to play them**. Built from 600+ games scraped from the 9 Sun Conference "
            "schools' athletics websites (2023 to 2026).")

tab_scout, tab_conf, tab_wins, tab_players, tab_data = st.tabs(
    ["🔍 Scout an opponent", "🏆 Conference rankings", "💡 What wins games", "⚽ Players", "📁 About the data"])

# =====================================================================
# 1. SCOUT AN OPPONENT
# =====================================================================
with tab_scout:
    others = [t for t in prof.index if t != us]
    venue_label = st.radio("Where is the game?", ["Home", "Away", "Neutral"], horizontal=True)
    venue = venue_label.lower()

    # --- pick an opponent: one card per team, click "Scout" to open its report ---
    st.markdown("#### Pick an opponent")
    if st.session_state.get("opp") not in others:
        st.session_state["opp"] = others[0]
    cols = st.columns(4)
    for i, team in enumerate(others):
        h2h_cards = sc.head_to_head(games, us, team)
        record = "-".join(str((h2h_cards["result"] == r).sum()) for r in ["W", "T", "L"])
        win = sc.result_chances(*sc.expected_goals(g, prof, us, team, venue))["Win"]
        with cols[i % 4].container(border=True):
            logo = f"assets/logos/{team}.png"
            if os.path.exists(logo):
                st.image(logo, width=70)
            else:
                st.markdown("### ⚽")
            st.markdown(f"**{eda.TEAM_NAMES[team]}**")
            st.caption(f"Our record vs them: {record} · Win chance {win:.0f}%")
            picked_now = st.session_state["opp"] == team
            if st.button("✓ Scouting" if picked_now else "Scout", key=f"scout_{team}", width="stretch",
                         type="primary" if picked_now else "secondary"):
                st.session_state["opp"] = team
                st.rerun()
    opp = st.session_state["opp"]
    opp_name = eda.TEAM_NAMES[opp]
    st.divider()

    # --- prediction ---
    our_xg, their_xg = sc.expected_goals(g, prof, us, opp, venue)
    chances = sc.result_chances(our_xg, their_xg)
    st.subheader(f"{our_name} vs {opp_name}: what to expect")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chance to win", f"{chances['Win']:.0f}%")
    m2.metric("Chance to tie", f"{chances['Tie']:.0f}%")
    m3.metric("Chance to lose", f"{chances['Loss']:.0f}%")
    m4.metric("Expected score", f"{our_xg:.1f} – {their_xg:.1f}")

    bar = pd.DataFrame({"Result": list(chances), "Chance": [round(v, 1) for v in chances.values()],
                        "game": ""})
    st.altair_chart(
        alt.Chart(bar).mark_bar(height=34).encode(
            x=alt.X("Chance:Q", stack="normalize", axis=None),
            y=alt.Y("game:N", axis=None),
            color=alt.Color("Result:N", scale=RESULT_COLORS, legend=alt.Legend(orient="bottom", title=None)),
            order=alt.Order("order:Q"),
            tooltip=["Result", alt.Tooltip("Chance:Q", format=".0f", title="Chance (%)")],
        ).transform_calculate(order="indexof(['Win','Tie','Loss'], datum.Result)").properties(height=60),
        width="stretch")
    scores = ", ".join(f"**{a}-{b}** ({p:.0f}%)" for a, b, p in sc.likely_scores(our_xg, their_xg))
    st.markdown(f"Most likely final scores: {scores}")

    # --- head to head ---
    h2h = sc.head_to_head(games, us, opp)
    st.subheader("Past meetings")
    if h2h.empty:
        st.info(f"{our_name} hasn't played {opp_name} in our data.")
    else:
        w, t, l = [(h2h["result"] == r).sum() for r in ["W", "T", "L"]]
        st.markdown(f"**{w} W, {t} T, {l} L** in {len(h2h)} games "
                    f"(goals {h2h['goals_for'].sum()}–{h2h['goals_against'].sum()}).")
        show = h2h.rename(columns=GAME_LABELS)
        show["Date"] = show["Date"].dt.strftime("%b %d, %Y")
        show["Where"] = show["Where"].str.title()
        show["Result"] = show["Result"].map({"W": "Win", "T": "Tie", "L": "Loss"})
        st.dataframe(show, hide_index=True, width="stretch")

    # --- side by side ---
    st.subheader("How the two teams compare")
    st.caption(f"Rank among the {len(prof)} conference teams (1 = best). Hover a dot for the real number.")
    ranks = sc.conference_ranks(prof)
    rows = []
    for stat, label in sc.LABELS.items():
        for team, color_name in [(us, our_name), (opp, opp_name)]:
            rows.append({"Stat": label, "Team": color_name, "Rank": int(ranks.loc[team, stat]),
                         "Value": float(prof.loc[team, stat])})
    comp = pd.DataFrame(rows)
    st.altair_chart(
        alt.Chart(comp).mark_circle(size=220, opacity=0.9).encode(
            x=alt.X("Rank:Q", scale=alt.Scale(domain=[len(prof) + 0.5, 0.5]),
                    axis=alt.Axis(values=list(range(1, len(prof) + 1)), title="← worse        rank in conference        better →")),
            y=alt.Y("Stat:N", sort=list(sc.LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("Team:N", scale=alt.Scale(domain=[our_name, opp_name], range=[US_COLOR, THEM_COLOR]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Team", "Stat", alt.Tooltip("Value:Q", format=".2f"), "Rank"],
        ).properties(height=420),
        width="stretch")

    # --- strengths and weaknesses ---
    strong, weak, total = sc.strengths_weaknesses(prof, opp)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"#### 💪 What {opp_name} is good at")
        if not strong:
            st.write("Nothing stands out: they are average or below in every stat.")
        for label, rank, value in strong:
            st.success(f"**{label}**: {value:.1f} (#{rank} of {total})")
    with s2:
        st.markdown(f"#### 🎯 Where {opp_name} is weak")
        if not weak:
            st.write("No clear weakness: they are in the top half in every stat.")
        for label, rank, value in weak:
            st.error(f"**{label}**: {value:.1f} (#{rank} of {total})")

    # --- when goals happen ---
    st.subheader(f"⏱️ When {opp_name} scores and concedes")
    timing = sc.goal_timing(goals, opp, picked, conf_only)
    if timing.sum().sum() == 0:
        st.info("No box scores for these settings.")
    else:
        t = timing.reset_index().melt(id_vars="window", var_name="Type", value_name="Goals")
        t["Share"] = t["Goals"] / t.groupby("Type")["Goals"].transform("sum").replace(0, 1) * 100
        st.altair_chart(
            alt.Chart(t).mark_bar().encode(
                x=alt.X("window:N", sort=sc.WINDOW_NAMES, title="Minute of the game", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("Type:N", sort=["Goals scored", "Goals allowed"]),
                y=alt.Y("Share:Q", title="% of their goals"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Goals scored", "Goals allowed"],
                                                          range=[THEM_COLOR, "#7b8794"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("window:N", title="Minutes"), "Type", "Goals",
                         alt.Tooltip("Share:Q", format=".0f", title="% of their goals")],
            ).properties(height=260),
            width="stretch")
        st.caption(f"From {timing['Goals scored'].sum()} goals scored and {timing['Goals allowed'].sum()} allowed "
                   "in the box scores of the seasons you picked.")

    # --- who to watch ---
    st.subheader(f"👀 Who to watch on {opp_name}")
    danger = sc.dangerous_players(goals, opp, picked, conf_only)
    if danger.empty:
        st.info("No goals in the box scores for these settings.")
    else:
        d = danger.reset_index().rename(columns={"index": "Player"})
        st.dataframe(d, hide_index=True, width="stretch")

    # --- game plan ---
    st.subheader("📋 Game plan")
    for tip in sc.game_plan(prof, us, opp, timing):
        st.markdown(f"- {tip}")

    # --- AI scouting report (the LLM part) ---
    st.subheader("📝 AI scouting report")
    st.caption("A small open-source language model (Qwen2.5, running on this computer) writes a report "
               "from the numbers above. It only gets those facts, so it can't look anything up.")
    report_key = f"report_{us}_{opp}_{venue}_{picked}_{conf_only}"
    if st.button("✍️ Write the scouting report", type="primary"):
        try:
            import llm
        except ImportError:
            st.error("The LLM libraries aren't installed. Run `pip install transformers accelerate torch` and restart.")
        else:
            facts = llm.build_facts(our_name, opp_name, venue, chances, (our_xg, their_xg), h2h,
                                    strong, weak, timing, danger, sc.game_plan(prof, us, opp, timing))
            with st.spinner("Writing the report (about 20-60 seconds, longer the first time)..."):
                generator = get_llm()
                st.session_state[report_key] = llm.scouting_report(generator, facts)
    # the online version has no AI libraries (the model is too big for the free server), so hide the button there
    has_llm = importlib.util.find_spec("transformers") is not None
    if not has_llm:
        st.info("The AI report only runs on the laptop version of this app. "
                "The model is about 3 GB, too big for the free web server.")
    elif st.button("✍️ Write the scouting report", type="primary"):
        import llm
        facts = llm.build_facts(our_name, opp_name, venue, chances, (our_xg, their_xg), h2h,
                                strong, weak, timing, danger, sc.game_plan(prof, us, opp, timing))
        with st.spinner("Writing the report (about 20-60 seconds, longer the first time)..."):
            generator = get_llm()
            st.session_state[report_key] = llm.scouting_report(generator, facts)
    if report_key in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state[report_key])