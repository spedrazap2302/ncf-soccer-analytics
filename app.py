# Streamlit app: scouting the Sun Conference for New College men's soccer
# Run with: streamlit run app.py
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
    if report_key in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state[report_key])
        st.caption("Written by an AI model from the stats above. Double-check it before sharing with the team.")

    with st.expander("How is the chance to win calculated?"):
        st.markdown(
            "1. For each team we compare goals scored and goals allowed per game with the conference average. "
            "That gives an **attack strength** and a **defense strength**.\n"
            "2. Expected goals = average goals × our attack strength × their defense weakness "
            "(and the other way around), adjusted for home field.\n"
            "3. A **Poisson distribution** turns expected goals into the chance of every score from 0-0 to 10-10, "
            "and we add up the wins, ties and losses.\n\n"
            "**Does it work?** For every 2025 and 2026 conference game, we ran the model using only the "
            "games played before that day (like a coach preparing for it) and checked what happened.")
        bt = get_backtest(games, half_life)
        b1, b2 = st.columns(2)
        b1.metric("Model picked the right result", f"{bt['correct'].mean() * 100:.0f}%", help=f"{len(bt)} games")
        ha = bt[bt["home_away"].isin(["home", "away"])]
        home_rule = ((ha["home_away"] == "home") == (ha["actual"] == "Win")).mean() * 100
        b2.metric("Just guessing 'home team wins'", f"{home_rule:.0f}%")
        st.caption("Small samples (about 20 games per team) mean the numbers are a guide, not a guarantee.")

# =====================================================================
# 2. CONFERENCE RANKINGS
# =====================================================================
with tab_conf:
    st.subheader("Who is best at what?")
    stat_label = st.selectbox("Pick a stat", list(sc.LABELS.values()))
    stat = {v: k for k, v in sc.LABELS.items()}[stat_label]
    lower_better = stat in sc.LOWER_IS_BETTER
    rank_df = prof.reset_index()[["team", "team_name", stat, "games"]].rename(columns={stat: "value"})
    rank_df["highlight"] = rank_df["team"].map(lambda t: our_name if t == us else "Other teams")
    st.altair_chart(
        alt.Chart(rank_df).mark_bar().encode(
            x=alt.X("value:Q", title=stat_label),
            y=alt.Y("team_name:N", sort="x" if lower_better else "-x", title=None, axis=alt.Axis(labelLimit=250)),
            color=alt.Color("highlight:N", scale=alt.Scale(domain=[our_name, "Other teams"],
                                                           range=[US_COLOR, OTHER_COLOR]), legend=None),
            tooltip=[alt.Tooltip("team_name:N", title="Team"), alt.Tooltip("value:Q", title=stat_label, format=".2f"),
                     alt.Tooltip("games:Q", title="Games")],
        ).properties(height=330),
        width="stretch")
    st.caption("Lower is better for this stat, so the best team is on top." if lower_better
               else "Best team on top.")

    st.subheader(f"{our_name} season by season")
    table = eda.team_season_table(games, conference_only=conf_only)
    ours = table[table["team_name"] == our_name]
    long = ours.melt(id_vars="season", value_vars=["goals_for_pg", "goals_against_pg"],
                     var_name="stat", value_name="per game")
    long["stat"] = long["stat"].map({"goals_for_pg": "Goals scored", "goals_against_pg": "Goals allowed"})
    st.altair_chart(
        alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=90)).encode(
            x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("per game:Q", title="Goals per game"),
            color=alt.Color("stat:N", scale=alt.Scale(domain=["Goals scored", "Goals allowed"],
                                                      range=[US_COLOR, THEM_COLOR]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["season", "stat", alt.Tooltip("per game:Q", format=".2f")],
        ).properties(height=280),
        width="stretch")

    with st.expander("Full standings table"):
        nice = table.rename(columns={"team_name": "Team", "season": "Season", "games": "Games", "wins": "Wins",
                                     "ties": "Ties", "losses": "Losses", "points_per_game": "Points per game",
                                     "goals_for_pg": "Goals scored per game", "goals_against_pg": "Goals allowed per game",
                                     "shots_pg": "Shots per game", "sog_pg": "Shots on target per game"})
        st.dataframe(nice.sort_values(["Season", "Points per game"], ascending=[False, False]),
                     hide_index=True, width="stretch")

# =====================================================================
# 3. WHAT WINS GAMES
# =====================================================================
with tab_wins:
    st.subheader("Which stats go with winning?")
    st.caption("Correlation with goal difference, using every game in the settings. "
               "Longer bar = stronger link. Green = helps you win, red = goes with losing.")
    corr = eda.correlation_table(g)["goal_diff"].drop(["goal_diff", "points", "goals_for", "goals_against"])
    corr_df = corr.reset_index()
    corr_df.columns = ["stat", "corr"]
    corr_df["Stat"] = corr_df["stat"].map(GAME_LABELS)
    corr_df["Effect"] = corr_df["corr"].map(lambda c: "Goes with winning" if c > 0 else "Goes with losing")
    corr_df["Strength"] = corr_df["corr"].abs().map(
        lambda c: "strong" if c >= 0.5 else "moderate" if c >= 0.3 else "weak")
    st.altair_chart(
        alt.Chart(corr_df).mark_bar().encode(
            x=alt.X("corr:Q", title="Link with goal difference (-1 to 1)", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("Stat:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
            color=alt.Color("Effect:N", scale=alt.Scale(domain=["Goes with winning", "Goes with losing"],
                                                        range=["#2e9e5b", "#d64545"]), legend=None),
            tooltip=["Stat", alt.Tooltip("corr:Q", format=".2f", title="Correlation"), "Strength"],
        ).properties(height=300),
        width="stretch")
    best = corr_df.sort_values("corr", ascending=False).iloc[0]
    worst = corr_df.sort_values("corr").iloc[0]
    st.info(f"**Takeaway:** {best['Stat']} has the strongest link with winning ({best['corr']:.2f}). "
            f"{worst['Stat']} goes most with losing ({worst['corr']:.2f}). "
            "Getting shots on target matters more than just shooting a lot.")

    st.subheader("Does home field matter?")
    ha = eda.home_away_table(g).reset_index()
    ha["Where"] = ha["home_away"].str.title()
    hc1, hc2 = st.columns([1, 1])
    hc1.altair_chart(
        alt.Chart(ha).mark_bar(size=60).encode(
            x=alt.X("Where:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("win_pct:Q", title="Win %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Where:N", scale=alt.Scale(domain=["Home", "Away"], range=[US_COLOR, OTHER_COLOR]),
                            legend=None),
            tooltip=["Where", alt.Tooltip("win_pct:Q", format=".0f", title="Win %"), alt.Tooltip("games:Q", title="Games")],
        ).properties(height=260),
        width="stretch")
    if len(ha) == 2:
        home_row = ha.set_index("home_away").loc["home"]
        away_row = ha.set_index("home_away").loc["away"]
        hc2.metric("Home teams win", f"{home_row['win_pct']:.0f}%")
        hc2.metric("Away teams win", f"{away_row['win_pct']:.0f}%")
        hc2.caption(f"Home teams also score {home_row['goals_for_pg']:.1f} goals per game vs "
                    f"{away_row['goals_for_pg']:.1f} away.")

    st.subheader("Does the heat matter?")
    st.caption("Game-day high temperature from the Open-Meteo weather API, at the home team's field.")
    heat = eda.weather_table(g)
    if heat.empty:
        st.info("No weather for these settings.")
    else:
        st.altair_chart(
            alt.Chart(heat).mark_bar(size=50).encode(
                x=alt.X("heat:N", sort=list(heat["heat"]), title="Game-day high", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("goals_per_game:Q", title="Goals per game (both teams)"),
                color=alt.value(US_COLOR),
                tooltip=[alt.Tooltip("heat:N", title="High"), alt.Tooltip("goals_per_game:Q", format=".2f",
                         title="Goals per game"), alt.Tooltip("games:Q", title="Team-games")],
            ).properties(height=260),
            width="stretch")
        most = heat.loc[heat["goals_per_game"].idxmax()]
        fewest = heat.loc[heat["goals_per_game"].idxmin()]
        st.info(f"**Takeaway:** the most goals come on {most['heat']} days ({most['goals_per_game']:.1f} per game) "
                f"and the fewest on {fewest['heat']} days ({fewest['goals_per_game']:.1f}). The smallest group has "
                f"only {heat['games'].min()} team-games, so treat this as a hint, not a rule.")

# =====================================================================
# 4. PLAYERS
# =====================================================================
with tab_players:
    st.subheader("Player leaderboard")
    pl = players[players["season"].isin(picked)]
    pc1, pc2 = st.columns(2)
    team_pick = pc1.selectbox("Team", ["All teams"] + list(eda.TEAM_NAMES.values()), index=1 + list(eda.TEAM_NAMES).index(us))
    stat_names = {"Goals": "goals", "Assists": "assists", "Points (2 per goal, 1 per assist)": "points",
                  "Shots": "shots", "Shots on target": "sog", "Games played": "games"}
    stat_pick = pc2.selectbox("Stat", list(stat_names))
    col = stat_names[stat_pick]
    if team_pick != "All teams":
        pl = pl[pl["team_name"] == team_pick]
    top = pl.sort_values(col, ascending=False).head(10).copy()
    if top.empty or top[col].max() == 0:
        st.info("No players with this stat for these settings.")
    else:
        top["label"] = top["player"] + " (" + top["season"].astype(str) + ")"
        top["highlight"] = top["team"].map(lambda t: our_name if t == us else "Other teams")
        st.altair_chart(
            alt.Chart(top).mark_bar().encode(
                x=alt.X(f"{col}:Q", title=stat_pick),
                y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
                color=alt.Color("highlight:N", scale=alt.Scale(domain=[our_name, "Other teams"],
                                                               range=[US_COLOR, OTHER_COLOR]), legend=None),
                tooltip=[alt.Tooltip("player:N", title="Player"), alt.Tooltip("team_name:N", title="Team"),
                         alt.Tooltip("season:O", title="Season"), alt.Tooltip("goals:Q", title="Goals"),
                         alt.Tooltip("assists:Q", title="Assists"), alt.Tooltip("shots:Q", title="Shots"),
                         alt.Tooltip("games:Q", title="Games")],
            ).properties(height=340),
            width="stretch")

    st.subheader("Most clinical finishers")
    st.caption("Goals per shot, for players with at least 15 shots.")
    eff = eda.efficient_shooters(pl if team_pick != "All teams" else players[players["season"].isin(picked)])
    if eff.empty:
        st.info("No player has 15+ shots for these settings.")
    else:
        eff = eff.rename(columns={"player": "Player", "team_name": "Team", "season": "Season", "goals": "Goals",
                                  "shots": "Shots", "goals_per_shot": "Goals per shot"})
        st.dataframe(eff, hide_index=True, width="stretch")

# =====================================================================
# 5. ABOUT THE DATA
# =====================================================================
with tab_data:
    st.subheader("Where the data comes from")
    st.markdown(
        "- Scraped with `scraper.py` from the stats page of each Sun Conference school's athletics website "
        "(New College, Keiser, St. Thomas, SCAD Savannah, Southeastern, Florida Memorial, Webber, Ave Maria, Warner).\n"
        "- Box scores of every game scraped with `boxscores.py` (goal minutes, scorers, assists, fouls).\n"
        "- Game-day weather from the **Open-Meteo** historical weather API with `weather.py`.\n"
        "- Cleaned with `clean.py`: names fixed, dates unified, scores split into for/against, "
        "conference games tagged, box scores and weather matched to each game.\n"
        f"- **{len(games)} games**, **{len(goals)} goals** with the minute and scorer, and "
        f"**{len(players)} player seasons**, 2023 to 2026.")

    st.subheader("Missing data")
    st.markdown("The main stats have **no empty cells** after cleaning. The gaps are whole games: "
                "New College has no 2023 conference games (it joined in 2024), and 2026 is still being played. "
                f"Box score extras are missing for {games['fouls_for'].isna().sum()} games, and weather is missing for "
                f"{games['temp_max_f'].isna().sum()} games played away from the 9 conference fields (we only know "
                "where those 9 fields are).")
    counts = eda.games_per_team_season(games).reset_index().melt(id_vars="team_name", var_name="season",
                                                                 value_name="games")
    st.altair_chart(
        alt.Chart(counts).mark_rect().encode(
            x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("team_name:N", title=None, axis=alt.Axis(labelLimit=250)),
            color=alt.Color("games:Q", scale=alt.Scale(scheme="teals"), legend=alt.Legend(title="Games")),
            tooltip=[alt.Tooltip("team_name:N", title="Team"), "season", "games"],
        ).properties(height=300) +
        alt.Chart(counts).mark_text(fontSize=13).encode(x="season:O", y="team_name:N", text="games:Q"),
        width="stretch")

    st.subheader("Outliers")
    out, (low, high) = eda.outlier_games(games)
    st.markdown(f"Using the IQR rule, a game is an outlier if the goal difference is below {low:.1f} or above "
                f"{high:.1f}. That flags **{len(out)} blowouts**. We keep them because they are real results, "
                "but they pull averages up for strong teams.")
    box_col, _ = st.columns([3, 2])
    box_col.pyplot(eda.goal_diff_boxplot(games))
    with st.expander("See the blowout games"):
        o = out.rename(columns=GAME_LABELS)
        o["Date"] = o["Date"].dt.strftime("%b %d, %Y")
        st.dataframe(o, hide_index=True, width="stretch")

    with st.expander("Summary statistics for every game"):
        st.dataframe(eda.describe_games(games).rename(columns=GAME_LABELS), width="stretch")
