# Scouting functions: compare two teams and estimate the chances of each result.
# Everything here is plain pandas + a Poisson model (no machine learning).
from math import exp, factorial

import pandas as pd

from eda import TEAM_NAMES

# readable names for every stat we show in the app
LABELS = {
    "win_pct": "Win %",
    "points_pg": "Points per game",
    "goals_for": "Goals scored per game",
    "goals_against": "Goals allowed per game",
    "shots_for": "Shots per game",
    "shots_against": "Shots allowed per game",
    "sog_for": "Shots on target per game",
    "sog_against": "Shots on target allowed per game",
    "shot_accuracy": "Shot accuracy (% on target)",
    "finishing": "Finishing (% of shots on target that score)",
    "save_pct": "Keeper save %",
    "corners_for": "Corners per game",
    "corners_against": "Corners allowed per game",
    "yellows_for": "Yellow cards per game",
    "fouls_for": "Fouls committed per game",
    "fouls_against": "Fouls drawn per game",
}

# for these stats a LOWER number is better
LOWER_IS_BETTER = ["goals_against", "shots_against", "sog_against", "corners_against", "yellows_for", "fouls_for"]

# 15-minute windows for goal timing
WINDOWS = [0, 15, 30, 45, 60, 75, 90, 200]
WINDOW_NAMES = ["0-15'", "16-30'", "31-45'", "46-60'", "61-75'", "76-90'", "Overtime"]


def pick_games(games, seasons, conference_only=True):
    g = games[games["season"].isin(seasons)]
    if conference_only:
        g = g[g["conference_game"]]
    return g


def add_weights(g, half_life=None):
    # recent form: a game loses half its weight every `half_life` days
    # (half_life=None means every game counts the same)
    g = g.copy()
    if half_life is None:
        g["weight"] = 1.0
    else:
        days_ago = (g["date"].max() - g["date"]).dt.days
        g["weight"] = 0.5 ** (days_ago / half_life)
    return g


def team_profile(g, half_life=None):
    # one row per team with (weighted) per-game averages and efficiency stats
    g = add_weights(g, half_life)
    g["win"] = (g["result"] == "W") * 100
    stats = {"win_pct": "win", "points_pg": "points", "goals_for": "goals_for",
             "goals_against": "goals_against", "shots_for": "shots_for", "shots_against": "shots_against",
             "sog_for": "sog_for", "sog_against": "sog_against", "corners_for": "corners_for",
             "corners_against": "corners_against", "yellows_for": "yellows_for",
             "fouls_for": "fouls_for", "fouls_against": "fouls_against"}
    sums = pd.DataFrame({"team": g["team"]})
    weights = pd.DataFrame({"team": g["team"]})
    for name, col in stats.items():
        # a few games have no box score (no fouls), so only count the weight where there is a number
        sums[name] = g[col] * g["weight"]
        weights[name] = g["weight"].where(g[col].notna())
    prof = sums.groupby("team").sum() / weights.groupby("team").sum()
    prof.insert(0, "games", g.groupby("team").size())
    # efficiency: use totals so one weird game doesn't dominate
    prof["shot_accuracy"] = prof["sog_for"] / prof["shots_for"] * 100
    prof["finishing"] = prof["goals_for"] / prof["sog_for"] * 100
    prof["save_pct"] = (1 - prof["goals_against"] / prof["sog_against"]) * 100
    prof["team_name"] = prof.index.map(TEAM_NAMES)
    return prof.round(2)


def conference_ranks(prof):
    # rank 1 = best in the conference for that stat
    ranks = pd.DataFrame(index=prof.index)
    for stat in LABELS:
        ranks[stat] = prof[stat].rank(ascending=stat in LOWER_IS_BETTER, method="min").astype(int)
    return ranks


def strengths_weaknesses(prof, team, n=3):
    # the stats where a team ranks best and worst in the conference
    ranks = conference_ranks(prof).loc[team]
    skip = ["win_pct", "points_pg"]  # those are results, not style
    ranks = ranks.drop(skip).sort_values()
    total = len(prof)
    # only call it a strength if they are in the top half (and weakness if bottom half)
    strong = [(LABELS[s], int(r), float(prof.loc[team, s])) for s, r in ranks.head(n).items() if r <= total / 2]
    weak = [(LABELS[s], int(r), float(prof.loc[team, s])) for s, r in ranks.tail(n)[::-1].items() if r > total / 2]
    return strong, weak, total


def goal_timing(goals, team, seasons, conference_only=True):
    # how many of a team's goals (scored and allowed) come in each 15-minute window
    g = goals[(goals["team"] == team) & goals["season"].isin(seasons)]
    if conference_only:
        g = g[g["conference_game"]]
    g = g.assign(window=pd.cut(g["minute"], WINDOWS, labels=WINDOW_NAMES, right=True))
    table = g.groupby(["side", "window"], observed=False).size().unstack(0, fill_value=0)
    table = table.reindex(columns=["for", "against"], fill_value=0)
    table.columns = ["Goals scored", "Goals allowed"]
    return table


def dangerous_players(goals, team, seasons, conference_only=True, n=5):
    # who scores and assists for a team in the chosen games
    g = goals[(goals["team"] == team) & goals["season"].isin(seasons) & (goals["side"] == "for")]
    if conference_only:
        g = g[g["conference_game"]]
    scored = g.groupby("scorer").size().rename("Goals")
    assists = g[g["assist"].fillna("") != ""]
    # "A, B" means two players assisted
    names = assists["assist"].str.split(r",\s*|;\s*").explode().str.strip()
    assisted = names.value_counts().rename("Assists")
    table = pd.concat([scored, assisted], axis=1).fillna(0).astype(int)
    table = table.drop(index=["", "Team", "TEAM"], errors="ignore")
    table["Goals + assists"] = table["Goals"] + table["Assists"]
    return table.sort_values(["Goals + assists", "Goals"], ascending=False).head(n)


def head_to_head(games, team, opp):
    # games from our own website only, so each game is counted once
    h = games[(games["team"] == team) & (games["opp_key"] == opp)]
    cols = ["date", "season", "home_away", "result", "goals_for", "goals_against",
            "shots_for", "shots_against"]
    return h[cols].sort_values("date", ascending=False)


def poisson(k, lam):
    # chance of scoring exactly k goals when you expect lam goals
    return lam ** k * exp(-lam) / factorial(k)


def expected_goals(g, prof, team, opp, venue="neutral"):
    # attack strength = how much a team scores compared with the average team
    # defense weakness = how much a team allows compared with the average team
    avg = g["goals_for"].mean()
    our_xg = avg * (prof.loc[team, "goals_for"] / avg) * (prof.loc[opp, "goals_against"] / avg)
    their_xg = avg * (prof.loc[opp, "goals_for"] / avg) * (prof.loc[team, "goals_against"] / avg)
    # home advantage: home teams score more than away teams in this league
    home = g[g["home_away"] == "home"]["goals_for"].mean() / avg
    away = g[g["home_away"] == "away"]["goals_for"].mean() / avg
    if venue == "home":
        our_xg, their_xg = our_xg * home, their_xg * away
    elif venue == "away":
        our_xg, their_xg = our_xg * away, their_xg * home
    return round(float(our_xg), 2), round(float(their_xg), 2)


def result_chances(our_xg, their_xg, max_goals=10):
    # add up the chance of every scoreline from 0-0 to 10-10
    win = tie = loss = 0
    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            p = poisson(a, our_xg) * poisson(b, their_xg)
            if a > b:
                win += p
            elif a == b:
                tie += p
            else:
                loss += p
    total = win + tie + loss
    return {"Win": win / total * 100, "Tie": tie / total * 100, "Loss": loss / total * 100}


def likely_scores(our_xg, their_xg, n=3):
    scores = [(a, b, poisson(a, our_xg) * poisson(b, their_xg) * 100)
              for a in range(6) for b in range(6)]
    scores = [(a, b, round(float(p), 1)) for a, b, p in scores]
    return sorted(scores, key=lambda s: s[2], reverse=True)[:n]


def backtest(games, test_seasons=(2025, 2026), first_season=2024, half_life=None):
    # does the model work? for every conference game, use ONLY the games played
    # before that day (like a coach would) and check if the most likely result happened
    conf = games[games["conference_game"] & (games["season"] >= first_season)]
    rows = []
    for _, game in conf[conf["season"].isin(test_seasons)].iterrows():
        past = conf[conf["date"] < game["date"]]
        prof = team_profile(past, half_life)
        if game["team"] not in prof.index or game["opp_key"] not in prof.index:
            continue
        chances = result_chances(*expected_goals(past, prof, game["team"], game["opp_key"], game["home_away"]))
        guess = max(chances, key=chances.get)
        real = {"W": "Win", "T": "Tie", "L": "Loss"}[game["result"]]
        rows.append({"season": game["season"], "team": game["team"], "opponent": game["opp_key"],
                     "home_away": game["home_away"], "predicted": guess, "actual": real,
                     "correct": guess == real})
    return pd.DataFrame(rows)


def game_plan(prof, team, opp, timing=None):
    # simple rules that turn the numbers into advice
    us, them = prof.loc[team], prof.loc[opp]
    avg = prof.mean(numeric_only=True)
    tips = []
    if them["save_pct"] < avg["save_pct"]:
        tips.append(f"Test their keeper: they save only {them['save_pct']:.0f}% of shots on target "
                    f"(conference average {avg['save_pct']:.0f}%). Shoot from distance and follow up rebounds.")
    else:
        tips.append(f"Their keeper is good ({them['save_pct']:.0f}% saves vs {avg['save_pct']:.0f}% average). "
                    "Work for high-quality chances instead of shooting from anywhere.")
    if them["shots_against"] > avg["shots_against"]:
        tips.append(f"They give up a lot of shots ({them['shots_against']:.1f} per game). Be direct and get bodies in the box.")
    if them["shots_for"] > avg["shots_for"]:
        tips.append(f"They shoot a lot ({them['shots_for']:.1f} per game). Stay compact and block shots.")
    if them["finishing"] > avg["finishing"]:
        tips.append(f"They are clinical: {them['finishing']:.0f}% of their shots on target go in. "
                    "Don't give their forwards free looks.")
    if them["corners_for"] > avg["corners_for"]:
        tips.append(f"They win {them['corners_for']:.1f} corners per game. Prepare set-piece defending.")
    if them["yellows_for"] > avg["yellows_for"]:
        tips.append(f"They pick up {them['yellows_for']:.1f} yellows per game. Run at defenders and draw fouls.")
    if us["shot_accuracy"] < avg["shot_accuracy"]:
        tips.append(f"For us: only {us['shot_accuracy']:.0f}% of our shots hit the target. "
                    "Pick better moments to shoot.")
    if them["fouls_for"] > avg["fouls_for"] and them["yellows_for"] <= avg["yellows_for"]:
        tips.append(f"They commit {them['fouls_for']:.1f} fouls per game. Run at them and make set pieces count.")
    if timing is not None:
        scored, allowed = timing["Goals scored"], timing["Goals allowed"]
        # only trust a pattern if there are enough goals behind it
        if allowed.sum() >= 8:
            late_allowed = allowed[["61-75'", "76-90'"]].sum() / allowed.sum() * 100
            if late_allowed >= 40:
                tips.append(f"They fade late: {late_allowed:.0f}% of the goals they allow come after the 60th minute. "
                            "Keep legs fresh and push in the last half hour.")
        if scored.sum() >= 8:
            early_scored = scored[["0-15'", "16-30'"]].sum() / scored.sum() * 100
            late_scored = scored[["76-90'", "Overtime"]].sum() / scored.sum() * 100
            if early_scored >= 35:
                tips.append(f"They start fast: {early_scored:.0f}% of their goals come in the first 30 minutes. "
                            "Be switched on from kickoff.")
            if late_scored >= 25:
                tips.append(f"They keep pushing to the end: {late_scored:.0f}% of their goals come after the 75th minute. "
                            "Don't switch off when you're ahead.")
    if us["goals_against"] > them["goals_for"]:
        tips.append("Our defense has been allowing more goals than they usually score. Stay organized early.")
    return tips


if __name__ == "__main__":
    # quick check: python scouting.py
    from eda import load_games
    games = load_games()
    g = pick_games(games, [2025, 2026])
    prof = team_profile(g)
    print(prof[["team_name", "games", "win_pct", "goals_for", "goals_against", "finishing", "save_pct"]])
    for opp in ["keiser", "warner"]:
        for venue in ["home", "away"]:
            xg = expected_goals(g, prof, "ncf", opp, venue)
            print(opp, venue, xg, {k: round(v) for k, v in result_chances(*xg).items()}, likely_scores(*xg))
    print(strengths_weaknesses(prof, "keiser"))
    print(head_to_head(games, "ncf", "keiser"))
    for hl in [None, 180, 365]:
        bt = backtest(games, half_life=hl)
        print("half life", hl, "games:", len(bt), "correct %:", round(bt["correct"].mean() * 100, 1))
    ha = bt[bt["home_away"].isin(["home", "away"])]
    print("baseline 'home team wins' %:", round(((ha["home_away"] == "home") == (ha["actual"] == "Win")).mean() * 100, 1))
    for t in game_plan(prof, "ncf", "keiser"):
        print("-", t)
