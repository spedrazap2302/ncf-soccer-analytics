# Scouting functions: compare two teams and estimate the chances of each result.
# Everything here is plain pandas + a Poisson model (no machine learning).
#
# Poisson in one sentence: if a team is expected to score 1.5 goals, the Poisson formula
# gives the chance it scores exactly 0, 1, 2, 3... goals. Doing that for both teams and
# multiplying gives the chance of every final score (0-0, 1-0, 2-1...), and adding those up
# gives the chance of a win, a tie or a loss.
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
    # win % and points are results, not style. Cards and fouls are discipline: being clean isn't
    # something an opponent has to plan for, so those go in the game plan instead
    skip = ["win_pct", "points_pg", "yellows_for", "fouls_for", "fouls_against"]
    ranks = ranks.drop(skip).sort_values()
    total = len(prof)
    # only call it a strength if they are top 3 in the conference (and a weakness if bottom 3)
    strong = [(LABELS[s], int(r), float(prof.loc[team, s])) for s, r in ranks.head(n).items() if r <= 3]
    weak = [(LABELS[s], int(r), float(prof.loc[team, s])) for s, r in ranks.tail(n)[::-1].items() if r > total - 3]
    return strong, weak, total


def rank_words(label, rank, total):
    # "#2 of 9" for goals allowed -> "2nd lowest of 9", so it's clear which way is good
    stat = {v: k for k, v in LABELS.items()}[label]
    ordinal = {1: "", 2: "2nd ", 3: "3rd "}.get(rank, f"{rank}th ")
    if rank > 3:  # weaknesses: count from the other end ("2nd highest goals allowed")
        back = total - rank + 1
        ordinal = {1: "", 2: "2nd ", 3: "3rd "}.get(back, f"{back}th ")
        word = "highest" if stat in LOWER_IS_BETTER else "lowest"
    else:
        word = "lowest" if stat in LOWER_IS_BETTER else "highest"
    return f"{ordinal}{word} of {total}"


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


def add_roster(danger, rosters, team):
    # add each player's class year and whether he is back next season (from roster.py)
    d = danger.copy()
    r = rosters[rosters["team"] == team]
    if r.empty:
        return d
    from eda import simple_name
    by_name = dict(zip(r["key"], r["class"]))
    # last names that appear only once on the roster, for names written differently in box scores
    last = r["key"].str.split().str[-1]
    by_last = dict(zip(last[~last.duplicated(keep=False)], r["class"][~last.duplicated(keep=False)]))
    classes = []
    for player in d.index:
        key = simple_name(player)
        classes.append(by_name.get(key) or by_last.get(key.split()[-1] if key else ""))
    d["Class (2026)"] = classes
    def next_season(c):
        if pd.isna(c):
            return "Already gone"  # played before 2026 but isn't on the 2026 roster
        return "❌ Leaving" if c in ["Senior", "Grad"] else "✅ Back"
    d["Next season"] = d["Class (2026)"].map(next_season)
    d["Class (2026)"] = d["Class (2026)"].fillna("-")
    return d


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
    points = {"W": 3, "T": 1, "L": 0}
    rows = []
    for _, game in conf[conf["season"].isin(test_seasons)].iterrows():
        past = conf[conf["date"] < game["date"]]
        prof = team_profile(past, half_life)
        if game["team"] not in prof.index or game["opp_key"] not in prof.index:
            continue
        chances = result_chances(*expected_goals(past, prof, game["team"], game["opp_key"], game["home_away"]))
        guess = max(chances, key=chances.get)
        real = {"W": "Win", "T": "Tie", "L": "Loss"}[game["result"]]
        # simple rules to compare the model against:
        # 1) the home team always wins (away teams always lose)
        home_pick = "Win" if game["home_away"] == "home" else "Loss"
        # 2) the team with more points per game in the earlier conference games wins
        #    (if they are level, the home team wins)
        ppg = past.assign(pts=past["result"].map(points)).groupby("team")["pts"].mean()
        ours, theirs = ppg.get(game["team"]), ppg.get(game["opp_key"])
        record_pick = "Win" if ours > theirs else "Loss" if ours < theirs else home_pick
        rows.append({"season": game["season"], "team": game["team"], "opponent": game["opp_key"],
                     "home_away": game["home_away"], "predicted": guess, "actual": real,
                     "correct": guess == real,
                     "home_pick_correct": home_pick == real,
                     "record_pick_correct": record_pick == real})
    return pd.DataFrame(rows)


def season_review(games, team, season, first_season=2024, half_life=None):
    # one team's season, game by game: what the model said BEFORE each game vs what happened
    # (same idea as backtest, but keeps the chances and the score so we can show them)
    conf = games[games["conference_game"] & (games["season"] >= first_season)]
    rows = []
    for _, game in conf[(conf["team"] == team) & (conf["season"] == season)].sort_values("date").iterrows():
        past = conf[conf["date"] < game["date"]]
        prof = team_profile(past, half_life)
        if team not in prof.index or game["opp_key"] not in prof.index:
            continue
        our_xg, their_xg = expected_goals(past, prof, team, game["opp_key"], game["home_away"])
        chances = result_chances(our_xg, their_xg)
        guess = max(chances, key=chances.get)
        real = {"W": "Win", "T": "Tie", "L": "Loss"}[game["result"]]
        rows.append({"date": game["date"], "opponent": TEAM_NAMES.get(game["opp_key"], game["opp_key"]),
                     "home_away": game["home_away"], "win": chances["Win"], "tie": chances["Tie"],
                     "loss": chances["Loss"], "expected": f"{our_xg:.1f} - {their_xg:.1f}",
                     "predicted": guess, "score": f"{game['goals_for']}-{game['goals_against']}",
                     "actual": real, "correct": guess == real,
                     "sog": f"{game['sog_for']:.0f}-{game['sog_against']:.0f}"})
    return pd.DataFrame(rows)


def clearly(value, average, higher=True, margin=0.10):
    # True only when a stat is clearly above (or below) the conference average,
    # so we don't give advice over tiny differences like 73% vs 71%
    if pd.isna(value) or pd.isna(average) or average == 0:
        return False
    change = (value - average) / abs(average)
    return change > margin if higher else change < -margin


def recent_form(games, team, n=5):
    # the team's last n conference games, newest last, like "W W T L W"
    g = games[(games["team"] == team) & games["conference_game"]].sort_values("date").tail(n)
    return list(g["result"])


def game_plan(prof, team, opp, timing=None, games=None, danger=None):
    # simple rules that turn the numbers into advice, grouped like a coach's game-week notes.
    # Returns {section: [tips]}. A tip only shows up when the difference is clear (see clearly()).
    us, them = prof.loc[team], prof.loc[opp]
    avg = prof.mean(numeric_only=True)
    plan = {"When we have the ball": [], "When they have the ball": [], "Set pieces and discipline": [],
            "Game management": []}
    attack, defend = plan["When we have the ball"], plan["When they have the ball"]
    sets, manage = plan["Set pieces and discipline"], plan["Game management"]

    # --- attacking them ---
    if clearly(them["save_pct"], avg["save_pct"], higher=False, margin=0.04):
        attack.append(f"Test their keeper: they save only {them['save_pct']:.0f}% of shots on target "
                      f"(conference average {avg['save_pct']:.0f}%). Shoot early, from distance too, and follow up rebounds.")
    elif clearly(them["save_pct"], avg["save_pct"], margin=0.04):
        attack.append(f"Their keeper is one of the better ones ({them['save_pct']:.0f}% saves vs {avg['save_pct']:.0f}% "
                      "average). Work the ball into the box for clear chances instead of shooting from range.")
    if clearly(them["shots_against"], avg["shots_against"]):
        attack.append(f"They give up a lot of shots ({them['shots_against']:.1f} per game vs {avg['shots_against']:.1f} "
                      "average). Be direct and get bodies in the box.")
    elif clearly(them["shots_against"], avg["shots_against"], higher=False):
        attack.append(f"They are hard to shoot against ({them['shots_against']:.1f} shots allowed per game). "
                      "Be patient, move them side to side and wait for the gap.")
    if clearly(us["shot_accuracy"], avg["shot_accuracy"], higher=False):
        attack.append(f"For us: only {us['shot_accuracy']:.0f}% of our shots hit the target "
                      f"(average {avg['shot_accuracy']:.0f}%). Pick better moments to shoot and work it closer.")

    # --- defending them ---
    if danger is not None and len(danger) and them["goals_for"] > 0 and games is not None:
        top = danger.index[0]
        total = danger["Goals"].sum()
        if total and danger["Goals"].iloc[0] / total >= 0.3:
            defend.append(f"{top} is their main threat ({danger['Goals'].iloc[0]} of the goals in our data). "
                          "Know where he is at all times and don't let him turn.")
    if clearly(them["shots_for"], avg["shots_for"]):
        line = f"They shoot a lot ({them['shots_for']:.1f} per game vs {avg['shots_for']:.1f} average). Stay compact and block shots."
        if clearly(us["shots_against"], avg["shots_against"]):
            line += f" This is a problem matchup: we also allow {us['shots_against']:.1f} shots per game."
        defend.append(line)
    if clearly(them["finishing"], avg["finishing"]):
        defend.append(f"They are clinical: {them['finishing']:.0f}% of their shots on target go in "
                      f"(average {avg['finishing']:.0f}%). Don't give their forwards free looks.")
    elif clearly(them["finishing"], avg["finishing"], higher=False):
        defend.append(f"They waste chances: only {them['finishing']:.0f}% of their shots on target go in. "
                      "Stay calm if they have spells of pressure.")
    if clearly(us["goals_against"], them["goals_for"], margin=0.0):
        defend.append(f"We have been allowing {us['goals_against']:.1f} goals per game, more than the "
                      f"{them['goals_for']:.1f} they usually score. Stay organized early.")

    # --- set pieces and discipline ---
    if clearly(them["corners_for"], avg["corners_for"]):
        sets.append(f"They win {them['corners_for']:.1f} corners per game (average {avg['corners_for']:.1f}). "
                    "Prepare set-piece defending and clear the first ball.")
    if clearly(them["corners_against"], avg["corners_against"]):
        sets.append(f"They give away {them['corners_against']:.1f} corners per game. Have our corner routines ready.")
    if clearly(them["yellows_for"], avg["yellows_for"], margin=0.2):
        sets.append(f"They pick up {them['yellows_for']:.1f} yellows per game. Run at defenders and draw fouls.")
    elif clearly(them["fouls_for"], avg["fouls_for"]):
        sets.append(f"They commit {them['fouls_for']:.1f} fouls per game. Run at them and make free kicks count.")

    # --- game management: when they score and allow goals, and their form ---
    if timing is not None:
        scored, allowed = timing["Goals scored"], timing["Goals allowed"]
        # only trust a pattern if there are enough goals behind it
        if allowed.sum() >= 8:
            late_allowed = allowed[["61-75'", "76-90'"]].sum() / allowed.sum() * 100
            if late_allowed >= 40:
                manage.append(f"They fade late: {late_allowed:.0f}% of the goals they allow come after the 60th minute. "
                              "Keep legs fresh and push in the last half hour.")
        if scored.sum() >= 8:
            early_scored = scored[["0-15'", "16-30'"]].sum() / scored.sum() * 100
            late_scored = scored[["76-90'", "Overtime"]].sum() / scored.sum() * 100
            if early_scored >= 35:
                manage.append(f"They start fast: {early_scored:.0f}% of their goals come in the first 30 minutes. "
                              "Be switched on from kickoff.")
            if late_scored >= 25:
                manage.append(f"They keep pushing to the end: {late_scored:.0f}% of their goals come after the 75th "
                              "minute. Don't switch off when you're ahead.")
    if games is not None:
        form = recent_form(games, opp)
        if len(form) >= 3:
            wins, losses = form.count("W"), form.count("L")
            if wins >= len(form) - 1:
                manage.append(f"They are in form: {' '.join(form)} in their last {len(form)} conference games.")
            elif losses >= len(form) - 1:
                manage.append(f"They are struggling: {' '.join(form)} in their last {len(form)} conference games. "
                              "An early goal could break them.")
    return {section: tips for section, tips in plan.items() if tips}

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
    print("baseline 'home team wins' %:", round(bt["home_pick_correct"].mean() * 100, 1))
    print("baseline 'better record wins' %:", round(bt["record_pick_correct"].mean() * 100, 1))
    for section, tips in game_plan(prof, "ncf", "keiser", games=g).items():
        print(section, tips)
