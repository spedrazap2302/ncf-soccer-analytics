# EDA functions for the NCF soccer project.
# Each function returns a table (DataFrame) or a chart (matplotlib figure),
# so the Streamlit app can show them.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAM_NAMES = {
    "ncf": "New College", "keiser": "Keiser", "stthomas": "St. Thomas",
    "scad": "SCAD Savannah", "southeastern": "Southeastern", "fmu": "Florida Memorial",
    "webber": "Webber International", "avemaria": "Ave Maria", "warner": "Warner",
}

GAME_STATS = ["goals_for", "goals_against", "shots_for", "shots_against", "sog_for",
              "sog_against", "corners_for", "corners_against", "yellows_for", "yellows_against"]


def load_games(path="data/clean/games.csv"):
    games = pd.read_csv(path, parse_dates=["date"])
    games["goal_diff"] = games["goals_for"] - games["goals_against"]
    # 3 points for a win, 1 for a tie
    games["points"] = games["result"].map({"W": 3, "T": 1, "L": 0})
    games["team_name"] = games["team"].map(TEAM_NAMES)
    return games


def load_goals(path="data/clean/goals.csv"):
    goals = pd.read_csv(path, parse_dates=["date"])
    goals["team_name"] = goals["team"].map(TEAM_NAMES)
    return goals


def weather_table(games):
    # does the heat change anything? group games by the day's high temperature
    g = games[games["temp_max_f"].notna() & games["home_away"].isin(["home", "away"])].copy()
    g["heat"] = pd.cut(g["temp_max_f"], [0, 80, 85, 90, 120],
                       labels=["Under 80°F", "80-85°F", "85-90°F", "90°F+"])
    g["total_goals"] = g["goals_for"] + g["goals_against"]
    table = g.groupby("heat", observed=True).agg(
        games=("result", "size"),
        goals_per_game=("total_goals", "mean"),
        shots_per_game=("shots_for", "mean"),
    ).round(2)
    return table.reset_index()


def load_players(path="data/clean/players.csv"):
    players = pd.read_csv(path)
    players["team_name"] = players["team"].map(TEAM_NAMES)
    return players


# ---------- 1. what's in the data ----------

def overview(games, players):
    return {
        "games": len(games),
        "teams": games["team"].nunique(),
        "seasons": sorted(games["season"].unique().tolist()),
        "conference games": int(games["conference_game"].sum()),
        "player seasons": len(players),
    }


def describe_games(games):
    return games[GAME_STATS].describe().round(2)


# ---------- 2. missing data and outliers ----------

def games_per_team_season(games):
    # shows gaps: NCF 2023 had no conference games, 2026 is not finished
    return games.groupby(["team_name", "season"]).size().unstack()


def outlier_games(games):
    # IQR rule on goal difference: anything past 1.5 * IQR from the middle 50%
    q1 = games["goal_diff"].quantile(0.25)
    q3 = games["goal_diff"].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = games[(games["goal_diff"] < low) | (games["goal_diff"] > high)]
    cols = ["team_name", "season", "date", "opponent", "goals_for", "goals_against", "conference_game"]
    return out[cols].sort_values("goals_for", ascending=False), (low, high)


def goal_diff_boxplot(games):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.boxplot(games["goal_diff"], vert=False)
    ax.set_xlabel("Goal difference in a game")
    ax.set_yticks([])
    ax.set_title("Most games are close; the dots are blowouts")
    fig.tight_layout()
    return fig


# ---------- 3. what wins games ----------

def correlation_table(games):
    cols = GAME_STATS + ["goal_diff", "points"]
    return games[cols].corr().round(2)


def correlation_heatmap(games):
    corr = correlation_table(games)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, corr.iloc[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("How game stats move together")
    fig.tight_layout()
    return fig


# ---------- 4. NCF vs the conference ----------

def team_season_table(games, conference_only=True):
    g = games[games["conference_game"]] if conference_only else games
    table = g.groupby(["team_name", "season"]).agg(
        games=("result", "size"),
        wins=("result", lambda r: (r == "W").sum()),
        ties=("result", lambda r: (r == "T").sum()),
        losses=("result", lambda r: (r == "L").sum()),
        points_per_game=("points", "mean"),
        goals_for_pg=("goals_for", "mean"),
        goals_against_pg=("goals_against", "mean"),
        shots_pg=("shots_for", "mean"),
        sog_pg=("sog_for", "mean"),
    ).round(2)
    return table.reset_index()


def ncf_vs_conference_chart(games, stat="goals_for"):
    # NCF per season next to the average of the other conference teams
    g = games[games["conference_game"]]
    by_team = g.groupby(["team", "season"])[stat].mean().reset_index()
    ncf = by_team[by_team["team"] == "ncf"]
    rest = by_team[by_team["team"] != "ncf"].groupby("season")[stat].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(rest.index, rest.values, marker="o", label="Conference average")
    ax.plot(ncf["season"], ncf[stat], marker="o", label="New College")
    ax.set_xticks(sorted(g["season"].unique()))
    ax.set_ylabel(stat.replace("_", " ") + " per game")
    ax.set_title("New College vs the Sun Conference (conference games)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- 5. home vs away ----------

def home_away_table(games):
    g = games[games["conference_game"] & games["home_away"].isin(["home", "away"])]
    table = g.groupby("home_away").agg(
        games=("result", "size"),
        win_pct=("result", lambda r: (r == "W").mean() * 100),
        goals_for_pg=("goals_for", "mean"),
        goals_against_pg=("goals_against", "mean"),
    ).round(2)
    return table


# ---------- 6. players ----------

def top_scorers(players, n=10, team=None, season=None):
    p = players
    if team:
        p = p[p["team"] == team]
    if season:
        p = p[p["season"] == season]
    cols = ["player", "team_name", "season", "games", "goals", "assists", "points", "shots"]
    return p.sort_values(["goals", "points"], ascending=False)[cols].head(n)


def efficient_shooters(players, n=10, min_shots=15):
    p = players[players["shots"] >= min_shots].copy()
    p["goals_per_shot"] = (p["goals"] / p["shots"]).round(3)
    cols = ["player", "team_name", "season", "goals", "shots", "goals_per_shot"]
    return p.sort_values("goals_per_shot", ascending=False)[cols].head(n)


if __name__ == "__main__":
    # quick check from the terminal: python eda.py
    games = load_games()
    players = load_players()
    print(overview(games, players))
    print(describe_games(games))
    print(games_per_team_season(games))
    out, fence = outlier_games(games)
    print("\noutlier fence:", fence, "outlier games:", len(out))
    print(out.head())
    print(correlation_table(games)["goal_diff"].sort_values())
    print(team_season_table(games).query("team_name == 'New College'"))
    print(home_away_table(games))
    print(top_scorers(players, n=5))
    print(efficient_shooters(players, n=5))
