import glob
import os
import numpy as np
import pandas as pd

TABLES = ["team_totals", "players", "goalies", "players_conf",
          "goalies_conf", "results", "comparison"]

# "15/7" columns -> clear names for us (_for) and the opponent (_against)
PAIR_COLS = {"G": "goals", "A": "assists", "Shots": "shots", "On Goal": "sog",
             "Corners": "corners", "YC": "yellows", "RC": "reds"}

# how each conference team's name is spelled on the different sites
CONF_NAMES = {
    "ncf": ["New College of Florida", "New College of Flori", "New College"],
    "keiser": ["Keiser"],
    "stthomas": ["St. Thomas", "St. Thomas (Fla.)", "St. Thomas (FL)"],
    "scad": ["SCAD Savannah", "SCAD", "SCAD (Ga.)"],
    "southeastern": ["Southeastern", "Southeastern (Fla.)"],
    "fmu": ["Florida Memorial (FL)", "Florida Memorial", "Florida Memorial (Fla.)", "Fla. Memorial"],
    "webber": ["Webber International (FL)", "Webber International", "Webber Int'l", "Webber International (Fla.)"],
    "avemaria": ["Ave Maria (FL)", "Ave Maria", "Ave Maria (Fla.)"],
    "warner": ["Warner (FL)", "Warner", "Warner (Fla.)"],
}

def combine(table):
    # stack every team/season file of one table type into one DataFrame
    files = sorted(glob.glob(f"data/raw/*_{table}.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if table == "team_totals":
            # 2nd and 3rd columns are named after the school, so give them one shared name
            df = df.rename(columns={df.columns[1]: "team_value", df.columns[2]: "opp_value"})
        if "SHO/CBO" in df.columns:
            # some sites call shutouts SHO/CBO
            df = df.rename(columns={"SHO/CBO": "SHO"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def split_opponent(text):
    # "vs Keiser" = home game, "at Keiser" = away game
    text = str(text).strip()
    if text.startswith("at "):
        return text[3:], "away"
    if text.startswith("vs. "):
        return text[4:], "home"
    if text.startswith("vs "):
        return text[3:], "home"
    return text, "neutral"

def fix_name(name):
    # some names are cut off like "Florida Memorial (FL"
    if "(" in name and not name.endswith(")"):
        name = name + ")"
    return name

def team_key(name):
    # "Webber Int'l" -> "webber"; None if not a conference team
    for key, names in CONF_NAMES.items():
        if name in names:
            return key
    return None

def clean_games(df):
    df = df.copy()
    # text -> real dates; the "Totals" row isn't a date, so it becomes empty and gets dropped
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    # a few games are written as 2025-09-14 instead, so try that format for the ones that failed
    df["date"] = df["date"].fillna(pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce"))
    df = df.dropna(subset=["date"])

    pairs = [split_opponent(t) for t in df["Opponent"]]
    df["opponent"] = [fix_name(p[0]) for p in pairs]
    df["home_away"] = [p[1] for p in pairs]

    for col, name in PAIR_COLS.items():
        parts = df[col].astype(str).str.split("/", expand=True)
        df[name + "_for"] = pd.to_numeric(parts[0], errors="coerce")
        df[name + "_against"] = pd.to_numeric(parts[1], errors="coerce")

    df["result"] = np.select(
        [df["goals_for"] > df["goals_against"], df["goals_for"] < df["goals_against"]],
        ["W", "L"], default="T")

    keep = ["team", "season", "date", "opponent", "home_away", "result"]
    for name in PAIR_COLS.values():
        keep += [name + "_for", name + "_against"]
    return df[keep]

def add_conference_flags(games):
    games = games.copy()
    games["opp_key"] = [team_key(n) for n in games["opponent"]]
    # NCF only joined the conference in 2024 (2023 was friendlies)
    ncf_2023 = (games["season"] == 2023) & ((games["team"] == "ncf") | (games["opp_key"] == "ncf"))
    games["conference_game"] = games["opp_key"].notna() & ~ncf_2023
    return games

if __name__ == "__main__":
    os.makedirs("data/combined", exist_ok=True)
    for table in TABLES:
        df = combine(table)
        df.to_csv(f"data/combined/{table}.csv", index=False)
        print(f"{table}: {df.shape[0]} rows, {df.shape[1]} columns")

    games = clean_games(pd.read_csv("data/combined/comparison.csv"))
    games = add_conference_flags(games)
    os.makedirs("data/clean", exist_ok=True)
    games.to_csv("data/clean/games.csv", index=False)
    print("\ngames:", games.shape)
    print("missing values:", games.isna().sum().sum(), "(opp_key is empty for non-conference opponents, that's fine)")

    print("\nconference games per team and season:")
    print(games[games["conference_game"]].groupby(["team", "season"]).size().unstack())

    # any opponent that looks like a conference team but didn't match a spelling?
    unmatched = games[games["opp_key"].isna()]["opponent"].unique()
    words = ["Thomas", "Webber", "Warner", "Memorial", "Keiser", "SCAD", "Southeastern", "Ave Maria", "New College"]
    for name in unmatched:
        for w in words:
            if w in name:
                print("check spelling:", name)