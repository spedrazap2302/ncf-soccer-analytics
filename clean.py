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

def fix_player_name(raw):
    # "Johnson, Cameron  22Johnson, Cameron" -> "Cameron Johnson"
    # the cell repeats the name after the jersey number, so keep what comes before the first digit
    name = ""
    for ch in str(raw):
        if ch.isdigit():
            break
        name += ch
    name = name.strip()
    if ", " in name:
        last, first = name.split(", ", 1)
        name = first + " " + last
    return name

def split_dash(df, col, first, second):
    # "5-0" -> two number columns
    parts = df[col].astype(str).str.split("-", expand=True)
    df[first] = pd.to_numeric(parts[0], errors="coerce")
    df[second] = pd.to_numeric(parts[1], errors="coerce")
    return df

def clean_players(df):
    df = df.copy()
    # drop the summary rows at the bottom of each table
    df = df[~df["Player"].isin(["Total", "Totals", "Opponent", "Opponents"])]
    df = df[~df["Player"].astype(str).str.startswith("Team")]
    df["player"] = [fix_player_name(p) for p in df["Player"]]
    df["number"] = pd.to_numeric(df["#"], errors="coerce")
    # a row with no jersey number is a team line (e.g. "Ave Maria University"), not a player
    df = df.dropna(subset=["number"])
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce")
    df = split_dash(df, "YC-RC", "yellows", "reds")
    df = split_dash(df, "PG-PA", "pk_goals", "pk_attempts")
    df = df.rename(columns={"GP": "games", "GS": "starts", "MIN": "minutes", "G": "goals",
                            "A": "assists", "PTS": "points", "SH": "shots", "SH%": "shot_pct",
                            "SOG": "sog", "SOG%": "sog_pct", "GW": "game_winners"})
    keep = ["team", "season", "player", "number", "games", "starts", "minutes", "goals",
            "assists", "points", "shots", "shot_pct", "sog", "sog_pct", "yellows", "reds",
            "game_winners", "pk_goals", "pk_attempts"]
    return df[keep]

def to_minutes(text):
    # goalie minutes look like "522:49" (minutes:seconds)
    text = str(text)
    if ":" in text:
        m, sec = text.split(":")
        return int(m) + int(sec) / 60
    return pd.to_numeric(text, errors="coerce")

def clean_goalies(df):
    df = df.copy()
    df = df[~df["Player"].isin(["Total", "Totals", "Opponent", "Opponents"])]
    df = df[~df["Player"].astype(str).str.startswith("Team")]
    df["player"] = [fix_player_name(p) for p in df["Player"]]
    df["minutes"] = [to_minutes(m) for m in df["MIN"]]
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce")
    # some sites write shutouts as "0/1" (solo/combined), keep the first number
    df["SHO"] = pd.to_numeric(df["SHO"].astype(str).str.split("/").str[0], errors="coerce")
    df = df.rename(columns={"GP": "games", "GS": "starts", "GA": "goals_against", "GAA": "gaa",
                            "SV": "saves", "SV%": "save_pct", "W": "wins", "L": "losses",
                            "T": "ties", "SHO": "shutouts", "SF": "shots_faced"})
    keep = ["team", "season", "player", "games", "starts", "minutes", "goals_against", "gaa",
            "saves", "save_pct", "wins", "losses", "ties", "shutouts", "shots_faced"]
    # the same goalie is sometimes listed twice with the exact same line
    return df[keep].drop_duplicates()

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

    for table in ["players", "players_conf"]:
        p = clean_players(pd.read_csv(f"data/combined/{table}.csv"))
        p.to_csv(f"data/clean/{table}.csv", index=False)
        print(f"\n{table}: {p.shape}, missing values: {p.isna().sum().sum()}")
    for table in ["goalies", "goalies_conf"]:
        g = clean_goalies(pd.read_csv(f"data/combined/{table}.csv"))
        g.to_csv(f"data/clean/{table}.csv", index=False)
        print(f"{table}: {g.shape}, missing values: {g.isna().sum().sum()}")
    print(p.head())
