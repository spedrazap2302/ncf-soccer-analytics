import os
import time
import requests
import pandas as pd
from io import StringIO

HEADERS = {"User-Agent": "Mozilla/5.0"}

# name: (columns to look for, which match to take: 0 = first, 1 = second)
WANTED = {
    "team_totals": (["Statistic"], 0),
    "players": (["Player", "G", "A", "SOG"], 0),
    "goalies": (["Player", "GA", "SV"], 0),
    "players_conf": (["Player", "G", "A", "SOG"], 1),
    "goalies_conf": (["Player", "GA", "SV"], 1),
    "results": (["Opponent", "W/L", "Score"], 0),
    "comparison": (["Opponent", "Corners", "YC"], 0),
}

def get_tables(base_url, season):
    url = f"{base_url}/sports/mens-soccer/stats/{season}"
    response = requests.get(url, headers=HEADERS, timeout=20)
    if response.status_code != 200:
        print("Could not load", url, "status:", response.status_code)
        return []
    return pd.read_html(StringIO(response.text))

def find_table(tables, cols, n=0):
    # collect every table that has the columns, then take the n-th one
    matches = []
    for t in tables:
        names = [str(c) for c in t.columns]
        if all(c in names for c in cols):
            matches.append(t)
    if n < len(matches):
        return matches[n]
    return None

def scrape_season(team, base_url, season):
    tables = get_tables(base_url, season)
    os.makedirs("data/raw", exist_ok=True)
    for name, (cols, n) in WANTED.items():
        t = find_table(tables, cols, n)
        if t is None:
            print(f"{team} {season}: no {name} table")
            continue
        t["team"] = team
        t["season"] = season
        t.to_csv(f"data/raw/{team}_{season}_{name}.csv", index=False)
        print(f"saved {team}_{season}_{name}.csv ({len(t)} rows)")

if __name__ == "__main__":
    for season in [2023, 2024, 2025, 2026]:
        scrape_season("ncf", "https://gomightybanyans.com", season)
        time.sleep(2)  # wait between requests so we don't hammer the site