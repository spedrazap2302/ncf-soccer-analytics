# Download each team's roster (name, number, position, class year) from its athletics site
# Saves data/raw/rosters/{team}_{season}.csv
# Run with: python roster.py
import os
import time
from io import StringIO

import pandas as pd
import requests
from lxml import html

from scraper import HEADERS, TEAMS

SEASON = 2026  # the season that just finished: seniors and grad students on it are leaving

# different sites call the class column different things
CLASS_COLUMNS = ["Cl.", "Class", "Yr.", "Year", "Academic Year", "Acad. Yr.", "Elig.", "Eligibility"]


def from_table(page_text):
    # the "table view" of the roster (?view=2) is a normal HTML table, so pandas can read it
    try:
        tables = pd.read_html(StringIO(page_text))
    except ValueError:  # no tables on the page
        return None
    for t in tables:
        cols = [str(c) for c in t.columns]
        name_col = next((c for c in cols if c in ["Name", "Full Name", "Player"]), None)
        class_col = next((c for c in cols if c in CLASS_COLUMNS), None)
        if name_col and class_col:
            out = pd.DataFrame({"player": t[name_col], "year": t[class_col]})
            for col, new in [("#", "number"), ("No.", "number"), ("Pos.", "position"), ("Position", "position")]:
                if col in cols:
                    out[new] = t[col]
            return out
    return None


def from_cards(page_text):
    # the normal roster view is a list of player "cards"; read the name and class from each one
    # (xpath = a path through the page's HTML, like a folder path; see boxscores.py)
    tree = html.fromstring(page_text)
    rows = []
    for card in tree.xpath('//li[contains(@class, "sidearm-roster-player")]'):
        name = " ".join(card.xpath('.//*[contains(@class, "sidearm-roster-player-name")]//a//text()')).strip()
        year = " ".join(card.xpath('.//*[contains(@class, "sidearm-roster-player-academic-year")]//text()')).strip()
        number = " ".join(card.xpath('.//*[contains(@class, "sidearm-roster-player-jersey-number")]//text()')).strip()
        position = " ".join(card.xpath('.//*[contains(@class, "sidearm-roster-player-position")]//span[1]//text()')).strip()
        if name:
            rows.append({"player": " ".join(name.split()), "year": year, "number": number, "position": position})
    return pd.DataFrame(rows) if rows else None


def get_roster(team, base_url, season):
    for url in [f"{base_url}/sports/mens-soccer/roster/{season}?view=2",
                f"{base_url}/sports/mens-soccer/roster/{season}",
                f"{base_url}/sports/mens-soccer/roster"]:
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            continue
        roster = from_table(response.text)
        if roster is None:
            roster = from_cards(response.text)
        if roster is not None and len(roster) > 0:
            # some sites put line breaks and tabs inside cells ("Goalkeeper\n\t\t GK"), so tidy every text cell
            for col in roster.columns:
                if roster[col].dtype == object:
                    roster[col] = roster[col].astype(str).str.split().str.join(" ")
            roster["team"] = team
            roster["season"] = season
            return roster, url
    return None, None


if __name__ == "__main__":
    os.makedirs("data/raw/rosters", exist_ok=True)
    for team, url in TEAMS.items():
        roster, used = get_roster(team, url, SEASON)
        if roster is None:
            print(f"{team}: no roster found")
        else:
            roster.to_csv(f"data/raw/rosters/{team}_{SEASON}.csv", index=False)
            print(f"{team}: {len(roster)} players from {used}; classes: {sorted(roster['year'].astype(str).unique())}")
        time.sleep(2)
