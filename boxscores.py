# Scrape the box score of every game (goal times, scorers, fouls, shots by half)
# Run with: python boxscores.py            (all teams, takes ~20 minutes)
#      or:  python boxscores.py ncf 2025   (one team and season, to test)
import os
import re
import sys
import time

import lxml.html
import pandas as pd
import requests

from scraper import HEADERS, TEAMS

SEASONS = [2023, 2024, 2025, 2026]
OUT = "data/raw/box"


def get_page(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    if response.status_code != 200:
        print("Could not load", url, "status:", response.status_code)
        return None
    return lxml.html.fromstring(response.text)


def boxscore_links(base_url, season, retry=True):
    # every box score linked from the schedule page of that season
    page = get_page(f"{base_url}/sports/mens-soccer/schedule/{season}")
    if page is None:
        return []
    links = []
    for href in page.xpath("//a/@href"):
        if f"/stats/{season}/" in href and "/boxscore/" in href:
            if href.startswith("/"):
                href = base_url + href
            if href not in links:
                links.append(href)
    # some sites name older seasons like "2023-24" instead of "2023"
    if not links and retry:
        links = boxscore_links(base_url, f"{season}-{str(season + 1)[-2:]}", retry=False)
    return links


def find_table(page, caption):
    for t in page.xpath("//table"):
        cap = t.xpath("./caption")
        if cap and caption.lower() in cap[0].text_content().lower():
            return t
    return None


def text(el):
    return " ".join(el.text_content().split())


def game_date(page):
    # the game details list: <dt>Date:</dt><dd>10/1/2025</dd>
    for dt in page.xpath("//dt"):
        if text(dt).startswith("Date"):
            dd = dt.getnext()
            return text(dd) if dd is not None else None
    return None


def parse_teams(page):
    # "Team Score By Period": one row per team with its abbreviation and final score
    table = find_table(page, "Score By Period")
    teams = []
    for row in table.xpath("./tbody/tr"):
        th = row.xpath("./th")[0]
        full = th.xpath(".//span[contains(@class, 'hide-on-small-down')]")
        short = th.xpath(".//span[contains(@class, 'hide-on-medium')]")
        if full and short:
            # the abbreviation can have spaces ("NEW COLL"), so read both spans
            name, abbr = text(full[0]), text(short[0])
        else:
            name = text(th).replace("Winner", "").strip()
            name, abbr = " ".join(name.split()[:-1]), name.split()[-1]
        teams.append({"abbr": abbr, "name": name, "goals": text(row.xpath("./td")[-1])})
    # "Team Statistics": label rows (SHOTS, SAVES, CORNER KICKS, FOULS) then one row per team
    stats = find_table(page, "Team Statistics")
    label = None
    if stats is not None:
        for row in stats.xpath("./tbody/tr"):
            cells = [text(c) for c in row.xpath("./th|./td")]
            if len(cells) == 1 or all(c == "" for c in cells[1:]):
                label = cells[0].lower().replace(" ", "_")
                continue
            for t in teams:
                if cells[0] == t["abbr"]:
                    # shots look like "7 (2)" = 7 shots, 2 on goal; keep the first half and the total
                    t[label + "_1st_half"] = cells[1].split()[0]
                    t[label] = cells[-1].split()[0]
    return teams


def parse_goals(page):
    table = find_table(page, "Scoring Summary")
    goals = []
    if table is None:
        return goals
    for row in table.xpath("./tbody/tr"):
        tds = row.xpath("./td")
        if len(tds) < 3:
            continue
        spans = [text(s) for s in tds[2].xpath("./span")]
        if not spans:
            spans = [text(tds[2])]
        scorer = re.sub(r"\s*\(\d+\)$", "", spans[0])
        assist = ""
        for s in spans:
            if s.startswith("Assisted By:"):
                assist = s.replace("Assisted By:", "").strip()
        desc = text(tds[2]).lower()
        goals.append({"time": text(tds[0]), "abbr": text(tds[1]), "scorer": scorer, "assist": assist,
                      "penalty": "penalty" in desc, "header": "header" in desc})
    return goals


def scrape_season(team, base_url, season):
    goals_file = f"{OUT}/{team}_{season}_goals.csv"
    teams_file = f"{OUT}/{team}_{season}_teams.csv"
    if os.path.exists(goals_file) and os.path.exists(teams_file):
        print(f"{team} {season}: already done, skipping")
        return
    links = boxscore_links(base_url, season)
    print(f"{team} {season}: {len(links)} box scores")
    all_goals, all_teams = [], []
    for url in links:
        try:
            page = get_page(url)
            if page is None:
                continue
            game_id = url.rstrip("/").split("/")[-1]
            date = game_date(page)
            for t in parse_teams(page):
                all_teams.append({"team": team, "season": season, "game_id": game_id, "date": date, **t})
            for g in parse_goals(page):
                all_goals.append({"team": team, "season": season, "game_id": game_id, "date": date, **g})
        except Exception as e:
            # one broken page shouldn't stop the whole run
            print("  problem with", url, "->", e)
        time.sleep(1)  # be polite to the sites
    pd.DataFrame(all_goals).to_csv(goals_file, index=False)
    pd.DataFrame(all_teams).to_csv(teams_file, index=False)
    print(f"  saved {len(all_goals)} goals, {len(all_teams) // 2} games")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    teams = [sys.argv[1]] if len(sys.argv) > 1 else list(TEAMS)
    seasons = [int(sys.argv[2])] if len(sys.argv) > 2 else SEASONS
    for team in teams:
        for season in seasons:
            scrape_season(team, TEAMS[team], season)
