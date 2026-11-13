# Download each team's logo for the app (saved to assets/logos/<team>.png)
# Run with: python get_logos.py
import os

import requests

from scraper import HEADERS

# the logos New College's own schedule page uses for each team
SITE = "https://gomightybanyans.com/images/logos/"
LOGOS = {
    "ncf": "site/site.png",
    "keiser": "KU_logo_200px.png",
    "stthomas": "STU Gameday.png",
    "scad": "SCAD.png",
    "southeastern": "CxoknFLbFqtOmL6U34slG1v2tJRtkeU2qg-removebg-preview.png",
    "fmu": "Florida_Memorial_2021.png",
    "webber": "Warriormascotfinal-011.png",
    "avemaria": "avemaria_logo_200px.png",
    "warner": "Warner_Athletics-removebg-preview.png",
}

if __name__ == "__main__":
    os.makedirs("assets/logos", exist_ok=True)
    for team, path in LOGOS.items():
        response = requests.get(SITE + path, headers=HEADERS, params={"width": 200}, timeout=20)
        if response.status_code == 200:
            with open(f"assets/logos/{team}.png", "wb") as f:
                f.write(response.content)
            print(f"{team}: saved ({len(response.content) // 1024} KB)")
        else:
            print(f"{team}: could not download (status {response.status_code})")
