# ncf-soccer-analytics
Scouting the Sun Conference
A Streamlit app for New College of Florida men's soccer. Pick a Sun Conference opponent and the app shows how likely New College is to win, what the opponent is good and bad at, when they score and concede, the players to watch, and a game plan. A local AI model can also write a short scouting report from those numbers.

Live app: 

What's in the app
Scout an opponent: win, tie and loss chances for home, away or neutral games, the most likely score, strengths and weaknesses, goal timing, recent form, who to watch (with whether they return next season), a game plan, and the AI scouting report.
Our season vs the model: every 2026 New College conference game, what the model predicted, and what happened.
Conference rankings, what wins games, players, and about the data: the nine teams side by side, the stats linked to winning, player leaderboards by season, and returning players.
Data sources
Web scraping: season stats, game results, box scores and 2026 rosters from the athletics websites of all nine Sun Conference schools (2023 to 2026). No ready-made API has this data, and web scraping was approved by the professor.
API: the Open-Meteo historical weather API (free, no key) for the weather at the home field on every game day.
The result is 631 games, 2,200+ goals with minute, scorer and assist, and 303 players on the 2026 rosters.
How the win model works
Each team gets an attack and a defense strength from goals scored and allowed, compared with the conference average, with recent games counting more. Those give the expected goals for each side (with a home-field bonus), and a Poisson distribution turns expected goals into the chance of every score. Tested game by game on the 2025 and 2026 conference seasons using only earlier games, it picked the right result 76% of the time, against 49% for always picking the home team.

The LLM part
The app turns the scouting numbers into a short facts sheet, and Qwen2.5-1.5B-Instruct (run locally with Hugging Face transformers, no API key) writes it up as a report with three parts: Overview, Their threats, and How we win. The model only sees the facts sheet and uses a low temperature (0.2) so it sticks to the numbers. The model is about 3 GB, so the report only works on a laptop; the online app hides the button.

Files
File	What it does
app.py	The Streamlit app
scraper.py	Scrapes season stats and results from each school's site
boxscores.py	Scrapes every box score (goal times, scorers, shots, fouls)
roster.py	Scrapes the 2026 rosters with class year
weather.py	Gets game-day weather from the Open-Meteo API
get_logos.py	Downloads each team's logo
clean.py	Cleans and joins everything into data/clean/
eda.py	Exploratory analysis: descriptive stats, missing data, outliers, correlations
scouting.py	Win model, strengths and weaknesses, game plan, season review
llm.py	Builds the facts sheet and runs Qwen for the AI report
How to run it
Install Python 3, then in the project folder:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
Optional, for the AI report: pip install -r requirements-llm.txt
Start the app: streamlit run app.py
The cleaned data is already in the repo, so the app runs right away. To collect fresh data, run scraper.py, boxscores.py, roster.py, weather.py and then clean.py (the box scores take about 20 minutes).

Limits
Each team plays about 20 conference games over two seasons, so the best and worst teams are clear but the middle of the table is hard to separate. The model almost never predicts a tie. Some stats only exist where a box score was posted, and New College has no 2023 conference games.

AI Acknowledgment
I used Claude (Anthropic) as a coding assistant throughout the project: to plan the steps, debug code, improve my own code. Inside the app, the open-source Qwen2.5-1.5B-Instruct model writes the scouting reports.

Author
Sebastián Pedraza, Programming for Data Science, New College of Florida, Fall 2026