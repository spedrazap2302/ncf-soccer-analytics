import requests
import pandas as pd
from io import StringIO

url = "https://gomightybanyans.com/sports/mens-soccer/stats/2025"

# some sites block requests that don't look like a browser
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
print("Status:", response.status_code)  # 200 means it worked

# read_html finds every <table> on the page and returns a list of DataFrames
tables = pd.read_html(StringIO(response.text))
print("Number of tables:", len(tables))

# look at the first rows of each table
for i, t in enumerate(tables):
    print(f"\n--- Table {i} ---")
    print(t.head(3))