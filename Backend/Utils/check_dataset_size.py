import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/williamgorman/Documents/GitHub/Capstone_Playlist_Creator/Untitled")
CSV_PATH = PROJECT_ROOT / "Frontend/songs_database.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Number of songs:", len(df))
print(df.head())