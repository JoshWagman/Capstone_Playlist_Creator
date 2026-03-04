import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/williamgorman/Documents/GitHub/Capstone_Playlist_Creator/Untitled")

csv_path = PROJECT_ROOT / "data/processed/mpd_unique_tracks_500k.csv"
# OR for the 500k version:
# csv_path = PROJECT_ROOT / "dataset/mpd_unique_tracks_500k.csv"

df = pd.read_csv(csv_path)

print("File:", csv_path)
print("Number of songs:", len(df))
print(df.head())
