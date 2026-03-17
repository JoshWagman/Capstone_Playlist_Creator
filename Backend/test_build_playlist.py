"""Quick smoke test for build_playlist."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from functions import build_playlist, FEATURE_NAMES

test_preferences = {
    "danceability": 0.72,
    "energy": 0.65,
    "key": 5,
    "loudness": -6.0,
    "mode": 1,
    "speechiness": 0.04,
    "acousticness": 0.12,
    "instrumentalness": 0.0,
    "liveness": 0.15,
    "valence": 0.55,
    "tempo": 120.0,
    "artist_pop": 70,
    "track_pop": 60,
}

print("Input preferences:")
for k, v in test_preferences.items():
    print(f"  {k}: {v}")

print("\nBuilding playlist (k=10)...\n")
results = build_playlist(test_preferences, k=10)

for i, song in enumerate(results, 1):
    print(f"--- Song {i} ---")
    for key in song:
        print(f"  {key}: {song[key]}")
    print()

print(f"Returned {len(results)} songs.")
