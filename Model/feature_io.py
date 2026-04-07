from __future__ import annotations

"""
feature_io.py
─────────────
Utilities for saving and loading the Music Matcher feature vector.
Any file in the repository can do:

    from feature_io import load_vector, save_vector, FEATURE_KEYS

The vector is stored as  feature_vector.json  in the repo root.
"""

import json
import os
from datetime import datetime

VECTOR_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend")),
    "feature_vector.json",
)

# Canonical ordered list of all feature keys — use this in other files
FEATURE_KEYS = [
    "genre",
    "tempo",
    "key",
    "mode",
    "danceability",
    "energy",
    "valence",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "min_track_popularity",
    "min_artist_popularity",
]

# Numeric-only keys (excludes genre/mode which are categorical)
NUMERIC_KEYS = [
    "tempo", "key", "danceability", "energy", "valence",
    "loudness", "acousticness", "instrumentalness",
    "liveness", "speechiness", "min_track_popularity", "min_artist_popularity",
]


def save_vector(vector: dict, path: str = VECTOR_PATH) -> None:
    """
    Write the feature vector dict to JSON.
    Automatically adds a 'saved_at' timestamp.
    """
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "vector":   vector,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_vector(path: str = VECTOR_PATH) -> dict | None:
    """
    Load the most recently saved feature vector.
    Returns the vector dict, or None if the file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload.get("vector")


def load_vector_with_meta(path: str = VECTOR_PATH) -> dict | None:
    """
    Load the full payload including 'saved_at' metadata.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def vector_to_numeric_list(vector: dict) -> list[float]:
    """
    Return a flat numeric list of the vector values in NUMERIC_KEYS order.
    Useful as direct input to a ML model or GAN.

    Example output (length 12):
        [120.0, 0.0, 0.5, 0.5, 0.5, -10.0, 0.1, 0.0, 0.1, 0.05, 40.0, 50.0]
    """
    return [float(vector.get(k, 0.0)) for k in NUMERIC_KEYS]
