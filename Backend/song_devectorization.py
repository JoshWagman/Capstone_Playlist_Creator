"""
Convert vectors (from functions.build_playlist or similar) back to original songs.

For each vector, searches data/cleaned_songs.csv for the best-matching row
by vector similarity, then returns the original song metadata (id, artist, name, genre).

cleaned_songs.csv format:
    artist_name, id, track_name, danceability, energy, key, loudness, mode,
    speechiness, acousticness, instrumentalness, liveness, valence, tempo,
    artist_pop, genres, track_pop, ...
"""
import os
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SONGS_DB_PATH = os.path.join(_ROOT, "data", "cleaned_songs.csv")

# Vector columns (matches cleaned_songs.csv and functions.FEATURE_NAMES order)
VECTOR_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "artist_pop", "track_pop",
]


def _load_songs_db() -> pd.DataFrame:
    """Load and return the songs database. Cached per process."""
    if not os.path.exists(_SONGS_DB_PATH):
        raise FileNotFoundError(f"Songs database not found: {_SONGS_DB_PATH}")
    return pd.read_csv(_SONGS_DB_PATH)


def _vector_from_row(row: dict | pd.Series) -> np.ndarray:
    """Extract a numeric vector from a row (dict or Series) using VECTOR_COLS."""
    out = []
    for col in VECTOR_COLS:
        val = row[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"Missing vector component: {col}")
        out.append(float(val))
    return np.array(out, dtype=np.float64)


def _row_to_vector(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Get vector for a row index in the database (cleaned_songs columns)."""
    row = df.iloc[idx]
    return np.array([float(row[c]) if pd.notna(row[c]) else 0.0 for c in VECTOR_COLS], dtype=np.float64)


def vectors_to_songs(vectors: list, db_path: str | None = None) -> list[dict]:
    """
    Map a list of vectors to original songs from cleaned_songs.csv.

    Args:
        vectors: List of vectors. Each element is a dict (with feature keys) or
                 a list/array of 13 values in order:
                 [danceability, energy, key, loudness, mode, speechiness,
                  acousticness, instrumentalness, liveness, valence, tempo,
                  artist_pop, track_pop]
        db_path: Optional path to cleaned_songs.csv. Defaults to data/cleaned_songs.csv.

    Returns:
        List of dicts with at least: id, artist, name, genre, plus any other
        columns from the database.
    """
    path = db_path or _SONGS_DB_PATH
    df = pd.read_csv(path)

    # Build vector matrix for the database
    db_vectors = np.stack([_row_to_vector(df, i) for i in range(len(df))])

    result = []
    for vec_spec in vectors:
        if isinstance(vec_spec, (list, tuple, np.ndarray)):
            vec = np.asarray(vec_spec, dtype=np.float64).ravel()
            if vec.size != 13:
                raise ValueError(
                    f"Expected vector of length 13, got {vec.size}. "
                    "Order: danceability, energy, key, loudness, mode, speechiness, "
                    "acousticness, instrumentalness, liveness, valence, tempo, "
                    "artist_pop, track_pop"
                )
            vec_ordered = vec  # Same order as cleaned_songs
        else:
            vec_ordered = _vector_from_row(vec_spec)

        dists = np.sqrt(np.sum((db_vectors - vec_ordered.reshape(1, -1)) ** 2, axis=1))
        best_idx = int(np.argmin(dists))
        best_row = df.iloc[best_idx]

        song = {
            "id": str(best_row.get("id", "")),
            "artist": str(best_row.get("artist_name", "")),
            "name": str(best_row.get("track_name", "")),
            "genre": str(best_row.get("genres", "")),
        }
        for k in df.columns:
            if k not in song:
                v = best_row[k]
                song[k] = v if pd.notna(v) else None
        result.append(song)

    return result
