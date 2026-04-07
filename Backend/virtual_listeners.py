"""
Virtual Listening Test Framework.

Defines 10 virtual listeners with coherent music preferences,
generates a playlist for each via build_playlist_from_seeds(),
then scores every song for fit and every playlist for cohesion.

Results are saved to results/song_scores.csv and results/playlist_scores.csv.

Usage:
    python Backend/virtual_listeners.py
"""
from __future__ import annotations

import os
import sys
import logging
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from functions import (
    build_playlist_from_seeds,
    _get_songs_df,
    _detect_lang,
    FEATURE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("virtual_listeners")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(_ROOT, "results")

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "popularity",
]

FEATURE_RANGES = {
    "danceability": 1.0, "energy": 1.0, "valence": 1.0,
    "acousticness": 1.0, "instrumentalness": 1.0, "liveness": 1.0,
    "speechiness": 1.0, "loudness": 60.0, "tempo": 150.0,
}

# ────────────────────────────────────────────────────────────────
# 10 VIRTUAL LISTENER PROFILES
# ────────────────────────────────────────────────────────────────

LISTENERS = [
    {
        "id": 1,
        "name": "Jazz/Blues Purist",
        "loved_genres": ["jazz", "blues", "soul"],
        "tolerated_genres": ["r&b", "funk", "bossa"],
        "despised_genres": ["pop", "edm", "k-pop", "reggaeton", "trap"],
        "ideal_features": {
            "danceability": 0.45, "energy": 0.35, "valence": 0.50,
            "acousticness": 0.75, "instrumentalness": 0.30,
            "speechiness": 0.04, "liveness": 0.25,
            "loudness": -12.0, "tempo": 110.0,
        },
        "seed_genre_keys": ["jazz"],
    },
    {
        "id": 2,
        "name": "Rock Veteran",
        "loved_genres": ["rock", "classic_rock", "album_rock", "hard_rock"],
        "tolerated_genres": ["metal", "punk", "indie", "alternative"],
        "despised_genres": ["pop", "k-pop", "latin", "reggaeton", "dance"],
        "ideal_features": {
            "danceability": 0.45, "energy": 0.80, "valence": 0.55,
            "acousticness": 0.15, "instrumentalness": 0.10,
            "speechiness": 0.04, "liveness": 0.20,
            "loudness": -6.0, "tempo": 128.0,
        },
        "seed_genre_keys": ["rock", "classic_rock"],
    },
    {
        "id": 3,
        "name": "Pop Mainstream",
        "loved_genres": ["pop", "dance_pop", "post-teen_pop"],
        "tolerated_genres": ["r&b", "hip-hop", "dance", "electropop"],
        "despised_genres": ["metal", "classical", "folk", "blues", "country"],
        "ideal_features": {
            "danceability": 0.70, "energy": 0.65, "valence": 0.65,
            "acousticness": 0.15, "instrumentalness": 0.0,
            "speechiness": 0.06, "liveness": 0.12,
            "loudness": -5.5, "tempo": 118.0,
        },
        "seed_genre_keys": ["pop", "dance_pop"],
    },
    {
        "id": 4,
        "name": "Hip-Hop Head",
        "loved_genres": ["hip-hop", "rap", "trap", "atl_hip_hop"],
        "tolerated_genres": ["r&b", "pop_rap", "pop"],
        "despised_genres": ["country", "folk", "classical", "metal", "jazz"],
        "ideal_features": {
            "danceability": 0.75, "energy": 0.70, "valence": 0.45,
            "acousticness": 0.10, "instrumentalness": 0.0,
            "speechiness": 0.20, "liveness": 0.15,
            "loudness": -6.0, "tempo": 130.0,
        },
        "seed_genre_keys": ["hip-hop", "rap", "trap"],
    },
    {
        "id": 5,
        "name": "EDM/Dance Fan",
        "loved_genres": ["edm", "electro", "house", "dance", "trance"],
        "tolerated_genres": ["pop", "techno", "electropop"],
        "despised_genres": ["folk", "classical", "country", "blues", "jazz"],
        "ideal_features": {
            "danceability": 0.72, "energy": 0.85, "valence": 0.45,
            "acousticness": 0.05, "instrumentalness": 0.15,
            "speechiness": 0.06, "liveness": 0.15,
            "loudness": -5.0, "tempo": 128.0,
        },
        "seed_genre_keys": ["edm", "electro", "house"],
    },
    {
        "id": 6,
        "name": "Indie Dreamer",
        "loved_genres": ["indie", "bedroom_pop", "indie_pop", "dream_pop"],
        "tolerated_genres": ["folk", "singer-songwriter", "rock", "alternative"],
        "despised_genres": ["edm", "rap", "trap", "reggaeton", "metal"],
        "ideal_features": {
            "danceability": 0.55, "energy": 0.50, "valence": 0.55,
            "acousticness": 0.40, "instrumentalness": 0.05,
            "speechiness": 0.04, "liveness": 0.12,
            "loudness": -8.0, "tempo": 115.0,
        },
        "seed_genre_keys": ["indie", "bedroom_pop"],
    },
    {
        "id": 7,
        "name": "Classical/Ambient",
        "loved_genres": ["classical", "ambient", "piano", "orchestra"],
        "tolerated_genres": ["jazz", "folk", "soundtrack", "new_age"],
        "despised_genres": ["rap", "metal", "edm", "trap", "reggaeton"],
        "ideal_features": {
            "danceability": 0.25, "energy": 0.15, "valence": 0.30,
            "acousticness": 0.90, "instrumentalness": 0.80,
            "speechiness": 0.03, "liveness": 0.10,
            "loudness": -20.0, "tempo": 95.0,
        },
        "seed_genre_keys": ["classical", "piano"],
    },
    {
        "id": 8,
        "name": "Latin Lover",
        "loved_genres": ["latin", "reggaeton", "salsa", "latino", "latin_pop"],
        "tolerated_genres": ["pop", "dance", "r&b"],
        "despised_genres": ["metal", "country", "folk", "classical", "ambient"],
        "ideal_features": {
            "danceability": 0.78, "energy": 0.75, "valence": 0.70,
            "acousticness": 0.12, "instrumentalness": 0.0,
            "speechiness": 0.10, "liveness": 0.18,
            "loudness": -5.0, "tempo": 100.0,
        },
        "seed_genre_keys": ["latin", "reggaeton"],
    },
    {
        "id": 9,
        "name": "Country/Folk Fan",
        "loved_genres": ["country", "folk", "bluegrass", "americana"],
        "tolerated_genres": ["rock", "singer-songwriter", "acoustic"],
        "despised_genres": ["edm", "rap", "k-pop", "trap", "metal"],
        "ideal_features": {
            "danceability": 0.55, "energy": 0.55, "valence": 0.65,
            "acousticness": 0.50, "instrumentalness": 0.02,
            "speechiness": 0.03, "liveness": 0.15,
            "loudness": -7.0, "tempo": 120.0,
        },
        "seed_genre_keys": ["country", "folk"],
    },
    {
        "id": 10,
        "name": "K-Pop/J-Pop Fan",
        "loved_genres": ["k-pop", "j-pop", "korean", "japanese"],
        "tolerated_genres": ["pop", "dance", "electropop"],
        "despised_genres": ["country", "metal", "folk", "blues", "classical"],
        "ideal_features": {
            "danceability": 0.72, "energy": 0.78, "valence": 0.60,
            "acousticness": 0.10, "instrumentalness": 0.0,
            "speechiness": 0.06, "liveness": 0.12,
            "loudness": -5.0, "tempo": 125.0,
        },
        "seed_genre_keys": ["k-pop"],
    },
]


# ────────────────────────────────────────────────────────────────
# SEED SONG SELECTION
# ────────────────────────────────────────────────────────────────

def select_seeds(listener: dict, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Pick the top-*n* most popular songs whose genre matches
    any of the listener's seed_genre_keys."""
    keys = listener["seed_genre_keys"]
    mask = df["genre"].str.lower().apply(
        lambda g: any(k in g for k in keys) if pd.notna(g) else False
    )
    candidates = df[mask].copy()
    if candidates.empty:
        log.warning("No seed candidates for %s, falling back to loved_genres", listener["name"])
        keys = listener["loved_genres"]
        mask = df["genre"].str.lower().apply(
            lambda g: any(k in g for k in keys) if pd.notna(g) else False
        )
        candidates = df[mask].copy()

    if candidates.empty:
        log.error("No candidates at all for %s", listener["name"])
        return pd.DataFrame()

    candidates = candidates.sort_values("popularity", ascending=False, na_position="last")
    return candidates.head(n)


# ────────────────────────────────────────────────────────────────
# PER-SONG FIT SCORING
# ────────────────────────────────────────────────────────────────

def _genre_match(song_genre: str, loved: list, tolerated: list, despised: list) -> float:
    """Return 10 for loved, 5 for tolerated, 1 for despised, 3 for neutral."""
    g = str(song_genre).lower()
    if any(k in g for k in loved):
        return 10.0
    if any(k in g for k in tolerated):
        return 5.0
    if any(k in g for k in despised):
        return 1.0
    return 3.0


def _feature_score(song: dict, ideal: dict) -> float:
    """Score 0-10 based on proximity to ideal audio features."""
    diffs = []
    for feat, rng in FEATURE_RANGES.items():
        actual = song.get(feat)
        target = ideal.get(feat)
        if actual is None or target is None or pd.isna(actual):
            continue
        diff = abs(float(actual) - float(target)) / rng
        diffs.append(min(diff, 1.0))
    if not diffs:
        return 5.0
    return (1.0 - np.mean(diffs)) * 10.0


def score_song(listener: dict, song: dict) -> dict:
    """Return genre_score, feature_score, fit_score for one song."""
    gs = _genre_match(
        song.get("genre", ""),
        listener["loved_genres"],
        listener["tolerated_genres"],
        listener["despised_genres"],
    )
    fs = _feature_score(song, listener["ideal_features"])
    fit = round(max(1.0, min(10.0, 0.5 * gs + 0.5 * fs)), 1)
    return {"genre_score": round(gs, 1), "feature_score": round(fs, 1), "fit_score": fit}


# ────────────────────────────────────────────────────────────────
# PLAYLIST COHESION SCORING
# ────────────────────────────────────────────────────────────────

def score_playlist_cohesion(listener: dict, songs: list[dict]) -> dict:
    """Evaluate how well the playlist holds together for this listener."""
    if not songs:
        return {
            "genre_consistency": 1.0, "feature_coherence": 1.0,
            "cohesion_score": 1.0, "pct_loved": 0, "pct_tolerated": 0,
            "pct_despised": 0, "num_genres": 0, "top_genres": "",
        }

    loved = listener["loved_genres"]
    tolerated = listener["tolerated_genres"]
    despised = listener["despised_genres"]

    n = len(songs)
    n_loved = sum(1 for s in songs if any(k in str(s.get("genre", "")).lower() for k in loved))
    n_tolerated = sum(1 for s in songs if any(k in str(s.get("genre", "")).lower() for k in tolerated))
    n_despised = sum(1 for s in songs if any(k in str(s.get("genre", "")).lower() for k in despised))

    pct_loved = round(100 * n_loved / n, 1)
    pct_tolerated = round(100 * n_tolerated / n, 1)
    pct_despised = round(100 * n_despised / n, 1)

    fraction_acceptable = (n_loved + n_tolerated) / n
    genre_consistency = round(fraction_acceptable * 10.0, 1)

    coherence_feats = ["energy", "danceability", "valence", "tempo"]
    stds = []
    for feat in coherence_feats:
        vals = [float(s[feat]) for s in songs if feat in s and pd.notna(s.get(feat))]
        if len(vals) >= 2:
            stds.append(np.std(vals) / FEATURE_RANGES.get(feat, 1.0))
    feature_coherence = round(10.0 * (1.0 - np.mean(stds)), 1) if stds else 5.0

    cohesion = round(max(1.0, min(10.0, 0.5 * genre_consistency + 0.5 * feature_coherence)), 1)

    genre_counts = Counter(str(s.get("genre", "unknown")) for s in songs)
    top_3 = [g for g, _ in genre_counts.most_common(3)]

    return {
        "genre_consistency": max(1.0, min(10.0, genre_consistency)),
        "feature_coherence": max(1.0, min(10.0, feature_coherence)),
        "cohesion_score": cohesion,
        "pct_loved": pct_loved,
        "pct_tolerated": pct_tolerated,
        "pct_despised": pct_despised,
        "num_genres": len(genre_counts),
        "top_genres": ", ".join(top_3),
    }


# ────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ────────────────────────────────────────────────────────────────

def run_all_tests(playlist_size: int = 20):
    """Generate playlists for all 10 listeners, score, and save CSVs."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    df = _get_songs_df()
    log.info("Catalog loaded: %d songs", len(df))

    song_rows = []
    playlist_rows = []

    for listener in LISTENERS:
        lid = listener["id"]
        name = listener["name"]
        log.info("=" * 60)
        log.info("Listener %d: %s", lid, name)
        log.info("=" * 60)

        seed_df = select_seeds(listener, df, n=3)
        if seed_df.empty:
            log.error("  Skipping — no seed songs found")
            continue

        seed_names = seed_df["track_name"].tolist()
        log.info("  Seeds: %s", seed_names)

        seed_specs_list = []
        for _, row in seed_df.iterrows():
            specs = row[FEATURE_COLS].to_dict()
            seed_specs_list.append(specs)

        seed_genres = seed_df["genre"].dropna().unique().tolist()

        lang_votes = [
            _detect_lang(str(t))
            for t in seed_df["track_name"].dropna()
            if _detect_lang(str(t)) is not None
        ]
        seed_language = Counter(lang_votes).most_common(1)[0][0] if lang_votes else None
        log.info("  Detected language: %s", seed_language)

        songs = build_playlist_from_seeds(
            seed_specs_list,
            k=playlist_size,
            seed_genres=seed_genres,
            seed_language=seed_language,
        )
        log.info("  Playlist: %d songs generated", len(songs))

        for idx, song in enumerate(songs, 1):
            scores = score_song(listener, song)
            song_rows.append({
                "listener_id": lid,
                "listener_name": name,
                "song_idx": idx,
                "track_name": song.get("track_name", ""),
                "artist_name": song.get("artist_name", ""),
                "genre": song.get("genre", ""),
                "genre_score": scores["genre_score"],
                "feature_score": scores["feature_score"],
                "fit_score": scores["fit_score"],
            })

        cohesion = score_playlist_cohesion(listener, songs)
        playlist_rows.append({
            "listener_id": lid,
            "listener_name": name,
            "num_songs": len(songs),
            "num_genres": cohesion["num_genres"],
            "top_genres": cohesion["top_genres"],
            "pct_loved": cohesion["pct_loved"],
            "pct_tolerated": cohesion["pct_tolerated"],
            "pct_despised": cohesion["pct_despised"],
            "genre_consistency": cohesion["genre_consistency"],
            "feature_coherence": cohesion["feature_coherence"],
            "cohesion_score": cohesion["cohesion_score"],
            "seed_songs": " | ".join(str(s) for s in seed_names),
        })

        log.info("  Cohesion: %.1f  (genre_cons=%.1f, feat_coh=%.1f)",
                 cohesion["cohesion_score"],
                 cohesion["genre_consistency"],
                 cohesion["feature_coherence"])
        log.info("  Loved=%.0f%%  Tolerated=%.0f%%  Despised=%.0f%%",
                 cohesion["pct_loved"], cohesion["pct_tolerated"], cohesion["pct_despised"])

    song_csv = os.path.join(_RESULTS_DIR, "song_scores.csv")
    playlist_csv = os.path.join(_RESULTS_DIR, "playlist_scores.csv")

    pd.DataFrame(song_rows).to_csv(song_csv, index=False)
    pd.DataFrame(playlist_rows).to_csv(playlist_csv, index=False)

    log.info("=" * 60)
    log.info("Results saved:")
    log.info("  Song scores:     %s  (%d rows)", song_csv, len(song_rows))
    log.info("  Playlist scores: %s  (%d rows)", playlist_csv, len(playlist_rows))
    log.info("=" * 60)

    return song_rows, playlist_rows


if __name__ == "__main__":
    run_all_tests()
