"""
Playlist building via latent-space proximity.

User supplies 12 audio/popularity specs -> encode to 12-d latent point
-> find nearest neighbors in pre-computed latent space -> return songs.

Everything (encoder weights, scaler, OHE, latent space) is loaded from
the single checkpoint saved by 02b_aae_encoder_training.ipynb.
"""
from __future__ import annotations

import os
import re
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from langdetect import detect, LangDetectException

log = logging.getLogger("music_matcher.backend")

_REMASTER_RE = re.compile(
    r"\s*[-–—(]\s*("
    r"remaster(ed)?|re-?master(ed)?|"
    r"deluxe(\s+edition)?|"
    r"anniversary(\s+edition)?|"
    r"bonus\s+track|"
    r"expanded(\s+edition)?|"
    r"special\s+edition|"
    r"\d{4}\s+(remaster(ed)?|mix|version)|"
    r"single\s+(remix|mix|version|edit)|"
    r"remix|"
    r"live|"
    r"club\s+mix|"
    r"radio\s+edit"
    r").*$",
    re.IGNORECASE,
)

def _clean_artist(artist: str) -> str:
    """Strip list brackets, quotes, and normalise artist for dedup keys."""
    a = str(artist).strip()
    a = re.sub(r"[\[\]\"']+", "", a)
    a = a.split(",")[0].split(";")[0].split("&")[0].strip()
    return a.lower()


def _normalize_title(artist: str, title: str) -> str:
    """Return a lowercased, remaster-stripped key for duplicate detection."""
    t = _REMASTER_RE.sub("", str(title)).strip().rstrip(")")
    return f"{_clean_artist(artist)}|||{t.lower()}"


def _detect_lang(text: str) -> str | None:
    """Best-effort language detection. Returns ISO 639-1 code or None."""
    if not text or len(text.strip()) < 3:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CHECKPOINT_PATH = os.path.join(_ROOT, "Model", "checkpoints", "aae_encoder_500k.pt")
_DATA_PATH = os.path.join(_ROOT, "data", "processed", "top500k_catalog.csv")

FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "popularity",
]
NUMERICAL_FOR_SCALER = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "popularity",
]


class Encoder(nn.Module):
    """Encoder: input_dim -> 19 (ReLU) -> 15 (ReLU) -> z_mean, z_log_var."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 19)
        self.fc2 = nn.Linear(19, 15)
        self.z_mean = nn.Linear(15, latent_dim)
        self.z_log_var = nn.Linear(15, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.z_mean(x), self.z_log_var(x)


# Lazy-loaded state — populated once by _load_checkpoint()
_encoder = None
_scaler = None
_ohe = None
_latent_space = None
_genre_groups = None
_genre_centroids = None
_songs_df = None


def _load_checkpoint():
    """Load encoder, preprocessing objects, and pre-computed latent space from the AAE checkpoint."""
    global _encoder, _scaler, _ohe, _latent_space, _genre_groups

    if _encoder is not None:
        return

    log.info("Loading checkpoint from %s …", _CHECKPOINT_PATH)
    ckpt = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    _encoder = Encoder(ckpt["input_dim"], ckpt["latent_dim"])
    _encoder.load_state_dict(ckpt["encoder"])
    _encoder.eval()

    _scaler = ckpt["scaler"]
    _ohe = ckpt["ohe"]
    _latent_space = ckpt["latent_space"]
    _genre_groups = ckpt["genre_groups"]
    log.info("Checkpoint loaded — encoder ready, latent space shape: %s", _latent_space.shape)


def _get_songs_df():
    """Load and clean song metadata so row indices match the latent space.

    Applies the same parsing/filtering pipeline used during training
    (parse key/mode/genre, drop NaN, drop rare genres, select features, dropna, reset index).
    """
    global _songs_df
    if _songs_df is not None:
        return _songs_df

    import ast as _ast

    raw = pd.read_csv(_DATA_PATH)

    _NOTE_TO_INT = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
    }
    _MODE_MAP = {'Major': 1, 'Minor': 0}

    def _pk(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in _NOTE_TO_INT:
            return _NOTE_TO_INT[s]
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return np.nan

    def _pm(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in _MODE_MAP:
            return _MODE_MAP[s]
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return np.nan

    def _pg(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s.startswith('['):
            try:
                lst = _ast.literal_eval(s)
                return str(lst[0]) if lst else np.nan
            except Exception:
                return np.nan
        return s

    raw['key'] = raw['key'].apply(_pk)
    raw['mode'] = raw['mode'].apply(_pm)
    raw['genre'] = raw['genre'].apply(_pg)
    raw = raw.dropna(subset=['genre', 'key', 'mode'])
    raw['key'] = raw['key'].astype(int)
    raw['mode'] = raw['mode'].astype(int)

    genre_counts = raw['genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 50].index
    raw = raw[raw['genre'].isin(valid_genres)].reset_index(drop=True)

    AUDIO_FEATURES = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity',
    ]
    keep_cols = AUDIO_FEATURES + ['key', 'mode', 'genre']
    songs = raw[keep_cols].copy()
    valid_mask = songs.dropna().index
    _songs_df = raw.loc[valid_mask].reset_index(drop=True)

    log.info("Songs DF loaded: %d rows (should match latent space dim-0)", len(_songs_df))
    return _songs_df


def _get_genre_centroids():
    """Compute and cache the mean latent vector (centroid) for every genre.

    Returns a dict mapping genre string -> np.ndarray of shape (latent_dim,).
    """
    global _genre_centroids
    if _genre_centroids is not None:
        return _genre_centroids

    _load_checkpoint()
    df = _get_songs_df()

    _genre_centroids = {}
    for genre in df["genre"].unique():
        mask = np.where(df["genre"].values == genre)[0]
        _genre_centroids[genre] = _latent_space[mask].mean(axis=0)

    log.info("Genre centroids computed for %d genres", len(_genre_centroids))
    return _genre_centroids


def _top_genres_near(
    latent_point, n=20, neighborhood_k=500, alpha=0.5, anchor_genres=None,
):
    """Return the *n* genres best matching *latent_point* using a hybrid
    score that combines neighbourhood density with centroid proximity.

    hybrid_score(g) = density(g) * exp(-alpha * centroid_dist(g))

    where density = (# songs of g in k-NN ball) / (total # songs of g),
    and centroid_dist = Euclidean distance from g's latent centroid to
    *latent_point*.

    *anchor_genres* (if provided) are always included in the result,
    regardless of their hybrid score, to guarantee the seed songs' own
    genres are represented.
    """
    _load_checkpoint()
    df = _get_songs_df()
    centroids = _get_genre_centroids()

    # --- centroid distances ---
    centroid_dists = {
        genre: np.sqrt(np.sum((c - latent_point) ** 2))
        for genre, c in centroids.items()
    }

    # --- neighbourhood density ---
    dists = np.sqrt(np.sum((_latent_space - latent_point) ** 2, axis=1))
    nearest = np.argsort(dists)[:neighborhood_k]

    ball_counts = pd.Series(df["genre"].values[nearest]).value_counts()
    total_counts = df["genre"].value_counts()
    density = (ball_counts / total_counts).dropna()

    # --- hybrid score ---
    scores = {}
    for genre in centroid_dists:
        d = density.get(genre, 0.0)
        scores[genre] = d * np.exp(-alpha * centroid_dists[genre])

    sorted_genres = sorted(scores, key=scores.get, reverse=True)

    # Guarantee anchor genres appear first, then fill remaining slots
    if anchor_genres:
        valid_anchors = [g for g in anchor_genres if g in centroid_dists]
        remaining = [g for g in sorted_genres if g not in valid_anchors]
        top = valid_anchors + remaining[: max(n - len(valid_anchors), 0)]
    else:
        top = sorted_genres[:n]

    log.info(
        "Hybrid genre ranking (top %d, alpha=%.2f, k=%d, anchors=%s): %s",
        n, alpha, neighborhood_k,
        anchor_genres or [],
        [(g, f"{scores[g]:.6f}") for g in top],
    )
    return top


def _preprocess_row(row_12: np.ndarray) -> np.ndarray:
    """Turn 12-d user specs into encoder input using the training-time scaler/OHE."""
    _load_checkpoint()
    row = row_12.reshape(1, -1)
    num = row[:, [FEATURE_NAMES.index(c) for c in NUMERICAL_FOR_SCALER]]
    key = row[:, FEATURE_NAMES.index("key")].reshape(-1, 1)
    mode = row[:, FEATURE_NAMES.index("mode")].reshape(-1, 1)
    num_scaled = _scaler.transform(num)
    key_ohe = _ohe.transform(key).toarray()
    cat = np.concatenate([key_ohe, mode], axis=1)
    return np.concatenate([num_scaled, cat], axis=1).astype(np.float32)


def user_specs_to_latent(specs) -> np.ndarray:
    """
    Map user playlist specs (12 values) to a latent point.
    specs: dict with keys in FEATURE_NAMES, or list/array of 12 values in that order.
    """
    if isinstance(specs, dict):
        row = np.array([float(specs[k]) for k in FEATURE_NAMES], dtype=np.float32)
    else:
        row = np.asarray(specs, dtype=np.float32).ravel()
    if row.size != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {row.size}")

    x = _preprocess_row(row)
    _load_checkpoint()
    with torch.no_grad():
        z_mean, _ = _encoder(torch.from_numpy(x))
        latent = z_mean.numpy().ravel()
    log.info("User preferences encoded to latent vector: %s", np.round(latent, 4))
    return latent


def build_playlist(
    user_preferences, k: int = 50, metric: str = "euclidean",
    n_genres: int = 10, seed_language=None,
):
    """
    Build a playlist: encode user preferences into the latent space,
    find the k nearest songs, return their metadata.

    Only songs from the *n_genres* genres whose latent-space centroids are
    closest to the encoded point are considered.

    Deduplication removes remastered / deluxe duplicates.
    If *seed_language* is provided, only matching-language songs are kept.

    Returns: list of dicts (one per song, includes all CSV columns).
    """
    _load_checkpoint()
    latent_user = user_specs_to_latent(user_preferences)
    df = _get_songs_df()

    top_genres = _top_genres_near(latent_user, n=n_genres)
    log.info("Top %d genres by neighborhood frequency: %s", n_genres, top_genres)

    genre_mask = df["genre"].isin(top_genres)
    filtered_indices = np.where(genre_mask)[0]
    filtered_latent = _latent_space[filtered_indices]

    if metric == "manhattan":
        dists = np.sum(np.abs(filtered_latent - latent_user), axis=1)
    else:
        dists = np.sqrt(np.sum((filtered_latent - latent_user) ** 2, axis=1))

    sorted_local = np.argsort(dists)

    used_titles = set()
    results = []
    for local_idx in sorted_local:
        if len(results) >= k:
            break
        orig_idx = filtered_indices[local_idx]
        row = df.iloc[orig_idx]
        artist = str(row.get("artist_name", ""))
        title = str(row.get("track_name", ""))

        norm_key = _normalize_title(artist, title)
        if norm_key in used_titles:
            continue

        if seed_language:
            detected = _detect_lang(title)
            if detected and detected != seed_language:
                continue

        used_titles.add(norm_key)
        results.append(row.to_dict())

    log.info(
        "KNN (%s, k=%d, %d genres): returned %d tracks",
        metric, k, n_genres, len(results),
    )
    return results


def build_playlist_from_seeds(
    seed_specs_list, k: int = 50, metric: str = "euclidean", n_genres: int = 5,
    seed_genres=None, seed_language=None,
):
    """
    Build a playlist by sampling nearest neighbors around each seed song
    individually, distributing k evenly across seeds.

    Genre filtering uses a hybrid score (neighbourhood density weighted by
    centroid proximity).  If *seed_genres* are provided, those genres are
    always included in the candidate set so that the seed songs' own
    styles are guaranteed representation.

    Deduplication: songs are skipped if a track with the same normalised
    artist+title already appears (catches remasters, deluxe editions, etc.).

    Language filtering: if *seed_language* is provided (ISO 639-1 code),
    only songs whose title is detected as that language (or undetectable)
    are included.

    seed_specs_list: list of dicts (one per seed), each with keys in FEATURE_NAMES.
    k: total number of songs in the final playlist.
    n_genres: number of closest genres (by hybrid score) to draw from.
    seed_genres: optional list of genre strings from the seed songs.
    seed_language: optional ISO 639-1 language code (e.g. "en", "es", "ko").

    Returns: list of dicts (one per song, includes all CSV columns).
    """
    _load_checkpoint()
    df = _get_songs_df()
    n_seeds = len(seed_specs_list)

    base = k // n_seeds
    remainder = k % n_seeds
    allocations = [base + (1 if i < remainder else 0) for i in range(n_seeds)]
    log.info(
        "Multi-seed playlist: %d seeds, allocations=%s (total=%d), lang=%s",
        n_seeds, allocations, sum(allocations), seed_language,
    )

    latent_seeds = [user_specs_to_latent(specs) for specs in seed_specs_list]

    seed_centroid = np.mean(latent_seeds, axis=0)
    top_genres = _top_genres_near(
        seed_centroid, n=n_genres, anchor_genres=seed_genres,
    )
    log.info("Top %d genres (anchors=%s): %s", n_genres, seed_genres or [], top_genres)

    genre_mask = df["genre"].isin(top_genres)
    filtered_indices = np.where(genre_mask)[0]
    filtered_latent = _latent_space[filtered_indices]
    log.info(
        "Genre filter: %d / %d songs remain after restricting to %d genres",
        len(filtered_indices), len(df), n_genres,
    )

    used_original = set()
    used_titles = set()
    all_results = []
    skipped_lang = 0
    skipped_dup = 0

    for i, (latent_seed, alloc) in enumerate(zip(latent_seeds, allocations)):
        if metric == "manhattan":
            dists = np.sum(np.abs(filtered_latent - latent_seed), axis=1)
        else:
            dists = np.sqrt(np.sum((filtered_latent - latent_seed) ** 2, axis=1))

        sorted_local = np.argsort(dists)

        count = 0
        for local_idx in sorted_local:
            if count >= alloc:
                break
            orig_idx = filtered_indices[local_idx]
            if orig_idx in used_original:
                continue

            row = df.iloc[orig_idx]
            artist = str(row.get("artist_name", ""))
            title = str(row.get("track_name", ""))

            norm_key = _normalize_title(artist, title)
            if norm_key in used_titles:
                skipped_dup += 1
                continue

            if seed_language:
                detected = _detect_lang(title)
                if detected and detected != seed_language:
                    skipped_lang += 1
                    continue

            used_original.add(orig_idx)
            used_titles.add(norm_key)
            all_results.append(row.to_dict())
            count += 1

        nearest = dists[sorted_local[0]]
        farthest = dists[sorted_local[min(count - 1, len(sorted_local) - 1)]] if count else 0.0
        log.info(
            "  Seed %d/%d: wanted %d, added %d (nearest=%.4f, farthest=%.4f)",
            i + 1, n_seeds, alloc, count, nearest, farthest,
        )

    log.info(
        "Multi-seed playlist complete: %d tracks (skipped %d dupes, %d lang mismatch)",
        len(all_results), skipped_dup, skipped_lang,
    )
    return all_results
