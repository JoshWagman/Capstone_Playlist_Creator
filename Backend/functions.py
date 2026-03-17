"""
Playlist building via latent-space proximity.

User supplies 12 audio/popularity specs -> encode to 12-d latent point
-> find nearest neighbors in pre-computed latent space -> return songs.

Everything (encoder weights, scaler, OHE, latent space) is loaded from
the single checkpoint saved by 02b_aae_encoder_training.ipynb.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("music_matcher.backend")

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


def build_playlist(user_preferences, k: int = 50, metric: str = "euclidean"):
    """
    Build a playlist: encode user preferences into the latent space,
    find the k nearest songs, return their metadata.

    Returns: list of dicts (one per song, includes all CSV columns).
    """
    _load_checkpoint()
    latent_user = user_specs_to_latent(user_preferences)
    df = _get_songs_df()

    if metric == "manhattan":
        dists = np.sum(np.abs(_latent_space - latent_user), axis=1)
    else:
        dists = np.sqrt(np.sum((_latent_space - latent_user) ** 2, axis=1))

    idx = np.argsort(dists)[:k]
    log.info(
        "KNN (%s, k=%d): nearest dist=%.4f, farthest dist=%.4f",
        metric, k, dists[idx[0]], dists[idx[-1]],
    )
    return df.iloc[idx].to_dict(orient="records")


def build_playlist_from_seeds(seed_specs_list, k: int = 50, metric: str = "euclidean"):
    """
    Build a playlist by sampling nearest neighbors around each seed song
    individually, distributing k evenly across seeds.

    Duplicates are skipped so the same song never appears twice.

    seed_specs_list: list of dicts (one per seed), each with keys in FEATURE_NAMES.
    k: total number of songs in the final playlist.

    Returns: list of dicts (one per song, includes all CSV columns).
    """
    _load_checkpoint()
    df = _get_songs_df()
    n_seeds = len(seed_specs_list)

    base = k // n_seeds
    remainder = k % n_seeds
    allocations = [base + (1 if i < remainder else 0) for i in range(n_seeds)]
    log.info(
        "Multi-seed playlist: %d seeds, allocations=%s (total=%d)",
        n_seeds, allocations, sum(allocations),
    )

    latent_seeds = [user_specs_to_latent(specs) for specs in seed_specs_list]

    used_indices = set()
    all_results = []

    for i, (latent_seed, alloc) in enumerate(zip(latent_seeds, allocations)):
        if metric == "manhattan":
            dists = np.sum(np.abs(_latent_space - latent_seed), axis=1)
        else:
            dists = np.sqrt(np.sum((_latent_space - latent_seed) ** 2, axis=1))

        sorted_idx = np.argsort(dists)

        count = 0
        for idx in sorted_idx:
            if count >= alloc:
                break
            if idx not in used_indices:
                used_indices.add(idx)
                all_results.append(df.iloc[idx].to_dict())
                count += 1

        nearest = dists[sorted_idx[0]]
        farthest = dists[sorted_idx[min(count - 1, len(sorted_idx) - 1)]] if count else 0.0
        log.info(
            "  Seed %d/%d: wanted %d, added %d (nearest=%.4f, farthest=%.4f)",
            i + 1, n_seeds, alloc, count, nearest, farthest,
        )

    log.info("Multi-seed playlist complete: %d total tracks", len(all_results))
    return all_results
