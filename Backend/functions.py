"""
Playlist building via latent-space proximity.

User supplies 13 audio/popularity specs -> encode to 12-d latent point
-> find nearest neighbors in pre-computed latent space -> return songs.

Everything (encoder weights, scaler, OHE, latent space) is loaded from
the single checkpoint saved by 02b_aae_encoder_training.ipynb.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CHECKPOINT_PATH = os.path.join(_ROOT, "Model", "checkpoints", "aae_encoder.pt")
_DATA_PATH = os.path.join(_ROOT, "data", "cleaned_songs_with_groups.csv")

FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "artist_pop", "track_pop",
]
NUMERICAL_FOR_SCALER = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "artist_pop", "track_pop",
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

    ckpt = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    _encoder = Encoder(ckpt["input_dim"], ckpt["latent_dim"])
    _encoder.load_state_dict(ckpt["encoder"])
    _encoder.eval()

    _scaler = ckpt["scaler"]
    _ohe = ckpt["ohe"]
    _latent_space = ckpt["latent_space"]
    _genre_groups = ckpt["genre_groups"]


def _get_songs_df():
    """Load song metadata (for returning results). Cached after first call."""
    global _songs_df
    if _songs_df is None:
        _songs_df = pd.read_csv(_DATA_PATH)
    return _songs_df


def _preprocess_row(row_13: np.ndarray) -> np.ndarray:
    """Turn 13-d user specs into 24-d encoder input using the training-time scaler/OHE."""
    _load_checkpoint()
    row = row_13.reshape(1, -1)
    num = row[:, [FEATURE_NAMES.index(c) for c in NUMERICAL_FOR_SCALER]]
    key = row[:, FEATURE_NAMES.index("key")].reshape(-1, 1)
    mode = row[:, FEATURE_NAMES.index("mode")].reshape(-1, 1)
    num_scaled = _scaler.transform(num)
    key_ohe = _ohe.transform(key).toarray()
    cat = np.concatenate([key_ohe, mode], axis=1)
    return np.concatenate([num_scaled, cat], axis=1).astype(np.float32)


def user_specs_to_latent(specs) -> np.ndarray:
    """
    Map user playlist specs (13 values) to a 12-d latent point.
    specs: dict with keys in FEATURE_NAMES, or list/array of 13 values in that order.
    """
    if isinstance(specs, dict):
        row = np.array([float(specs[k]) for k in FEATURE_NAMES], dtype=np.float32)
    else:
        row = np.asarray(specs, dtype=np.float32).ravel()
    if row.size != 13:
        raise ValueError("Expected 13 features")

    x24 = _preprocess_row(row)
    _load_checkpoint()
    with torch.no_grad():
        z_mean, _ = _encoder(torch.from_numpy(x24))
        return z_mean.numpy().ravel()


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
    return df.iloc[idx].to_dict(orient="records")
