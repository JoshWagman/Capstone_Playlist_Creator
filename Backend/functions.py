"""
Playlist building via latent-space proximity.
User supplies 13 audio/popularity specs -> fake point in 12-d latent space -> recommend by nearest neighbors.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Project root (parent of Backend)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_PATH = os.path.join(_ROOT, "data", "cleaned_songs_with_groups.csv")
_ENCODER_PATH = os.path.join(_ROOT, "encoder.pkl")

# 13 user-facing features (order must match CSV minus genre_group)
FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "artist_pop", "track_pop",
]
NUMERICAL_FOR_SCALER = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "artist_pop", "track_pop",
]
INPUT_DIM = 24
LATENT_DIM = 12


class Encoder(nn.Module):
    """Encoder: input_dim -> 19 (ReLU) -> 15 (ReLU) -> z_mean (12), z_log_var (12)."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 19)
        self.fc2 = nn.Linear(19, 15)
        self.z_mean = nn.Linear(15, latent_dim)
        self.z_log_var = nn.Linear(15, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


# Lazy-loaded state
_encoder = None
_scaler = None
_ohe = None
_latent_space = None
_songs_df = None


def _get_data():
    global _songs_df
    if _songs_df is None:
        _songs_df = pd.read_csv(_DATA_PATH)
    return _songs_df


def _get_scaler_ohe():
    global _scaler, _ohe
    if _scaler is None:
        df = _get_data()
        X = df.drop(columns=["genre_group"], errors="ignore")
        _scaler = StandardScaler()
        _ohe = OneHotEncoder(sparse_output=False)
        num = X[NUMERICAL_FOR_SCALER]
        _scaler.fit(num)
        _ohe.fit(X[["key"]].values)
    return _scaler, _ohe


def _preprocess_row(row_13: np.ndarray) -> np.ndarray:
    """Turn 13-d user specs into 24-d encoder input (same pipeline as 02_encoder_training)."""
    scaler, ohe = _get_scaler_ohe()
    # row_13 order: danceability, energy, key, loudness, mode, speechiness, ...
    row = row_13.reshape(1, -1)
    num = row[:, [FEATURE_NAMES.index(c) for c in NUMERICAL_FOR_SCALER]]
    key = row[:, FEATURE_NAMES.index("key")].reshape(-1, 1)
    mode = row[:, FEATURE_NAMES.index("mode")].reshape(-1, 1)
    num_scaled = scaler.transform(num)
    key_ohe = ohe.transform(key)
    cat = np.concatenate([key_ohe, mode], axis=1)
    return np.concatenate([num_scaled, cat], axis=1).astype(np.float32)


def _preprocess_df(df: pd.DataFrame) -> np.ndarray:
    """Preprocess full dataframe (13 feature cols) to 24-d."""
    scaler, ohe = _get_scaler_ohe()
    num = df[NUMERICAL_FOR_SCALER].values
    key = df[["key"]].values
    mode = df[["mode"]].values
    num_scaled = scaler.transform(num)
    key_ohe = ohe.transform(key)
    cat = np.concatenate([key_ohe, mode], axis=1)
    return np.concatenate([num_scaled, cat], axis=1).astype(np.float32)


def _get_encoder():
    global _encoder
    if _encoder is None:
        # Pickle was saved from a notebook (__main__.Encoder); make our class available
        import __main__
        __main__.Encoder = Encoder
        with open(_ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)
        _encoder.eval()
    return _encoder


def _get_latent_space():
    global _latent_space, _songs_df
    if _latent_space is None:
        df = _get_data()
        X = df.drop(columns=["genre_group"], errors="ignore")
        X_24 = _preprocess_df(X)
        enc = _get_encoder()
        with torch.no_grad():
            t = torch.from_numpy(X_24)
            z_mean, _ = enc(t)
            _latent_space = z_mean.numpy()
        _songs_df = df
    return _latent_space, _songs_df


def user_specs_to_latent(specs: dict) -> np.ndarray:
    """
    Map user playlist specs (13 values) to a 12-d latent point.
    specs: dict with keys in FEATURE_NAMES (or list/array of 13 values in that order).
    """
    if isinstance(specs, dict):
        row = np.array([float(specs[k]) for k in FEATURE_NAMES], dtype=np.float32)
    else:
        row = np.asarray(specs, dtype=np.float32).ravel()
    if row.size != 13:
        raise ValueError("Expected 13 features")
    x24 = _preprocess_row(row)
    enc = _get_encoder()
    with torch.no_grad():
        t = torch.from_numpy(x24)
        z_mean, _ = enc(t)
        return z_mean.numpy().ravel()


def build_playlist(user_preferences, k: int = 50, metric: str = "euclidean"):
    """
    Build a playlist by creating a fake point in latent space from user specs
    and returning the k nearest songs in that space.

    user_preferences: dict with keys danceability, energy, key, loudness, mode,
        speechiness, acousticness, instrumentalness, liveness, valence, tempo,
        artist_pop, track_pop (or list/array of 13 values in that order).

    Returns: list of dicts, each a song row (incl. genre_group) for the top k neighbors.
    """
    latent_user = user_specs_to_latent(user_preferences)
    space, df = _get_latent_space()

    if metric == "manhattan":
        dists = np.sum(np.abs(space - latent_user), axis=1)
    else:
        dists = np.sqrt(np.sum((space - latent_user) ** 2, axis=1))

    idx = np.argsort(dists)[:k]
    out = df.iloc[idx].to_dict(orient="records")
    return out
