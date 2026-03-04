# Use this to run the FastAPI "fastapi dev main.py"
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import polars as pl
from sklearn.metrics import DistanceMetric
import numpy as np
import torch

from functions import build_playlist as build_playlist_fn

model_name = 'multitask_checkpoint.pt'
model_path = '../Model/checkpoints/' + model_name

# load the model
model = torch.load(model_path)

# load the music data
music_data = pl.read_csv('../data/cleaned_songs_with_groups.csv')

# create distance metric object
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)

# create FastAPI app
app = FastAPI()


class PlaylistSpecs(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    artist_pop: float
    track_pop: float
    k: Optional[int] = 50


# API Operations
@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/recommend_songs")
def recommend_songs():
    return "Recommended songs"


@app.post("/build_playlist")
def build_playlist(specs: PlaylistSpecs):
    """Build playlist from 13 audio/popularity specs via latent-space nearest neighbors."""
    k = specs.k or 50
    prefs = {
        "danceability": specs.danceability,
        "energy": specs.energy,
        "key": specs.key,
        "loudness": specs.loudness,
        "mode": specs.mode,
        "speechiness": specs.speechiness,
        "acousticness": specs.acousticness,
        "instrumentalness": specs.instrumentalness,
        "liveness": specs.liveness,
        "valence": specs.valence,
        "tempo": specs.tempo,
        "artist_pop": specs.artist_pop,
        "track_pop": specs.track_pop,
    }
    playlist = build_playlist_fn(prefs, k=k)
    return {"playlist": playlist, "count": len(playlist)}