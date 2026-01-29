# Use this to run the FastAPI "fastapi dev main.py"
from fastapi import FastAPI
import polars as pl
from sklearn.metrics import DistanceMetric
import numpy as np
import torch
# Importing some function from functions.py

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

# API Operations
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/recommend_songs")
def recommend_songs():
    return "Recommended songs"

@app.get("/build_playlist")
def build_playlist():
    return "Built playlist"