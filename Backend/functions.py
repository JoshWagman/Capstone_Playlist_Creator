import polars as pl
from sklearn.metrics import DistanceMetric
import numpy as np
import gradio as gr

df = pl.read_csv('../data/cleaned_songs_with_groups.csv')

def build_playlist(user_preferences):
    """
    This function builds a playlist based on the user's preferences.
    It takes in a dictionary of user preferences and returns a playlist.
    The user preferences are a dictionary of the user's preferences for the playlist.
    The user preferences are a dictionary of the user's preferences for the playlist.
    """


#def recommend_songs(user_preferences):
