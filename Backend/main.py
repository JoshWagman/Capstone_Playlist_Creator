from fastapi import FastAPI
import polars as pl
from sklearn.metrics import DistanceMetric
import numpy as np
# Importing some function from functions.py

model_name = 'multitask_checkpoint.pt'
model_path = '../Model/checkpoints/' + model_name