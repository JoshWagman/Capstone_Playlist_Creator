import pandas as pd
from pathlib import Path



df = pd.read_csv(csv_path)
print("Number of songs:", len(df))
print(df.head())
