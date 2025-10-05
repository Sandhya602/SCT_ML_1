"""load_data.py
Utilities to load and preprocess the housing dataset.
"""


from typing import Tuple
import pandas as pd
import numpy as np


FEATURES = ["sqft", "bedrooms", "bathrooms"]
TARGET = "price"




def load_csv(path: str) -> pd.DataFrame:
"""Load CSV and basic validation."""
df = pd.read_csv(path)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
raise ValueError(f"Missing required columns in CSV: {missing}")
return df




def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
"""Return (X, y) arrays ready for model training.


- Fill or drop missing values
- Convert types
"""
df = df.copy()
# simple cleaning
df = df[FEATURES + [TARGET]]
# drop rows with missing target
df = df.dropna(subset=[TARGET])
# fill feature NAs with median
for f in FEATURES:
if df[f].isna().any():
df[f] = df[f].fillna(df[f].median())
X = df[FEATURES].astype(float).values
y = df[TARGET].astype(float).values
return X, y




if __name__ == "__main__":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="path to CSV file")
args = parser.parse_args()
df = load_csv(args.data)
X, y = preprocess(df)
print(f"Loaded {len(X)} rows")
