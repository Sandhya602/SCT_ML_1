# SCT_ML_1
# House Price Linear Regression 
house-price-linear-regression/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── housing.csv            # example dataset (or your real dataset)
├── src/
│   ├── __init__.py
│   ├── load_data.py           # load & preprocess dataset
│   ├── train_model.py         # train & save model
│   └── predict.py             # load model and predict on new samples
├── notebooks/
│   └── exploratory.ipynb      # optional
└── sample_requests/           # example input JSON for prediction
    └── sample_input.json
```

---

## `README.md`

```markdown
# House Price Prediction (Linear Regression)

This repository contains code to train a simple linear regression model to predict house prices using three features:
- `sqft` (square footage)
- `bedrooms`
- `bathrooms`

The project uses scikit-learn and a small, reproducible pipeline. The code includes dataset loading, preprocessing, model training/saving, and a prediction script.

## Files

- `requirements.txt` - Python dependencies
- `data/housing.csv` - CSV dataset (see format below)
- `src/load_data.py` - data loading and preprocessing utility
- `src/train_model.py` - trains and saves model to `model.joblib`
- `src/predict.py` - loads saved model and runs predictions
- `sample_requests/sample_input.json` - example input for prediction

## Dataset format

Place a CSV file at `data/housing.csv` with columns (header row required):

```

sqft,bedrooms,bathrooms,price
1200,3,2,250000
1500,3,2,300000
...

````

`price` is the target.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
````

2. Train the model (outputs `model.joblib`):

```bash
python src/train_model.py --data data/housing.csv --output model.joblib
```

3. Run a single prediction:

```bash
python src/predict.py --model model.joblib --sqft 1400 --bedrooms 3 --bathrooms 2
```

Or use the example JSON input:

```bash
python src/predict.py --model model.joblib --input_json sample_requests/sample_input.json
```

## Notes

* The included implementation uses `StandardScaler` + `LinearRegression` inside a `Pipeline`.
* For better performance on real datasets, consider feature engineering, outlier removal, cross-validation, and regularized regression (Ridge/Lasso).

## License

MIT

```
```

---

## `.gitignore`

```text
venv/
__pycache__/
*.pyc
model.joblib
.env
.DS_Store
```

---

## `requirements.txt`

```text
python-dateutil>=2.8.2
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
joblib>=1.2
```

---

## `src/load_data.py`

```python
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
```

---

## `src/train_model.py`

```python
"""train_model.py
Train a linear regression model and save it.
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from load_data import load_csv, preprocess


def build_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    return pipe


def train(data_path: str, output_path: str, test_size=0.2, random_state=42):
    df = load_csv(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # evaluate
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)

    # cross-val RMSE (neg_mean_squared_error -> convert)
    import numpy as np
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_cv = np.sqrt(-cv_scores).mean()

    # save
    joblib.dump(pipe, output_path)

    print(f"Model saved to {output_path}")
    print(f"Train R^2: {train_score:.4f}")
    print(f"Test R^2: {test_score:.4f}")
    print(f"CV RMSE (5-fold): {rmse_cv:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to CSV file")
    p.add_argument("--output", default="model.joblib", help="output model path")
    args = p.parse_args()
    train(args.data, args.output)
```

> **Note:** import paths assume you run `python src/train_model.py` from the repository root. If you encounter `ModuleNotFoundError`, run using `python -m src.train_model` or add `src` to `PYTHONPATH`.

---

## `src/predict.py`

```python
"""predict.py
Load trained model and predict house price from CLI or JSON file.
"""

import argparse
import joblib
import json
import numpy as np
from pathlib import Path

FEATURES = ["sqft", "bedrooms", "bathrooms"]


def predict_from_values(model_path: str, values: list) -> float:
    model = joblib.load(model_path)
    X = np.array(values, dtype=float).reshape(1, -1)
    pred = model.predict(X)
    return float(pred[0])


def predict_from_json(model_path: str, json_path: str) -> list:
    with open(json_path, "r") as f:
        data = json.load(f)
    model = joblib.load(model_path)
    results = []
    # expected: list of {"sqft":..., "bedrooms":..., "bathrooms":...}
    if isinstance(data, dict):
        data = [data]
    for item in data:
        X = [item.get(f) for f in FEATURES]
        pred = model.predict(np.array(X, dtype=float).reshape(1, -1))
        results.append({"input": item, "predicted_price": float(pred[0])})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to trained model (.joblib)")
    parser.add_argument("--sqft", type=float, help="square footage")
    parser.add_argument("--bedrooms", type=float, help="number of bedrooms")
    parser.add_argument("--bathrooms", type=float, help="number of bathrooms")
    parser.add_argument("--input_json", help="json file with list of inputs")
    args = parser.parse_args()

    if args.input_json:
        out = predict_from_json(args.model, args.input_json)
        print(json.dumps(out, indent=2))
    else:
        if args.sqft is None or args.bedrooms is None or args.bathrooms is None:
            parser.error("Provide either --input_json or all three features: --sqft --bedrooms --bathrooms")
        val = [args.sqft, args.bedrooms, args.bathrooms]
        pred = predict_from_values(args.model, val)
        print(f"Predicted price: {pred:.2f}")
```

---

## `sample_requests/sample_input.json`

```json
[
  {"sqft": 1400, "bedrooms": 3, "bathrooms": 2},
  {"sqft": 2000, "bedrooms": 4, "bathrooms": 3}
]
```

---

## Optional: synthetic data generator

If you don't have a dataset yet, use this small script to generate a toy dataset at `data/housing.csv`.

```python
# scripts/generate_synthetic_data.py
import csv
import random
from pathlib import Path

OUT = Path("data/housing.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sqft", "bedrooms", "bathrooms", "price"])
    for _ in range(1000):
        sqft = random.randint(500, 4000)
        bedrooms = random.randint(1, 6)
        bathrooms = random.randint(1, 5)
        # simple synthetic price function + noise
        price = 50 * sqft + 20000 * bedrooms + 15000 * bathrooms + random.gauss(0, 30000)
        price = max(10000, int(price))
        writer.writerow([sqft, bedrooms, bathrooms, price])

print(f"Synthetic data written to {OUT}")
```

Run with:

```bash
python scripts/generate_synthetic_data.py
```

---

## Tips & next steps

* Replace `LinearRegression` with `Ridge` or `Lasso` if you see overfitting.
* Use `sklearn.model_selection.GridSearchCV` to tune hyperparameters.
* Log training metrics to a file or experiment tracker.

---

If you'd like, I can also produce a ready-to-copy zip of these files or commit them directly to a GitHub repo (I cannot access your GitHub without your authorization). Tell me which format you prefer.
