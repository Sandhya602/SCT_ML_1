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
