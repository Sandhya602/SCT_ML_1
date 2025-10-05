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
