import os
import time
import joblib
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer

#-----Config---------
CSV_PATH = r"../examples/Linear_regression_1.csv"
TARGET_COL = "median_house_value"
Test_Size = 0.2
RANDOM_STATE = 42
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

#-----Load Data--------
if CSV_PATH:
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values
else:
    X, y = make_regression(
        n_samples=1000, n_features=5, noise=15.0,
        random_state=RANDOM_STATE
    )

#--------Split---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=Test_Size, random_state=RANDOM_STATE
)

#----Pipeline--------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])

#--------Train (with time)--------
t0 = time.perf_counter()
pipe.fit(X_train, y_train)
train_time = (time.perf_counter() - t0) * 1000  # ms

#-----------Eval (with time)--------
t1 = time.perf_counter()
y_pred = pipe.predict(X_test)
predict_time = (time.perf_counter() - t1) * 1000  # ms

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LinearRegression results:")
print(f"MSE        : {mse:.4f}")
print(f"R2_score   : {r2:.4f}")
print(f"Train Time : {train_time:.3f} ms")
print(f"Predict Time: {predict_time:.3f} ms")

#---------Save Model----------
joblib.dump(pipe, os.path.join(OUT_DIR, "linear_model.joblib"))
print(f"Saved Model -> {os.path.join(OUT_DIR, 'linear_model.joblib')}")

