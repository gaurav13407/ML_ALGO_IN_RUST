import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import os


PATH = r"../examples/KMeans_1.csv"   # change if needed
N_CLUSTERS = 2
RANDOM_STATE = 42
N_INIT = 10              # safer across sklearn versions
MAX_ITER = 300
PREDICT_SUBSET = None    # e.g., 1000 or None to use full dataset for predict timing
OUTPUT_BENCH = "kmeans_sklearn_benchmark.csv"


# Load data
data = pd.read_csv(PATH)

# One-hot encode categorical columns (if any)
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = enc.fit_transform(data[cat_cols])
    enc_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(cat_cols), index=data.index)
    data = pd.concat([data.drop(columns=cat_cols), enc_df], axis=1)

X = data.values.astype(float)
n_samples = X.shape[0]

# Build model
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    n_init=N_INIT,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)

# Measure training time (fit)
t0 = time.perf_counter()
kmeans.fit(X)
t1 = time.perf_counter()
train_time_s = t1 - t0

labels = kmeans.labels_
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Measure predict time
if PREDICT_SUBSET is None or PREDICT_SUBSET >= n_samples:
    X_pred = X
else:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n_samples, size=min(PREDICT_SUBSET, n_samples), replace=False)
    X_pred = X[idx]

t0 = time.perf_counter()
pred_labels = kmeans.predict(X_pred)
t1 = time.perf_counter()
predict_time_s = t1 - t0
predict_per_sample_ms = (predict_time_s / X_pred.shape[0]) * 1000.0

# Compute clustering quality metrics (some require >1 cluster and n_samples > n_clusters)
sil_score = None
db_score = None
ch_score = None

try:
    if len(set(labels)) > 1 and n_samples > N_CLUSTERS:
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
except Exception as e:

    print("Warning: metric calculation failed:", e)

# Print results
print("=== sklearn KMeans benchmark ===")
print(f"n_samples: {n_samples}, n_features: {X.shape[1]}, n_clusters: {N_CLUSTERS}")
print(f"train_time_s: {train_time_s:.6f} s")
print(f"predict_time_s (on {X_pred.shape[0]} samples): {predict_time_s:.6f} s")
print(f"predict_per_sample_ms: {predict_per_sample_ms:.6f} ms")
print(f"inertia: {inertia:.6f}")
print(f"silhouette_score: {sil_score}")
print(f"davies_bouldin_score: {db_score}")
print(f"calinski_harabasz_score: {ch_score}")


