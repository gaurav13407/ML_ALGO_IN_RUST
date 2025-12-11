
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X = pd.read_csv("examples/KMeans_1.csv").values  # ensure this matches Rust X
p = PCA(n_components=5)
p.fit(X)
# components_ is shape (n_components, n_features)
np.savetxt("examples/pca_components_sklearn.csv", p.components_, delimiter=",")
print("Saved sklearn components to examples/pca_components_sklearn.csv")
