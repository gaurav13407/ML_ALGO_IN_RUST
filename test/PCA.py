import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(X, n_components=None, random_state=0, verbose=True):
    X = np.asarray(X, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_
    ev = pca.explained_variance_
    cum = np.cumsum(evr)

    if verbose:
        print("PCA fitted. n_components =", pca.n_components_)
        print("Explained variance ratio:", np.round(evr, 6))
        print("Cumulative explained variance (%):", np.round(cum * 100, 3))
        print("X_reduced shape:", X_reduced.shape)

    return {
        "pca": pca,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "X_reduced": X_reduced,
        "components": pca.components_,
        "explained_variance_ratio": evr,
        "explained_variance": ev,
        "cumulative_variance": cum
    }

def export_csvs(df_num, pca_res, out_prefix="pca"):
    # Reduced data
    n_reduced = pca_res["X_reduced"].shape[1]
    reduced_df = pd.DataFrame(pca_res["X_reduced"], columns=[f"PC{i+1}" for i in range(n_reduced)])
    reduced_path = f"{out_prefix}_reduced.csv"
    reduced_df.to_csv(reduced_path, index=False)

    # Components: rows = PC, columns = feature names (important: preserve feature order)
    features = list(df_num.columns)
    comps = pd.DataFrame(pca_res["components"], columns=features)
    # add a Component column for clarity
    comps.insert(0, "Component", [f"PC{i+1}" for i in range(comps.shape[0])])
    comps_path = f"{out_prefix}_components.csv"
    comps.to_csv(comps_path, index=False)

    # Explained variance
    evr_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_res["explained_variance_ratio"]))],
        "ExplainedVarianceRatio": pca_res["explained_variance_ratio"],
        "ExplainedVariance": pca_res["explained_variance"],
        "CumulativeVariance": pca_res["cumulative_variance"]
    })
    evr_path = f"{out_prefix}_explained.csv"
    evr_df.to_csv(evr_path, index=False)

    # Optional: scaled input
    scaled_df = pd.DataFrame(pca_res["X_scaled"], columns=features)
    scaled_path = f"{out_prefix}_scaled.csv"
    scaled_df.to_csv(scaled_path, index=False)

    print("Saved CSVs:")
    print(" -", reduced_path)
    print(" -", comps_path)
    print(" -", evr_path)
    print(" -", scaled_path, "(optional)")

    return reduced_path, comps_path, evr_path, scaled_path

if __name__ == "__main__":
    PATH = r"../examples/KMeans_1.csv"   # <-- change to your file path
    if not os.path.exists(PATH):
        raise SystemExit(f"Data file not found: {PATH}")

    df = pd.read_csv(PATH)
    df_num = df.select_dtypes(include=[np.number])
    print("Loaded", PATH, "shape:", df.shape, "numeric shape:", df_num.shape)

    # Full PCA to inspect explained variance
    full = run_pca(df_num, n_components=None)

    # choose #components to reach >=95% variance
    cum = full["cumulative_variance"]
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    print("Components needed for >=95% variance:", k95)

    # Final PCA with k95 components
    pca_res = run_pca(df_num, n_components=k95, verbose=True)

    # Export CSVs
    export_csvs(df_num, pca_res, out_prefix="pca")

