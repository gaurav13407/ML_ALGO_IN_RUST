use std::error::Error;
use std::time::Instant;
use ndarray::Array2;
use ML_ALGO_Rewite::{load_data, data};
use ML_ALGO_Rewite::PCA::PCA;

fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_1.csv";
    
    let ds = load_data::DatasetLoader::from_path(path).build()?;
    let x: Array2<f64> = ds.x;
    let (n_samples, n_features) = (x.nrows(), x.ncols());
    println!("Loaded {} with shape ({}, {})", path, n_samples, n_features);
    
    // Fix: ds.headers doesn't exist, use ds.feature_names
  let feature_names: Vec<String> = if ds.feature_names.is_empty() {
    (1..=n_features).map(|i| format!("Feature_{}", i)).collect()
} else {
    ds.feature_names.clone()
}; 
    
    // Step 1: Full PCA fit to inspect explained variance
   let max_components = n_samples.min(n_features);
let mut pca_full = PCA::new(max_components); 
    let start_full = Instant::now();
    pca_full.fit(&x)?;
    let elapsed_full = start_full.elapsed();
    println!("Full PCA fitted in {:?}", elapsed_full);
    
    let ratios_full = pca_full.explained_variance_ratio(&x);
    println!("Explained variance ratios (full):");
    for (i, r) in ratios_full.iter().enumerate() {
        println!("PC{}: {:.6}", i + 1, r);
    }
    
    // Determine k for >= 95% cumulative variance
    let mut cum = 0.0;
    let mut k95 = max_components;
    for (i, r) in ratios_full.iter().enumerate() {
        cum += *r;
        if cum >= 0.95 {
            k95 = i + 1;
            break;
        }
    }
    println!("Components needed for >=95% variance: {}", k95);
    
    // Step 2: Refit PCA with k95 components
    let mut pca = PCA::new(k95);
    let start_fit = Instant::now();
    pca.fit(&x)?;
    let fit_elapsed = start_fit.elapsed();
    println!("PCA (k={}) fitted in {:?}", k95, fit_elapsed);
    
    // Transform data
    let reduced = pca.transform(&x);
    println!("Reduced shape: ({}, {})", reduced.nrows(), reduced.ncols());
    
    // Step 3: Export CSVs - Fixed function calls
    data::export_components_csv(&pca, &feature_names)?;
    data::export_explained_csv(&pca, &x)?;
    data::export_reduced_csv("rust_pca_reduced.csv", &reduced)?;
    
    println!("Saved CSVs:");
    println!(" - rust_pca_components.csv");
    println!(" - rust_pca_explained.csv");
    println!(" - rust_pca_reduced.csv");
    
    Ok(())
}
