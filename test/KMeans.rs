use std::error::Error;
use std::time::Instant;

use ndarray::{s, Array1, Array2};
use ML_ALGO_Rewite::{load_data, metrics, Kmeans};

fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_1.csv";
    let k = 2;
    let max_iter = 300;
    let tol = 1e-4;
    let rng_seed = Some(42u64); // or None for random every run

    // 1) Load dataset (X only, no target)
    let ds = load_data::DatasetLoader::from_path(path).build()?;
    let x: Array2<f64> = ds.x;
    let x_view = x.view(); // reuse this for fit + metrics

    // 2) Init KMeans
    let mut kmeans = Kmeans::KMeans::new(k)
        .max_iter(max_iter)
        .tol(tol);

    if let Some(seed) = rng_seed {
        kmeans = kmeans.rng_seed(seed);
    }

    // 3) Time FIT
    let start_fit = Instant::now();
    let labels: Array1<usize> = kmeans.fit(&x_view);
    let fit_elapsed = start_fit.elapsed();
    println!("Fit time         = {:?}", fit_elapsed);

    // 4) Time PREDICT (using same X here)
    let start_pred = Instant::now();
    let pred_labels: Array1<usize> = kmeans.predict(&x_view);
    let pred_elapsed = start_pred.elapsed();

    let n_samples = x_view.nrows();
    let per_sample = pred_elapsed.as_secs_f64() / n_samples as f64;

    println!("Predict time     = {:?}", pred_elapsed);
    println!("Per-sample pred  ≈ {:.9} seconds", per_sample);

    let n_show = labels.len().min(10);
    println!("First {} labels (fit):     {:?}", n_show, labels.slice(s![..n_show]));
    println!("First {} labels (predict): {:?}", n_show, pred_labels.slice(s![..n_show]));

    // 5) CLUSTERING METRICS (from metrics.rs)
    let centroids = kmeans
        .centroids()
        .expect("KMeans must be fitted before computing metrics");

    // Inertia
    let inertia_val = metrics::inertia(&x_view, &labels, centroids);
    println!("Inertia                 = {}", inertia_val);

    // Silhouette score (higher is better)
    match metrics::silhouette_score(&x_view, &labels) {
        Some(s) => println!("Silhouette score        = {}", s),
        None => println!("Silhouette score        = undefined (need >1 cluster and n > k)"),
    }

    // Davies–Bouldin (lower is better)
    match metrics::davies_bouldin_score(&x_view, &labels, centroids) {
        Some(db) => println!("Davies–Bouldin index    = {}", db),
        None => println!("Davies–Bouldin index    = undefined"),
    }

    // Calinski–Harabasz (higher is better)
    match metrics::calinski_harabasz_score(&x_view, &labels, centroids) {
        Some(ch) => println!("Calinski–Harabasz score = {}", ch),
        None => println!("Calinski–Harabasz score = undefined"),
    }

    Ok(())
}

