use std::error::Error;
use std::time::Instant;
use ndarray::{s, Array1, Array2};
use ML_ALGO_Rewite::{load_data, metrics, Kmeans};

fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_5.csv";
    let k = 2;
    let max_iter = 300;
    let tol = 1e-4;
    let rng_seed = Some(42u64);
    
    let ds = load_data::DatasetLoader::from_path(path).build()?;
    let x: Array2<f64> = ds.x;
    let x_view = x.view();
    
    let mut kmeans = Kmeans::KMeans::new(k)
        .max_iter(max_iter)
        .tol(tol);
    if let Some(seed) = rng_seed {
        kmeans = kmeans.rng_seed(seed);
    }
    
    let start_fit = Instant::now();
    let labels: Array1<usize> = kmeans.fit(&x_view);
    let fit_elapsed = start_fit.elapsed();
    println!("Fit time         = {:?}", fit_elapsed);
    
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
    
    let centroids = kmeans
        .centroids()
        .expect("KMeans must be fitted before computing metrics");
    
    let inertia_val = metrics::inertia(&x_view, &labels, centroids);
    println!("Inertia                 = {}", inertia_val);
    
    // THIS IS THE ONLY CHANGE - ADD THIS CHECK
    let n_samples = x_view.nrows();
    if n_samples > 5000 {
        println!("Silhouette score        = skipped (dataset too large: {} samples)", n_samples);
    } else {
        match metrics::silhouette_score(&x_view, &labels) {
            Some(s) => println!("Silhouette score        = {}", s),
            None => println!("Silhouette score        = undefined (need >1 cluster and n > k)"),
        }
    }
    
    match metrics::davies_bouldin_score(&x_view, &labels, centroids) {
        Some(db) => println!("Davies–Bouldin index    = {}", db),
        None => println!("Davies–Bouldin index    = undefined"),
    }
    
    match metrics::calinski_harabasz_score(&x_view, &labels, centroids) {
        Some(ch) => println!("Calinski–Harabasz score = {}", ch),
        None => println!("Calinski–Harabasz score = undefined"),
    }
    
    Ok(())
}
