use std::time::Instant;
use std::error::Error;
use ndarray::{Array1, Array2};
use ML_ALGO_Rewite::{data, metrics, KNN, preprocess, split};

fn main() -> Result<(), Box<dyn Error>> {
    // --- config
    let path = "examples/Logistic_regression_5.csv";
    let target = "Outcome";
    let test_size = 0.2_f64;
    let seed = 42u64;
    let k_neighbors = 5usize;
    let max_samples = 5000; // Limit samples for faster KNN (KNN is O(n*m*d))

    // ---------------------------
    // Load dataset (fast CSV loading)
    // ---------------------------
    println!("Loading data...");
    let start_load = Instant::now();
    let (mut x, mut y_raw): (Array2<f64>, Array1<f64>) = data::load_csv_by_name(path, target)?;
    println!("Loaded X shape: {:?}, y.len: {} in {:.2?}", x.dim(), y_raw.len(), start_load.elapsed());
    
    // Subsample if dataset is too large (KNN is slow on large datasets)
    if x.nrows() > max_samples {
        println!("Dataset too large ({} rows), sampling {} rows for faster KNN...", x.nrows(), max_samples);
        use rand::seq::SliceRandom;
        use rand::{rngs::StdRng, SeedableRng};
        
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..x.nrows()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(max_samples);
        
        let x_sampled = x.select(ndarray::Axis(0), &indices);
        let y_sampled = y_raw.select(ndarray::Axis(0), &indices);
        x = x_sampled;
        y_raw = y_sampled;
        println!("Sampled X shape: {:?}", x.dim());
    }

    // ---------------------------
    // Train/test split
    // ---------------------------
    // Assumes split::train_test_split(&x, &y, test_size, shuffle, Some(seed))
    let (x_train, x_test, y_train_f64, y_test_f64) =
        split::train_test_split(&x, &y_raw, test_size, true, Some(seed))?;
    
    // Convert y to usize for KNN
    let y_train: Array1<usize> = y_train_f64.mapv(|v| v as usize);
    let y_test: Array1<usize> = y_test_f64.mapv(|v| v as usize);

    // ---------------------------
    // Impute missing values (median)
    // ---------------------------
    let median = preprocess::column_median(&x_train);
    let x_train_imp = preprocess::impute_median(&x_train, &median);
    let x_test_imp = preprocess::impute_median(&x_test, &median);

    // ---------------------------
    // Scale features (StandardScaler)
    // ---------------------------
    let scaler = preprocess::StandardScaler::fit(&x_train_imp);
    let x_train_scaled = scaler.transform(&x_train_imp);
    let x_test_scaled = scaler.transform(&x_test_imp);

    // ---------------------------
    // NOTE: Do NOT add bias column for KNN
    // KNN uses distances over feature space — adding a constant bias column will
    // change distances and typically hurts KNN. (Leave commented.)
    // let xb_train = preprocess::add_bias(&x_train_scaled);
    // let xb_test  = preprocess::add_bias(&x_test_scaled);

    // ---------------------------
    // Init model (KNN)
    // ---------------------------
    // Your earlier KNN::new(k) accepted `k` not n_features — using k_neighbors.
    let knn = KNN::KNN::new(k_neighbors)
        .with_distance(KNN::Distance::Euclid)   // optional: Euclid or Manhattan
        .with_weighting(KNN::Weighting::Uniform); // optional: Uniform or Distance

    // ---------------------------
    // "Training" timing (KNN stores data)
    // ---------------------------
    let start_train = Instant::now();
    // For KNN there's no heavy training; if you need to "fit" (store) do it here.
    // If your KNN has a `.fit()` method, call it here. Example:
    // knn.fit(&x_train_scaled.view(), &y_train);
    let train_time = start_train.elapsed().as_secs_f64();

    // ---------------------------
    // Prediction timing + call
    // ---------------------------
    let start_pred = Instant::now();
    let (y_pred, probs_opt) = knn.predict_class(&x_train_scaled.view(), &y_train, &x_test_scaled.view());
    let pred_time = start_pred.elapsed().as_secs_f64();

    // ---------------------------
    // Extract class-1 probabilities for ROC (binary)
    // ---------------------------
    let y_prob_class1: Vec<f64> = match probs_opt {
        Some(probs) => {
            // assume binary classification; if not, adapt indexing to the positive class index
            probs.into_iter().map(|row| {
                if row.len() > 1 { row[1] } else { row[0] } // fallback
            }).collect()
        }
        None => {
            // fallback: build probabilities from hard preds (0 or 1)
            y_pred.iter().map(|&p| if p == 1 { 1.0 } else { 0.0 }).collect()
        }
    };

    // ---------------------------
    // Metrics (your metrics module)
    // ---------------------------
    // Convert usize predictions to f64 for metrics
    let y_test_f64_metrics = y_test.mapv(|v| v as f64);
    let y_pred_f64 = y_pred.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let y_pred_f64_arr = Array1::from(y_pred_f64);
    let y_prob_arr = Array1::from(y_prob_class1.clone());
    
    let threshold = 0.5; // threshold for binary classification
    let acc = metrics::accuracy(&y_test_f64_metrics, &y_pred_f64_arr, threshold);
    let prec = metrics::precision(&y_test_f64_metrics, &y_pred_f64_arr, threshold);
    let recall = metrics::recall(&y_test_f64_metrics, &y_pred_f64_arr, threshold);
    let f1 = metrics::f1_score(&y_test_f64_metrics, &y_pred_f64_arr, threshold);
    let auc = metrics::roc_auc_score(&y_test_f64_metrics, &y_prob_arr);
    let cm = metrics::confusion_matrix_array(&y_test_f64_metrics, &y_pred_f64_arr, threshold);

    // Format time display: ms or μs for better readability
    let train_time_ms = train_time * 1000.0;
    let pred_time_ms = pred_time * 1000.0;
    
    if train_time_ms >= 1.0 {
        println!("Train Time: {:.3} ms", train_time_ms);
    } else {
        println!("Train Time: {:.3} μs", train_time_ms * 1000.0);
    }
    
    if pred_time_ms >= 1.0 {
        println!("Predict Time: {:.3} ms", pred_time_ms);
    } else {
        println!("Predict Time: {:.3} μs", pred_time_ms * 1000.0);
    }

    println!("Accuracy: {:.4}", acc);
    println!("Precision: {:.4}", prec);
    println!("Recall: {:.4}", recall);
    println!("F1 Score: {:.4}", f1);
    println!("ROC-AUC: {:.4}", auc);
    println!("Confusion Matrix:");
    println!("[[{}, {}],", cm[[0, 0]], cm[[0, 1]]);
    println!(" [{}, {}]]", cm[[1, 0]], cm[[1, 1]]);

    Ok(())
}

