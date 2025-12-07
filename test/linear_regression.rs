use ML_ALGO_Rewite::{data, preprocess, split, linear, model_io};
use ndarray::{Array1, Array2, s};
use std::time::Instant;

/// Helper stats (works with ndarray::Array1<f64>)
fn arr_min(a: &Array1<f64>) -> f64 {
    a.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn arr_max(a: &Array1<f64>) -> f64 {
    a.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}
fn arr_mean(a: &Array1<f64>) -> f64 {
    a.iter().sum::<f64>() / (a.len() as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "examples/Linear_refression_6.xlsx";
    let target_column_name = "Volume"; // Use column name instead of index!
    let test_size = 0.2;
    let seed: u64 = 42;
    
    // --- load excel by column name ---
    let (x, y): (Array2<f64>, Array1<f64>) = data::load_excel_by_name(path, target_column_name, None)?;
    println!("Loaded X Shaped:{:?}, y.len {}", x.dim(), y.len());
    
    // ---- splitting the dataset -----
    let (x_train, x_test, y_train, y_test) = split::train_test_split(&x, &y, test_size, false, Some(seed))?;
    println!("Train Rows:{}, Test rows:{}", x_train.nrows(), x_test.nrows());
    
    // Impute Missing (median)
    let medians = preprocess::column_median(&x_train);
    let x_train_imp = preprocess::impute_median(&x_train, &medians);
    let x_test_imp = preprocess::impute_median(&x_test, &medians);

    // Scale
    let scaler = preprocess::StandardScaler::fit(&x_train_imp);
    let x_train_scaled = scaler.transform(&x_train_imp);
    let x_test_scaled = scaler.transform(&x_test_imp);

    // Add bias (intercept) - bias is prepended at column 0
    let xb_train = preprocess::add_bias(&x_train_scaled);
    let xb_test = preprocess::add_bias(&x_test_scaled);

    // Train (OLS) with timing
    let t0 = Instant::now();
    let coef: Array1<f64> = linear::train_ols(&xb_train, &y_train)?; // length = n_features + 1
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Trained. Train time: {:.3} ms", train_ms);

    // Predict & evaluate
    let t1 = Instant::now();
    let y_pred = linear::predict(&xb_test, &coef);
    let predict_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mse = linear::mse(&y_test, &y_pred);
    let r2 = linear::r2(&y_test, &y_pred);

    println!("MSE: {:.6}", mse);
    println!("R2 : {:.6}", r2);
    println!("Predict time: {:.3} ms", predict_ms);

    // === DIAGNOSTICS ===
    println!("\n--- Debug diagnostics ---");
    println!(
        "Rust: y_test n={}, mean={}, min={}, max={}",
        y_test.len(),
        arr_mean(&y_test),
        arr_min(&y_test),
        arr_max(&y_test)
    );
    println!(
        "Rust: y_pred n={}, mean={}, min={}, max={}",
        y_pred.len(),
        arr_mean(&y_pred),
        arr_min(&y_pred),
        arr_max(&y_pred)
    );

    println!("\nFirst 10 pairs (y_test, y_pred):");
    for i in 0..y_test.len().min(10) {
        println!("{:2}: {:12.6}  {:12.6}", i, y_test[i], y_pred[i]);
    }

    // show first 3 rows of xb_test to check bias location + order
    println!("\nxb_test shape: {:?}", xb_test.dim());
    for i in 0..xb_test.nrows().min(3) {
        let row = xb_test.slice(s![i, ..]);
        println!("xb_test[{}] = {:?}", i, row);
    }

    // print coef (show up to first 12 values)
    let show_n = 12.min(coef.len());
    println!("\ncoef len: {}, first {} coef values = {:?}", coef.len(), show_n, coef.slice(s![..show_n]));

    // manual SSE / MSE / SST / R2 calculation
    let sse: f64 = y_test.iter().zip(y_pred.iter()).map(|(a,b)| (a - b).powi(2)).sum();
    let mse_manual = sse / (y_test.len() as f64);
    let y_mean = arr_mean(&y_test);
    let sst: f64 = y_test.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r2_manual = 1.0 - sse / sst;
    println!("\nManual SSE: {:.6}, Manual MSE: {:.6}", sse, mse_manual);
    println!("Manual SST: {:.6}, Manual R2: {:.6}", sst, r2_manual);
    println!("Compare: linear::mse = {:.6}, manual_mse = {:.6}", mse, mse_manual);

    // Save model (coef, scaler.mean, scaler.std)
    std::fs::create_dir_all("models")?;
    model_io::save_model("models/linear_model_rust.json", &coef, &scaler.mean, &scaler.std)?;
    println!("Saved model -> models/linear_model_rust.json");

    // Load and sanity-check
    let (loaded_coef, loaded_mean, loaded_std) = model_io::load_model("models/linear_model_rust.json")?;
    println!("Loaded coef len: {}, mean len: {}, std len: {}", loaded_coef.len(), loaded_mean.len(), loaded_std.len());

    let loaded_scaler = preprocess::StandardScaler { mean: loaded_mean.clone(), std: loaded_std.clone() };
    let xb_test_from_loaded = preprocess::add_bias(&loaded_scaler.transform(&x_test_imp));
    let y_pred2 = linear::predict(&xb_test_from_loaded, &loaded_coef);

    // Compare predictions (max abs diff)
    let max_abs_diff = y_pred.iter().zip(y_pred2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    println!("Max abs diff between before-save and after-load predictions: {:.12}", max_abs_diff);

    Ok(())
}

