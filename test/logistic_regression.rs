// src/bin/logistic_regression.rs
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use csv::StringRecord;

use ML_ALGO_Rewite::{data, preprocess, split, linear, model_io, metrics, encoding, logistic};

const OUT_DIR: &str = "models";

fn main() -> Result<(), Box<dyn Error>> {
    // --- config
    let path = "examples/logistic_regression_5.csv";
    let target_column_name = "Outcome"; // set your target column name
    let test_size = 0.2f64;
    let seed:u64 = 42;
    let l2=1.0;

    // --- load CSV (expects your data::load_csv_by_name to return (X, y))
    let (x, y): (Array2<f64>, Array1<f64>) = data::load_csv_by_name(path, target_column_name)?;
    println!("Loaded X shape: {:?}, y.len: {}", x.dim(), y.len());

    // --- train/test split (expects spilt::train_test_split to return (x_train, x_test, y_train, y_test))
    let (x_train, x_test, y_train, y_test) =
        split::train_test_split(&x, &y, test_size, false, Some(seed))?;
    println!(
        "Data sets — Train rows: {}, Test rows: {}",
        x_train.nrows(),
        x_test.nrows()
    );

    // --- impute missing (median)
    let median = preprocess::column_median(&x_train);
    let x_train_imp = preprocess::impute_median(&x_train, &median);
    let x_test_imp = preprocess::impute_median(&x_test, &median);

    // --- scale (use your StandardScaler API)
    let scaler = preprocess::StandardScaler::fit(&x_train_imp);
    let x_train_scaled = scaler.transform(&x_train_imp);
    let x_test_scaled = scaler.transform(&x_test_imp);

    // --- add bias (assumes preprocess::add_bias exists)
    let xb_train = preprocess::add_bias(&x_train_scaled);
    let xb_test = preprocess::add_bias(&x_test_scaled);

    // --- init model
    let n_features_with_bias = xb_train.ncols();
    let mut model = logistic::LogisticRegression::new(n_features_with_bias);

    // --- train the model
    let epochs = 1000usize;
    let lr = 0.1f64;
    let verbose = true;
    let print_every = epochs / 10;
    let t0 = Instant::now();
    model.fit(&xb_train, &y_train, epochs, lr,l2, verbose,print_every);
    let train_dur = t0.elapsed();

    // --- predict
    let t1 = Instant::now();
    let y_proba = model.predict_proba(&xb_test);
    let y_pred = model.predict(&xb_test);
    let predict_dur = t1.elapsed();

    // --- metrics (use probability threshold)
    let thres = 0.5;
    let acc = metrics::accuracy(&y_test, &y_proba, thres);
    let prec = metrics::precision(&y_test, &y_proba, thres);
    let rec = metrics::recall(&y_test, &y_proba, thres);
    let f1 = metrics::f1_score(&y_test, &y_proba, thres);
    let conf = metrics::confusion_matrix_array(&y_test, &y_proba, thres);
    let rocauc = metrics::roc_auc_score(&y_test, &y_proba);

    println!("\nResults:");
    println!("ROC-AUC: {:.4}", rocauc);
    println!("Accuracy: {:.4}", acc);
    println!("Precision: {:.4}", prec);
    println!("Recall: {:.4}", rec);
    println!("F1: {:.4}", f1);
    println!("Confusion matrix:\n{:?}", conf);
    println!("Train time: {:?}, Predict time: {:?}", train_dur, predict_dur);

    // --- save weights (simple CSV)
    std::fs::create_dir_all(OUT_DIR)?;
    let wpath = Path::new(OUT_DIR).join("logistic_weights.csv");
    let mut f = File::create(&wpath)?;
    writeln!(f, "weight")?;
    for val in model.w.iter() {
        writeln!(f, "{}", val)?;
    }
    println!("Saved weights to {}", wpath.display());

    Ok(())
}
// logistic_regression.rts.csv", OUT_DIR);
