mod data;
mod preprocess;
mod spilt;
mod linear;
mod modle_io;
use ndarray::{Array1,Array2};

use std::time::Instant;

pub fn main()-> Result<(), Box<dyn std::error::Error>> {
let PATH="../examples/Linear_regression_1.csv";
let Target_column="medain_house_price";
let TEST_SIZE=0.2;
let seed:u64=42;
    //---load csv---
    let (mut X,y):(Array2<f64>,Array1<f64>)=data::load_csv(PATH,Target_column)?;
    println!("Loaded X Shaped:{:?},y.len{}",X.dim(),y.len());
    //----spilting the dataset-----

    let (X_train,X_test,y_train,y_test)=spilt::train_test_spilt(&X,&y,TEST_SIZE,seed);
    println!("Train Rows:{}, Test rows:{}",X_train.nrows(),X_test.nrows());
    //Impute Missing (median)------
    let medains=preprocess::column_median(&X_train);
    let X_train_imp=preprocess::impute_medain(&X_train,&medains);
    let X_test_imp=preprocess::impute_medain(&X_train,&medains);

    //------scale---------
    let scaler=preprocess::StandardScaler::fit(&X_train_imp);
    let X_train_scaled = scaler.transform(&X_train_imp);
    let X_test_scaled  = scaler.transform(&X_test_imp);

    // ---------- 5) add bias (intercept) ----------
    let Xb_train = preprocess::add_bias(&X_train_scaled);
    let Xb_test  = preprocess::add_bias(&X_test_scaled);

    // ---------- 6) train (OLS) with timing ----------
    let t0 = Instant::now();
    let coef: Array1<f64> = linear::train_ols(&Xb_train, &y_train)?; // length = n_features + 1
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Trained. Train time: {:.3} ms", train_ms);

    // ---------- 7) predict & evaluate ----------
    let t1 = Instant::now();
    let y_pred = Xb_test.dot(&coef);
    let predict_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mse = crate::linear::mse(&y_test, &y_pred);
    let r2  = crate::linear::r2(&y_test, &y_pred);

    println!("MSE: {:.6}", mse);
    println!("R2 : {:.6}", r2);
    println!("Predict time: {:.3} ms", predict_ms);

    // ---------- 8) save model ----------
    // Save coef, scaler.mean, scaler.std
    std::fs::create_dir_all("models")?;
    model_io::save_model("models/linear_model_rust.json", &coef, &scaler.mean, &scaler.std)?;
    println!("Saved model -> models/linear_model_rust.json");

    // ---------- 9) load model back and sanity-check ----------
    let (loaded_coef, loaded_mean, loaded_std) = model_io::load_model("models/linear_model_rust.json")?;
    println!("Loaded coef len: {}, mean len: {}, std len: {}", loaded_coef.len(), loaded_mean.len(), loaded_std.len());

    // Reconstruct scaler and predict again to ensure exact load
    let loaded_scaler = preprocess::StandardScaler { mean: loaded_mean.clone(), std: loaded_std.clone() };
    let Xb_test_from_loaded = preprocess::add_bias(&loaded_scaler.transform(&X_test_imp));
    let y_pred2 = Xb_test_from_loaded.dot(&loaded_coef);

    // Compare predictions (max abs diff)
    let max_abs_diff = y_pred.iter().zip(y_pred2.iter()).map(|(a,b)| (a - b).abs()).fold(0.0_f64, f64::max);
    println!("Max abs diff between before-save and after-load predictions: {:.12}", max_abs_diff);

    Ok(())
}
