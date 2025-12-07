
// src/linear.rs
use std::error::Error;

use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};

/// Train OLS using normal equation:
/// coef = (X^T X)^(-1) X^T y
/// - X: design matrix (n_samples, n_features) — include bias column if used
/// - y: target vector (n_samples)
///
/// If inversion fails (singular matrix), we add a tiny ridge regularizer (lambda*I)
/// and retry to obtain a stable solution.
pub fn train_ols(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (n_samples, n_features) = x.dim();
    if y.len() != n_samples {
        return Err("X and y row count mismatch".into());
    }

    // Convert ndarray to nalgebra for matrix inversion
    let x_data: Vec<f64> = x.iter().copied().collect();
    let y_data: Vec<f64> = y.iter().copied().collect();
    
    let x_mat = DMatrix::from_row_slice(n_samples, n_features, &x_data);
    let y_vec = DVector::from_vec(y_data);
    
    let xt = x_mat.transpose();
    let xtx = &xt * &x_mat;
    
    // Try to invert XtX
    let xtx_inv = match xtx.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            // Fallback: add a tiny ridge regularization lambda * I and invert again
            let diag_sum: f64 = xtx.diagonal().sum();
            let avg_diag = if n_features > 0 { diag_sum / (n_features as f64) } else { 1.0 };
            let lambda = avg_diag * 1e-8_f64;
            
            let identity = DMatrix::identity(n_features, n_features);
            let xtx_reg = xtx + identity * lambda;
            
            match xtx_reg.try_inverse() {
                Some(inv_reg) => inv_reg,
                None => {
                    return Err("XtX inversion failed even after tiny regularization".into());
                }
            }
        }
    };
    
    let xty = &xt * y_vec;
    let coef_vec = xtx_inv * xty;
    
    // Convert back to ndarray
    let coef = Array1::from_vec(coef_vec.data.as_vec().clone());
    Ok(coef)
}

/// Predict using coef vector
pub fn predict(x: &Array2<f64>, coef: &Array1<f64>) -> Array1<f64> {
    x.dot(coef)
}

/// Mean Squared Error
pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// R² score
pub fn r2(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|a| (a - mean).powi(2)).sum();
    if ss_tot == 0.0 {
        if ss_res == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Accuracy-style metric for regression:
/// fraction of predictions within absolute tolerance `tol`.
/// Example: tol = 10000.0 means count predictions where |y_true - y_pred| <= 10000
#[allow(dead_code)]
pub fn accuracy_abs_tol(y_true: &Array1<f64>, y_pred: &Array1<f64>, tol: f64) -> f64 {
    if y_true.len() == 0 { return 0.0; }
    let count_within = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| (*a - *b).abs() <= tol)
        .count() as f64;
    count_within / (y_true.len() as f64)
}

/// Accuracy-style metric within percentage tolerance.
/// pct: e.g. 0.1 means within 10% of true value.
/// If y_true == 0, fallback to absolute tolerance of `abs_tol_for_zero`.
#[allow(dead_code)]
pub fn accuracy_pct_tol(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    pct: f64,
    abs_tol_for_zero: f64,
) -> f64 {
    if y_true.len() == 0 { return 0.0; }
    let count_within = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| {
            if **a == 0.0 {
                (*a - *b).abs() <= abs_tol_for_zero
            } else {
                (*a - *b).abs() <= pct * a.abs()
            }
        })
        .count() as f64;
    count_within / (y_true.len() as f64)
}
