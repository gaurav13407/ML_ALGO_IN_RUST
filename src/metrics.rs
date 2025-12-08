// src/utils/metrics.rs
use ndarray::{Array1, Array2, array};

fn to_binary_labels(arr: &Array1<f64>, threshold: f64) -> Vec<usize> {
    arr.iter().map(|v| if *v >= threshold { 1usize } else { 0usize }).collect()
}

/// Return counts (tn, fp, fn, tp)
pub fn confusion_counts(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    threshold: f64,
) -> (usize, usize, usize, usize) {
    assert_eq!(y_true.len(), y_pred.len(), "y_true and y_pred must have same length");
    let t = to_binary_labels(y_true, threshold);
    let p = to_binary_labels(y_pred, threshold);

    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tp = 0usize;

    for i in 0..t.len() {
        match (t[i], p[i]) {
            (0, 0) => tn += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            (1, 1) => tp += 1,
            _ => (),
        }
    }
    (tn, fp, fn_, tp)
}

/// Accuracy = (tp + tn) / total
pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let (tn, _fp, _fn, tp) = confusion_counts(y_true, y_pred, threshold);
    let total = y_true.len() as f64;
    (tp + tn) as f64 / total
}

/// Precision = tp / (tp + fp)
pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let (_tn, fp, _fn, tp) = confusion_counts(y_true, y_pred, threshold);
    let denom = (tp + fp) as f64;
    if denom == 0.0 { 0.0 } else { tp as f64 / denom }
}

/// Recall = tp / (tp + fn)
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let (_tn, _fp, fn_, tp) = confusion_counts(y_true, y_pred, threshold);
    let denom = (tp + fn_) as f64;
    if denom == 0.0 { 0.0 } else { tp as f64 / denom }
}

/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let p = precision(y_true, y_pred, threshold);
    let r = recall(y_true, y_pred, threshold);
    if (p + r) == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// Return confusion matrix as 2x2 ndarray: [[tn, fp], [fn, tp]]
pub fn confusion_matrix_array(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> Array2<usize> {
    let (tn, fp, fn_, tp) = confusion_counts(y_true, y_pred, threshold);
    Array2::from_shape_vec((2, 2), vec![tn, fp, fn_, tp]).unwrap()
}

// ROC-AUC SCORE (binary) using rank-based method
pub fn roc_auc_score(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64 {
    assert_eq!(
        y_true.len(),
        y_proba.len(),
        "y_true and y_proba must have same length"
    );

    // Convert to Vec<(proba, true_label)>
    let mut pairs: Vec<(f64, f64)> = y_true
        .iter()
        .zip(y_proba.iter())
        .map(|(&yt, &yp)| (yp, yt))
        .collect();

    // Sort by predicted probability
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut rank_sum_pos = 0.0;
    let mut rank = 1.0;

    let mut n_pos = 0.0;
    let mut n_neg = 0.0;

    for (_p, yt) in pairs {
        if yt == 1.0 {
            n_pos += 1.0;
            rank_sum_pos += rank;
        } else {
            n_neg += 1.0;
        }
        rank += 1.0;
    }

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.0; // undefined case
    }

    let auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
    auc
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_metrics() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred_prob = array![0.9, 0.6, 0.2, 0.1];
        let thresh = 0.5;
        let (tn, fp, fn_, tp) = confusion_counts(&y_true, &y_pred_prob, thresh);
        assert_eq!((tn, fp, fn_, tp), (1, 1, 1, 1));
        assert!((accuracy(&y_true, &y_pred_prob, thresh) - 0.5).abs() < 1e-12);
        assert!((precision(&y_true, &y_pred_prob, thresh) - 0.5).abs() < 1e-12);
        assert!((recall(&y_true, &y_pred_prob, thresh) - 0.5).abs() < 1e-12);
        assert!((f1_score(&y_true, &y_pred_prob, thresh) - 0.5).abs() < 1e-12);
    }
}

