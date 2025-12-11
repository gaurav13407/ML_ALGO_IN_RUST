use nalgebra::DVector;
#[macro_use]
use ndarray::{Array1, Array2, array, ArrayView2, Axis, s};
use std::{f64, usize};

fn to_binary_labels(arr: &Array1<f64>, threshold: f64) -> Vec<usize> {
    arr.iter()
        .map(|v| if *v >= threshold { 1usize } else { 0usize })
        .collect()
}

/// Return counts (tn, fp, fn, tp)
pub fn confusion_counts(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    threshold: f64,
) -> (usize, usize, usize, usize) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have same length"
    );
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
    if denom == 0.0 {
        0.0
    } else {
        tp as f64 / denom
    }
}

/// Recall = tp / (tp + fn)
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let (_tn, _fp, fn_, tp) = confusion_counts(y_true, y_pred, threshold);
    let denom = (tp + fn_) as f64;
    if denom == 0.0 {
        0.0
    } else {
        tp as f64 / denom
    }
}

/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64 {
    let p = precision(y_true, y_pred, threshold);
    let r = recall(y_true, y_pred, threshold);
    if (p + r) == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// Return confusion matrix as 2x2 ndarray: [[tn, fp], [fn, tp]]
pub fn confusion_matrix_array(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    threshold: f64,
) -> Array2<usize> {
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
/// Compute inertia: sum of squared distances from each sample to its assigned centroid.
pub fn inertia(x: &ArrayView2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> f64 {
    let mut total = 0f64;
    for (i, row) in x.rows().into_iter().enumerate() {
        let c = labels[i];
        let diff = &row - &centroids.slice(s![c, ..]);
        total += diff.dot(&diff);
    }
    total
}

/// Euclidean distance between 1-D arrays
fn euclid(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Pairwise distance matrix (n x n). O(n^2) memory/time.
fn pairwise_distances(x: &ArrayView2<f64>) -> Array2<f64> {
    let (n, _d) = x.dim();
    let mut dmat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let xi = x.row(i).to_owned();
        for j in (i + 1)..n {
            let xj = x.row(j).to_owned();
            let d = euclid(&xi, &xj);
            dmat[[i, j]] = d;
            dmat[[j, i]] = d;
        }
    }
    dmat
}

/// Silhouette score (mean over samples). Returns None when undefined.
pub fn silhouette_score(x: &ArrayView2<f64>, labels: &Array1<usize>) -> Option<f64> {
    let (n, _d) = x.dim();

    // unique labels
    let mut unique_labels: Vec<usize> = labels.iter().cloned().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();

    if k <= 1 || n <= k {
        return None;
    }

    // Build index lists per cluster
    use std::collections::HashMap;
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lab) in labels.iter().enumerate() {
        clusters.entry(lab).or_default().push(i);
    }

    // Compute pairwise distances once (O(n^2))
    let dmat = pairwise_distances(x);

    let mut s_vals = Vec::with_capacity(n);

    for i in 0..n {
        let li = labels[i];
        let intra_idxs = &clusters[&li];

        // a = average distance to other points in same cluster (exclude self)
        let a = if intra_idxs.len() == 1 {
            0.0
        } else {
            let mut sum = 0.0;
            for &j in intra_idxs.iter() {
                if j == i {
                    continue;
                }
                sum += dmat[[i, j]];
            }
            sum / ((intra_idxs.len() - 1) as f64)
        };

        // b = min average distance to points in any other cluster
        let mut b = f64::INFINITY;
        for (&other_lab, other_idxs) in clusters.iter() {
            if other_lab == li {
                continue;
            }
            let mut sum = 0.0;
            for &j in other_idxs.iter() {
                sum += dmat[[i, j]];
            }
            let avg = sum / (other_idxs.len() as f64);
            if avg < b {
                b = avg;
            }
        }

        let denom = a.max(b);
        let s = if denom == 0.0 { 0.0 } else { (b - a) / denom };
        s_vals.push(s);
    }

    let mean_s = s_vals.iter().sum::<f64>() / (n as f64);
    Some(mean_s)
}

/// Davies–Bouldin index. Lower is better. Returns None if undefined.
pub fn davies_bouldin_score(
    x: &ArrayView2<f64>,
    labels: &Array1<usize>,
    centroids: &Array2<f64>,
) -> Option<f64> {
    use std::collections::HashMap;
    let (_n, _d) = x.dim();

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lab) in labels.iter().enumerate() {
        clusters.entry(lab).or_default().push(i);
    }
    let k = clusters.len();
    if k <= 1 {
        return None;
    }

    // compute S_i = avg distance of points in cluster i to centroid i
    let mut labs_vec: Vec<usize> = clusters.keys().cloned().collect();
    labs_vec.sort_unstable();

    let mut s_vals: Vec<f64> = Vec::with_capacity(labs_vec.len());
    for &lab in labs_vec.iter() {
        let idxs = &clusters[&lab];
        if idxs.is_empty() {
            s_vals.push(0.0);
            continue;
        }
        let mut sum = 0.0;
        let c = centroids.slice(s![lab, ..]).to_owned();
        for &i in idxs.iter() {
            let x_i = x.row(i).to_owned();
            sum += euclid(&x_i, &c);
        }
        s_vals.push(sum / (idxs.len() as f64));
    }

    // compute R_ij = (S_i + S_j) / M_ij and R_i = max_j R_ij
    let k_len = labs_vec.len();
    let mut r_i_vals = vec![0f64; k_len];
    for (ii, &i_lab) in labs_vec.iter().enumerate() {
        let c_i = centroids.slice(s![i_lab, ..]).to_owned();
        let s_i = s_vals[ii];
        let mut max_r = 0.0;
        for (jj, &j_lab) in labs_vec.iter().enumerate() {
            if i_lab == j_lab {
                continue;
            }
            let c_j = centroids.slice(s![j_lab, ..]).to_owned();
            let s_j = s_vals[jj];
            let m_ij = euclid(&c_i, &c_j);
            if m_ij == 0.0 {
                continue;
            }
            let r = (s_i + s_j) / m_ij;
            if r > max_r {
                max_r = r;
            }
        }
        r_i_vals[ii] = max_r;
    }

    let db = r_i_vals.iter().sum::<f64>() / (k_len as f64);
    Some(db)
}

/// Calinski–Harabasz score. Higher is better. Returns None if undefined.
pub fn calinski_harabasz_score(
    x: &ArrayView2<f64>,
    labels: &Array1<usize>,
    centroids: &Array2<f64>,
) -> Option<f64> {
    use std::collections::HashMap;
    let (n, _d) = x.dim();

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lab) in labels.iter().enumerate() {
        clusters.entry(lab).or_default().push(i);
    }
    let k = clusters.len();
    if k <= 1 || n <= k {
        return None;
    }

    // global centroid
    let mut global = Array1::<f64>::zeros(x.shape()[1]);
    for row in x.rows() {
        global += &row;
    }
    global /= n as f64;

    // between-cluster dispersion B = sum n_k * ||c_k - global||^2
    let mut b_disp = 0.0;
    for (&lab, idxs) in clusters.iter() {
        let c_k = centroids.slice(s![lab, ..]).to_owned();
        let diff = &c_k - &global;
        let val = diff.dot(&diff);
        b_disp += (idxs.len() as f64) * val;
    }

    // within-cluster dispersion W = inertia
    let w_disp = inertia(x, labels, centroids);
    if w_disp == 0.0 {
        return None;
    }

    let numerator = b_disp / ((k as f64) - 1.0);
    let denominator = w_disp / ((n as f64) - (k as f64));
    Some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_binary_metrics() {
        let y_true = array![0., 1., 1., 0., 1.];
        let y_pred = array![0., 1., 0., 0., 1.];
        assert_eq!(confusion_counts(&y_true, &y_pred, 0.5), (3, 0, 1, 1));
        assert!((accuracy(&y_true, &y_pred, 0.5) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_clustering_metrics_small() {
        let x = array![[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]];
        let labels = array![0usize, 0, 0, 1, 1, 1];
        let centroids = array![[1., 2.], [4., 2.]];

        let inj = inertia(&x.view(), &labels, &centroids);
        assert!(inj > 0.0);

        let sil = silhouette_score(&x.view(), &labels).unwrap();
        assert!(sil > 0.0);

        let db = davies_bouldin_score(&x.view(), &labels, &centroids).unwrap();
        assert!(db >= 0.0);

        let ch = calinski_harabasz_score(&x.view(), &labels, &centroids).unwrap();
        assert!(ch > 0.0);
    }
}

#[cfg(test)]
mod tests_kmeans_metrics {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_inertia_basic() {
        // Two perfect clusters
        // Cluster 0 around (1, 2)
        // Cluster 1 around (4, 2)
        let x = array![[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]];

        let labels = array![0usize, 0, 0, 1, 1, 1];
        let centroids = array![
            [1., 2.], // centroid for cluster 0
            [4., 2.]  // centroid for cluster 1
        ];

        let inj = inertia(&x.view(), &labels, &centroids);
        assert!(inj > 0.0);
    }

    #[test]
    fn test_silhouette_score_basic() {
        let x = array![[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]];

        let labels = array![0usize, 0, 0, 1, 1, 1];

        let sil = silhouette_score(&x.view(), &labels);
        assert!(sil.is_some());
        let sil_val = sil.unwrap();
        assert!(sil_val > 0.0);
    }

    #[test]
    fn test_davies_bouldin_score_basic() {
        let x = array![[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]];

        let labels = array![0usize, 0, 0, 1, 1, 1];
        let centroids = array![[1., 2.], [4., 2.]];

        let db = davies_bouldin_score(&x.view(), &labels, &centroids);
        assert!(db.is_some());
        let db_val = db.unwrap();
        assert!(db_val >= 0.0); // DB index is always >= 0
    }

    #[test]
    fn test_calinski_harabasz_score_basic() {
        let x = array![[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]];

        let labels = array![0usize, 0, 0, 1, 1, 1];
        let centroids = array![[1., 2.], [4., 2.]];

        let ch = calinski_harabasz_score(&x.view(), &labels, &centroids);
        assert!(ch.is_some());
        assert!(ch.unwrap() > 0.0);
    }
}
