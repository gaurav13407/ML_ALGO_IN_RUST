use ndarray::{Array1, Array2, array, ArrayView2, s};
use std::collections::HashMap;

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

/// ROC-AUC SCORE (binary) using rank-based method
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

    // Sort by predicted probability (handle NaN gracefully)
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
/// Warning: O(n²) complexity - use with caution on large datasets (>5000 samples)
pub fn silhouette_score(x: &ArrayView2<f64>, labels: &Array1<usize>) -> Option<f64> {
    let (n, _d) = x.dim();

    // Skip for very large datasets to prevent freeze
    if n > 10000 {
        eprintln!("Warning: Dataset too large ({} samples) for silhouette_score. Consider using silhouette_score_sampled() instead.", n);
        return None;
    }

    // unique labels
    let mut unique_labels: Vec<usize> = labels.iter().cloned().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();

    if k <= 1 || n <= k {
        return None;
    }

    // Build index lists per cluster
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

/// Silhouette score with sampling for large datasets
/// Randomly samples up to max_samples points for calculation
pub fn silhouette_score_sampled(
    x: &ArrayView2<f64>,
    labels: &Array1<usize>,
    max_samples: usize,
) -> Option<f64> {
    let n = x.nrows();
    
    if n <= max_samples {
        // Use regular silhouette if dataset is small enough
        return silhouette_score(x, labels);
    }
    
    // Sample indices
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut thread_rng());
    let sample_indices = &indices[..max_samples];
    
    // Extract sampled rows
    let mut x_sample = Array2::<f64>::zeros((max_samples, x.ncols()));
    let mut labels_sample = Array1::<usize>::zeros(max_samples);
    
    for (i, &idx) in sample_indices.iter().enumerate() {
        x_sample.row_mut(i).assign(&x.row(idx));
        labels_sample[i] = labels[idx];
    }
    
    silhouette_score(&x_sample.view(), &labels_sample)
}

/// Davies–Bouldin index. Lower is better. Returns None if undefined.
pub fn davies_bouldin_score(
    x: &ArrayView2<f64>,
    labels: &Array1<usize>,
    centroids: &Array2<f64>,
) -> Option<f64> {
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
        global = global + &row;
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

/// Cumulative explained variance
pub fn cumulative_explained_variance(explained_variance: &Array1<f64>) -> Array1<f64> {
    let mut cum = Array1::<f64>::zeros(explained_variance.len());
    let mut acc = 0.0;
    for (i, &v) in explained_variance.iter().enumerate() {
        acc += v;
        cum[i] = acc;
    }
    cum
}

/// Sign-invariant cosine similarity between corresponding components
pub fn component_similarity(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let k = a.nrows();
    let mut sims = Array1::<f64>::zeros(k);
    for i in 0..k {
        let ai = a.row(i);
        let bi = b.row(i);
        let dot = ai.dot(&bi);
        // Manual L2 norm: sqrt(sum of squares)
        let norm_ai = ai.dot(&ai).sqrt();
        let norm_bi = bi.dot(&bi).sqrt();
        let denom = norm_ai * norm_bi;
        if denom < 1e-10 {
            sims[i] = 0.0;
        } else {
            sims[i] = (dot / denom).abs();
        }
    }
    sims
}

/// Reconstruction R^2
pub fn reconstruction_r2(original: &Array2<f64>, recon: &Array2<f64>) -> f64 {
    assert_eq!(original.dim(), recon.dim());
    let (n, d) = original.dim();

    let mut mean = Array1::<f64>::zeros(d);
    for row in original.rows() {
        mean = mean + &row;
    }
    mean /= n as f64;

    let mut sse = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        let orig_row = original.row(i);
        let recon_row = recon.row(i);
        for j in 0..d {
            let diff_r = orig_row[j] - recon_row[j];
            sse += diff_r * diff_r;
            let diff_t = orig_row[j] - mean[j];
            sst += diff_t * diff_t;
        }
    }

    if sst == 0.0 {
        return 1.0;
    }
    1.0 - (sse / sst)
}

/// Purity score
pub fn purity_score(labels_true: &Array1<usize>, labels_pred: &Array1<usize>) -> f64 {
    assert_eq!(labels_true.len(), labels_pred.len());
    let n = labels_true.len();

    let mut cluster_map: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for i in 0..n {
        let lt = labels_true[i];
        let lp = labels_pred[i];
        cluster_map
            .entry(lp)
            .or_default()
            .entry(lt)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    let mut correct = 0usize;
    for (_cluster, cmap) in cluster_map.into_iter() {
        let max_in_cluster = cmap.into_iter().map(|(_lab, cnt)| cnt).max().unwrap_or(0);
        correct += max_in_cluster;
    }
    (correct as f64) / (n as f64)
}

/// Helper: n choose 2
fn comb2(n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        (n * (n - 1) / 2) as f64
    }
}

/// Adjusted Rand Index (ARI)
pub fn adjusted_rand_index(labels_true: &Array1<usize>, labels_pred: &Array1<usize>) -> f64 {
    assert_eq!(labels_true.len(), labels_pred.len());
    let n = labels_true.len();

    let mut contingency: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    let mut sum_rows: HashMap<usize, usize> = HashMap::new();
    let mut sum_cols: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let t = labels_true[i];
        let p = labels_pred[i];
        contingency
            .entry(p)
            .or_default()
            .entry(t)
            .and_modify(|c| *c += 1)
            .or_insert(1);
        *sum_rows.entry(p).or_insert(0) += 1;
        *sum_cols.entry(t).or_insert(0) += 1;
    }

    let mut sum_comb_c = 0.0;
    for (_p, colmap) in contingency.iter() {
        for (_t, &cnt) in colmap.iter() {
            sum_comb_c += comb2(cnt);
        }
    }

    let mut sum_comb_rows = 0.0;
    for (_p, &cnt) in sum_rows.iter() {
        sum_comb_rows += comb2(cnt);
    }

    let mut sum_comb_cols = 0.0;
    for (_t, &cnt) in sum_cols.iter() {
        sum_comb_cols += comb2(cnt);
    }

    let total_comb = comb2(n);
    let index = sum_comb_c;
    let expected_index = (sum_comb_rows * sum_comb_cols) / total_comb;
    let max_index = 0.5 * (sum_comb_rows + sum_comb_cols);

    if (max_index - expected_index).abs() < 1e-12 {
        return 0.0;
    }
    (index - expected_index) / (max_index - expected_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_binary_metrics() {
        let y_true = array![0., 1., 1., 0., 1.];
        let y_pred = array![0., 1., 0., 0., 1.];
        assert_eq!(confusion_counts(&y_true, &y_pred, 0.5), (2, 0, 1, 2));
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
