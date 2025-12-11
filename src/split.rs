// src/split.rs
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::error::Error;

/// Train/test split for ndarray arrays.
///
/// - `X` shape: (n_samples, n_features)
/// - `y` shape: (n_samples,)
/// - `test_size`: fraction in (0.0, 1.0)
/// - `shuffle`: if true, rows are shuffled before splitting
/// - `seed`: Some(seed) uses deterministic StdRng(seed), None uses random StdRng::from_entropy()
///
/// Returns: (X_train, X_test, y_train, y_test)
pub fn train_test_split(
    X: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> {
    let n = X.nrows();
    if n == 0 {
        return Err("Empty dataset".into());
    }
    if n != y.len() {
        return Err("X.nrows() must equal y.len()".into());
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err("test_size must be between 0 and 1".into());
    }

    // build indices
    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
    }

    // compute n_test and ensure at least 1 test and at least 1 train sample
    let mut n_test = ((test_size * (n as f64)).round() as usize).min(n);
    // clamp to [1, n-1]
    if n_test == 0 {
        n_test = 1;
    } else if n_test >= n {
        n_test = n - 1;
    }
    let _n_train = n - n_test;

    // helper to build X by copying selected rows
    let build_X = |idxs: &[usize]| -> Result<Array2<f64>, Box<dyn Error>> {
        if idxs.is_empty() {
            // return empty with correct number of columns
            return Ok(Array2::from_shape_vec((0, X.ncols()), vec![])?);
        }
        let mut flat = Vec::with_capacity(idxs.len() * X.ncols());
        for &i in idxs {
            let row = X.row(i);
            for v in row.iter() {
                flat.push(*v);
            }
        }
        Ok(Array2::from_shape_vec((idxs.len(), X.ncols()), flat)?)
    };

    // helper to build y by copying selected elements
    let build_y = |idxs: &[usize]| -> Array1<f64> {
        let mut v = Vec::with_capacity(idxs.len());
        for &i in idxs {
            v.push(y[i]);
        }
        Array1::from_vec(v)
    };

    let test_idx = &indices[..n_test];
    let train_idx = &indices[n_test..];

    let X_train = build_X(train_idx)?;
    let X_test = build_X(test_idx)?;
    let y_train = build_y(train_idx);
    let y_test = build_y(test_idx);

    Ok((X_train, X_test, y_train, y_test))
}

/// Variant that also returns the train and test index vectors (zero-based).
/// Useful for writing indices to disk so Python and Rust can evaluate on identical rows.
pub fn train_test_split_with_indices(
    X: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Vec<usize>,
        Vec<usize>,
    ),
    Box<dyn Error>,
> {
    // reuse train_test_split to get arrays, but regenerate the indices the same way
    let n = X.nrows();
    if n == 0 {
        return Err("Empty dataset".into());
    }
    if n != y.len() {
        return Err("X.nrows() must equal y.len()".into());
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err("test_size must be between 0 and 1".into());
    }

    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
    }

    let mut n_test = ((test_size * (n as f64)).round() as usize).min(n);
    if n_test == 0 {
        n_test = 1;
    } else if n_test >= n {
        n_test = n - 1;
    }

    let test_idx = indices[..n_test].to_vec();
    let train_idx = indices[n_test..].to_vec();

    // build arrays (same logic as above)
    let build_X_from = |idxs: &[usize]| -> Result<Array2<f64>, Box<dyn Error>> {
        if idxs.is_empty() {
            return Ok(Array2::from_shape_vec((0, X.ncols()), vec![])?);
        }
        let mut flat = Vec::with_capacity(idxs.len() * X.ncols());
        for &i in idxs {
            let row = X.row(i);
            for v in row.iter() {
                flat.push(*v);
            }
        }
        Ok(Array2::from_shape_vec((idxs.len(), X.ncols()), flat)?)
    };

    let build_y_from = |idxs: &[usize]| -> Array1<f64> {
        let mut v = Vec::with_capacity(idxs.len());
        for &i in idxs {
            v.push(y[i]);
        }
        Array1::from_vec(v)
    };

    let X_train = build_X_from(&train_idx)?;
    let X_test = build_X_from(&test_idx)?;
    let y_train = build_y_from(&train_idx);
    let y_test = build_y_from(&test_idx);

    Ok((X_train, X_test, y_train, y_test, train_idx, test_idx))
}
