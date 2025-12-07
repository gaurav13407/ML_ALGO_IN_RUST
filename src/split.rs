use ndarray::{Array1, Array2};
use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::StdRng;
use std::error::Error;

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

    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
    }

    let n_test = ((test_size * (n as f64)).round() as usize).min(n);
    let n_train = n - n_test;

    // Build X
    let build_X = |idxs: &[usize]| -> Result<Array2<f64>, Box<dyn Error>> {
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

    // Build y
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

