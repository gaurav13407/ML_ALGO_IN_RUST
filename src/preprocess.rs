use ndarray::{s, Array1, Array2, Axis};

pub fn normalize(X: &mut Array2<f64>) {
    let n_features = X.ncols();

    for col in 0..n_features {
        let column = X.column(col);
        let mean = column.mean().unwrap();
        let std = column.std(0.0);

        let std = if std == 0.0 { 1.0 } else { std };

        for v in X.column_mut(col) {
            *v = (*v - mean) / std;
        }
    }
}

pub fn add_bias(X: &Array2<f64>) -> Array2<f64> {
    let (n_samples, n_features) = X.dim();
    let mut Xb = Array2::<f64>::zeros((n_samples, n_features + 1));

    for i in 0..n_samples {
        Xb[(i, 0)] = 1.0;
    }

    for j in 0..n_features {
        let col = X.column(j);
        let mut target_col = Xb.slice_mut(s![.., j + 1]);
        target_col.assign(&col);
    }
    Xb
}

pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl StandardScaler {
    pub fn fit(X: &Array2<f64>) -> Self {
        let mean = X.mean_axis(Axis(0)).unwrap();
        let mut std = X.std_axis(Axis(0), 0.0);
        std.mapv_inplace(|v| if v == 0.0 { 1.0 } else { v });
        Self { mean, std }
    }

    pub fn transform(&self, X: &Array2<f64>) -> Array2<f64> {
        (X - &self.mean) / &self.std
    }

    pub fn fit_transform(X: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(X);
        let xs = scaler.transform(X);
        (scaler, xs)
    }
}

/// Compute column-wise median (ignores NaN)
pub fn column_median(X: &Array2<f64>) -> Array1<f64> {
    let ncols = X.ncols();
    let mut medians = Array1::<f64>::zeros(ncols);

    for j in 0..ncols {
        let mut col: Vec<f64> = X
            .column(j)
            .iter()
            .cloned()
            .filter(|v| !v.is_nan())
            .collect();

        if col.is_empty() {
            medians[j] = 0.0;
            continue;
        }

        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = col.len();
        medians[j] = if m % 2 == 1 {
            col[m / 2]
        } else {
            (col[m / 2 - 1] + col[m / 2]) / 2.0
        };
    }
    medians
}

/// Impute NaN values with provided column medians (returns new Array2)
pub fn impute_median(X: &Array2<f64>, medians: &Array1<f64>) -> Array2<f64> {
    let mut out = X.clone();
    for j in 0..out.ncols() {
        for i in 0..out.nrows() {
            if out[(i, j)].is_nan() {
                out[(i, j)] = medians[j];
            }
        }
    }
    out
}
