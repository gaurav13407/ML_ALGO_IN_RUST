use ndarray::{Array1, Array2, Axis, s};

pub fn normalize(X: &mut Array2<f64>) {
    let n_features = X.ncols();

    for col in 0..n_features {
        let column = X.column(col);
        let mean = column.mean().unwrap();
        let std = column.std(0.0);

        let std = if std == 0.0 { 1.0 } else { std };

        // remove `mut` here — `v` is a mutable reference from column_mut()
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
        // avoid zeros
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

