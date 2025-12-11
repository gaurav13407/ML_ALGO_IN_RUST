use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::svd::SVD;
use std::f64;

#[derive(Debug, Clone)]
pub struct PCA {
    pub n_components: usize,
    pub components: Option<Array2<f64>>,         // shape: (n_components, n_features)
    pub explained_variance: Option<Array1<f64>>, // length n_components
    pub mean: Option<Array1<f64>>,               // length n_features
    pub singular_values: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            components: None,
            explained_variance: None,
            mean: None,
            singular_values: None,
        }
    }

    /// Fit PCA using SVD on centered X (shape: n_samples x n_features)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
        let (n_samples, n_features) = x.dim();
        assert!(
            self.n_components <= n_features,
            "n_components <= n_features required"
        );

        // compute mean and center X
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x.clone();
        for mut row in x_centered.outer_iter_mut() {
            row -= &mean;
        }

        // SVD: X_centered = U * S * Vt
        let svd = x_centered.svd(true, true)?;
        let s = svd.s; // singular values
        let vt = svd.v_t.expect("vt must be available");

        // components are first n_components rows of Vt
        let comps = vt.slice(s![0..self.n_components, ..]).to_owned();

        // explained_variance: (S^2) / (n_samples - 1)
        let mut explained = Array1::<f64>::zeros(self.n_components);
        for i in 0..self.n_components {
            let si = s[i];
            explained[i] = (si * si) / ((n_samples - 1) as f64);
        }

        self.components = Some(comps);
        self.explained_variance = Some(explained);
        self.mean = Some(mean);
        self.singular_values = Some(s.slice(s![0..self.n_components]).to_owned());

        Ok(())
    }

    /// Transform data to principal components (scores): shape -> (n_samples, n_components)
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("PCA not fitted");
        let comps = self.components.as_ref().expect("PCA not fitted"); // (k, d)

        let mut x_centered = x.clone();
        for mut row in x_centered.outer_iter_mut() {
            row -= mean;
        }

        // scores = X_centered dot components.T  => (n_samples, k)
        x_centered.dot(&comps.t())
    }

    /// Inverse transform from scores back to original space: X_recon = scores dot components + mean
    pub fn inverse_transform(&self, scores: &Array2<f64>) -> Array2<f64> {
        let comps = self.components.as_ref().expect("PCA not fitted"); // (k, d)
        let mean = self.mean.as_ref().expect("PCA not fitted");

        let mut recon = scores.dot(comps); // (n_samples, d)
        for mut row in recon.outer_iter_mut() {
            row += mean;
        }
        recon
    }

    /// Explained variance ratio per component: explained_variance / total_variance
    pub fn explained_variance_ratio(&self, x: &Array2<f64>) -> Array1<f64> {
        let explained = self.explained_variance.as_ref().expect("PCA not fitted");

        let n_samples = x.nrows();
        // total variance = sum of variances of features
        let mut total_var = 0f64;
        let mean = x.mean_axis(Axis(0)).unwrap();
        for col in 0..x.ncols() {
            let colvec = x.slice(s![.., col]);
            let mut sumsq = 0f64;
            for v in colvec.iter() {
                let diff = v - mean[col];
                sumsq += diff * diff;
            }
            total_var += sumsq / ((n_samples - 1) as f64);
        }

        let mut ratio = Array1::<f64>::zeros(explained.len());
        for i in 0..explained.len() {
            ratio[i] = explained[i] / total_var;
        }
        ratio
    }

    /// Reconstruction mean squared error between original X and its reconstruction using given n_components
    pub fn reconstruction_mse(&self, x: &Array2<f64>) -> f64 {
        let scores = self.transform(x);
        let recon = self.inverse_transform(&scores);
        let diff = &recon - x;
        let mse = diff.mapv(|v| v * v).sum() / ((x.nrows() * x.ncols()) as f64);
        mse
    }
}

