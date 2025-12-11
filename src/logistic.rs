// src/logistic.rs
use ndarray::{Array1, Array2};
use std::fmt;

/// Logistic Regression (binary) using gradient descent.
///
/// Expects input X to already include a bias column (leading 1.0).
/// Labels y must be 0.0 or 1.0.
pub struct LogisticRegression {
    pub w: Array1<f64>, // weights including bias
}

impl fmt::Debug for LogisticRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogisticRegression {{ w: {:?} }}", self.w)
    }
}

impl LogisticRegression {
    /// Create new model with zero weights of length n_features_with_bias.
    pub fn new(n_features_with_bias: usize) -> Self {
        let w = Array1::<f64>::zeros(n_features_with_bias);
        Self { w }
    }

    /// Numerically stable sigmoid applied element-wise.
    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| {
            if v >= 0.0 {
                let z = (-v).exp();
                1.0 / (1.0 + z)
            } else {
                let z = v.exp();
                z / (1.0 + z)
            }
        })
    }

    /// Predict probabilities P(y=1) (shape: n_samples)
    pub fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let logits = X.dot(&self.w);
        Self::sigmoid(&logits)
    }

    /// Predict binary labels {0.0, 1.0} using threshold 0.5
    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let probs = self.predict_proba(X);
        probs.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 })
    }

    /// Binary cross-entropy (log loss), mean over samples.
    pub fn log_loss(probs: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let eps = 1e-15;
        let clipped = probs.mapv(|p| p.max(eps).min(1.0 - eps)); // avoid log(0)
        let n = clipped.len() as f64;
        let mut sum = 0.0;
        for i in 0..clipped.len() {
            let p = clipped[i];
            let yi = y[i];
            sum += -(yi * p.ln() + (1.0 - yi) * (1.0 - p).ln());
        }
        sum / n
    }

    /// Full-batch gradient descent for logistic regression.
    ///
    /// X: (n_samples, n_features_with_bias)
    /// y: (n_samples,) with values 0.0 or 1.0
    /// epochs: number of epochs
    /// lr: learning rate
    /// verbose: if true prints loss
    /// print_every: how often to print (0 => never print)
    pub fn fit(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        epochs: usize,
        lr: f64,
        l2: f64,
        verbose: bool,
        print_every: usize,
    ) {
        let n = X.nrows() as f64;
        let print_every = if print_every == 0 {
            usize::MAX
        } else {
            print_every
        };

        for epoch in 0..epochs {
            // forward
            let probs = self.predict_proba(X);

            // gradient: (1/n) * X^T (probs - y)
            let residual = &probs - y; // shape (n_samples,)
            let mut grad = (1.0 / n) * X.t().dot(&residual); // shape (n_features_with_bias,)
            if l2 > 0.0 {
                let mut reg = self.w.clone(); // clone current weights
                reg *= l2 / n; // scale by lambda/n
                let bias_index = self.w.len() - 1; // assume bias is last column
                reg[bias_index] = 0.0; // no regularization on bias
                grad += &reg; // add regularization to gradient
            }

            // update weights
            self.w = &self.w - &(grad * lr);

            if verbose && (epoch % print_every == 0 || epoch + 1 == epochs) {
                let loss = Self::log_loss(&probs, y);
                println!("epoch {}/{}  log_loss = {:.6}", epoch + 1, epochs, loss);
            }
        }
    }
}
