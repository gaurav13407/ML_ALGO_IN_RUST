use ndarray::{Array1, Array2, ArrayView2};
use std::f64;

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Euclid,
    Manhattan,
}

#[derive(Debug, Clone, Copy)]
pub enum Weighting {
    Uniform,
    Distance, // weight = 1/(d+eps)
}

#[derive(Debug)]
pub struct KNN {
    pub k: usize,
    pub distance: Distance,
    pub weighting: Weighting,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "k must be >= 1");
        Self {
            k,
            distance: Distance::Euclid,
            weighting: Weighting::Uniform,
        }
    }

    pub fn with_distance(mut self, d: Distance) -> Self {
        self.distance = d;
        self
    }

    pub fn with_weighting(mut self, w: Weighting) -> Self {
        self.weighting = w;
        self
    }

    // compute distance between two 1-D arrays
    fn distance_fn(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.distance {
            Distance::Euclid => {
                let diff = a - b;
                diff.dot(&diff).sqrt()
            }
            Distance::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        }
    }

    /// For classification: returns predicted labels and per-sample class probabilities.
    /// x_train: (n_train, d), y_train: (n_train,) with class ids as usize (0..C-1)
    /// x_test:  (n_test, d)
    pub fn predict_class(
        &self,
        x_train: &ArrayView2<f64>,
        y_train: &Array1<usize>,
        x_test: &ArrayView2<f64>,
    ) -> (Array1<usize>, Option<Vec<Vec<f64>>>) {
        let n_train = x_train.nrows();
        let n_test = x_test.nrows();
        assert_eq!(n_train, y_train.len());

        // find number of classes (assumes labels are 0..C-1 or similar)
        let n_classes = *y_train.iter().max().unwrap_or(&0) + 1;

        let mut preds = Array1::<usize>::zeros(n_test);
        let mut probs: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; n_test];

        for ti in 0..n_test {
            let xt = x_test.row(ti).to_owned();

            // compute distance to all train points
            let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n_train);
            for i in 0..n_train {
                let xi = x_train.row(i).to_owned();
                let d = self.distance_fn(&xt, &xi);
                dists.push((i, d));
            }

            // sort by distance ascending
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // voting
            let mut class_scores = vec![0.0f64; n_classes];
            for j in 0..self.k.min(dists.len()) {
                let (idx, d) = dists[j];
                let label = y_train[idx];
                let weight = match self.weighting {
                    Weighting::Uniform => 1.0,
                    Weighting::Distance => 1.0 / (d + 1e-12),
                };
                class_scores[label] += weight;
            }

            // predicted class = argmax
            let (best_class, _) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            preds[ti] = best_class;

            // convert scores to probabilities (normalize)
            let sum: f64 = class_scores.iter().sum();
            if sum > 0.0 {
                for c in 0..n_classes {
                    probs[ti][c] = class_scores[c] / sum;
                }
            } else {
                // fallback uniform
                for c in 0..n_classes {
                    probs[ti][c] = 1.0 / (n_classes as f64);
                }
            }
        }

        (preds, Some(probs))
    }

    /// For regression: returns predicted float values (mean or weighted mean)
    pub fn predict_reg(
        &self,
        x_train: &ArrayView2<f64>,
        y_train: &Array1<f64>,
        x_test: &ArrayView2<f64>,
    ) -> Array1<f64> {
        let n_train = x_train.nrows();
        let n_test = x_test.nrows();
        assert_eq!(n_train, y_train.len());

        let mut preds = Array1::<f64>::zeros(n_test);

        for ti in 0..n_test {
            let xt = x_test.row(ti).to_owned();

            let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n_train);
            for i in 0..n_train {
                let xi = x_train.row(i).to_owned();
                let d = self.distance_fn(&xt, &xi);
                dists.push((i, d));
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut num = 0.0;
            let mut denom = 0.0;
            for j in 0..self.k.min(dists.len()) {
                let (idx, d) = dists[j];
                let y = y_train[idx];
                let w = match self.weighting {
                    Weighting::Uniform => 1.0,
                    Weighting::Distance => 1.0 / (d + 1e-12),
                };
                num += w * y;
                denom += w;
            }
            preds[ti] = if denom == 0.0 { 0.0 } else { num / denom };
        }
        preds
    }
}
