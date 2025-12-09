use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rand::prelude::*;
use rand_distr::WeightedIndex;
use std::f64;

#[derive(Debug, Clone)]
pub struct KMeans {
    k: usize,
    max_iter: usize,
    tol: f64,
    pub centroids: Option<Array2<f64>>,
    rng_seed: Option<u64>,
    pub inertia: Option<f64>,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "k must be >= 1");
        KMeans {
            k,
            max_iter: 300,
            tol: 1e-4,
            centroids: None,
            rng_seed: None,
            inertia: None,
        }
    }

    pub fn max_iter(mut self, it: usize) -> Self {
        self.max_iter = it;
        self
    }

    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    pub fn rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = Some(seed);
        self
    }

    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Array1<usize> {
        let (n_samples, n_features) = x.dim();
        assert!(n_samples >= self.k, "n_samples must be >= k");

        let mut rng = match self.rng_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut centroids = Self::kmeans_plus_plus_init(self.k, x, &mut rng);

        let mut labels = Array1::<usize>::zeros(n_samples);
        let mut prev_centroids = centroids.clone();

        for _ in 0..self.max_iter {
            // assign labels
            for (i, row) in x.rows().into_iter().enumerate() {
                labels[i] = nearest_centroid_index(&row.to_owned(), &centroids);
            }

            // update centroids
            let mut new_centroids = Array2::<f64>::zeros((self.k, n_features));
            let mut counts = vec![0usize; self.k];

            for (i, row) in x.rows().into_iter().enumerate() {
                let c = labels[i];
                {
                    let mut target = new_centroids.slice_mut(s![c, ..]);
                    target += &row;
                }
                counts[c] += 1;
            }

            for c in 0..self.k {
                if counts[c] == 0 {
                    // empty cluster -> reinitialize to a random sample
                    let idx = rng.gen_range(0..n_samples);
                    new_centroids
                        .slice_mut(s![c, ..])
                        .assign(&x.row(idx));
                } else {
                    let mut row = new_centroids.slice_mut(s![c, ..]);
                    row /= counts[c] as f64;
                }
            }

            // check movement
            let max_move = max_centroid_move(&prev_centroids, &new_centroids);
            prev_centroids.assign(&new_centroids);
            centroids = new_centroids;

            if max_move <= self.tol {
                break;
            }
        }

        // compute inertia (sum squared distance to nearest centroid)
        let mut inertia = 0f64;
        for (i, row) in x.rows().into_iter().enumerate() {
            let cidx = labels[i];
            let diff = &row - &centroids.slice(s![cidx, ..]);
            inertia += diff.dot(&diff);
        }

        self.centroids = Some(centroids);
        self.inertia = Some(inertia);

        labels
    }

    pub fn predict(&self, x: &ArrayView2<f64>) -> Array1<usize> {
        let centroids = self
            .centroids
            .as_ref()
            .expect("Must fit KMeans before calling predict");
        let (n_samples, _n_features) = x.dim();
        let mut labels = Array1::<usize>::zeros(n_samples);
        for (i, row) in x.rows().into_iter().enumerate() {
            labels[i] = nearest_centroid_index(&row.to_owned(), centroids);
        }
        labels
    }

    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }

    // KMeans++ initialization
    fn kmeans_plus_plus_init<R: Rng + ?Sized>(
        k: usize,
        x: &ArrayView2<f64>,
        rng: &mut R,
    ) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let mut centers = Array2::<f64>::zeros((k, n_features));

        // choose first uniformly
        let first = rng.gen_range(0..n_samples);
        centers.slice_mut(s![0, ..]).assign(&x.row(first));

        let mut distances = vec![0f64; n_samples];

        for c in 1..k {
            // compute squared distance to nearest existing center
            for (i, row) in x.rows().into_iter().enumerate() {
                let mut min_d2 = f64::INFINITY;
                for j in 0..c {
                    let diff = &row - &centers.slice(s![j, ..]);
                    let d2 = diff.dot(&diff);
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                distances[i] = min_d2;
            }

            let dist_sum: f64 = distances.iter().sum();
            if dist_sum == 0.0 {
                // all points identical relative to chosen centers
                let idx = rng.gen_range(0..n_samples);
                centers.slice_mut(s![c, ..]).assign(&x.row(idx));
                continue;
            }

            let weights: Vec<f64> = distances.iter().map(|d| d / dist_sum).collect();
            let dist = WeightedIndex::new(&weights).expect("weights valid");
            let chosen = dist.sample(rng);
            centers.slice_mut(s![c, ..]).assign(&x.row(chosen));
        }

        centers
    }
}

// helper: nearest centroid index
fn nearest_centroid_index(row: &Array1<f64>, centroids: &Array2<f64>) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for (j, c_row) in centroids.rows().into_iter().enumerate() {
        let diff = row - &c_row.to_owned();
        let d2 = diff.dot(&diff);
        if d2 < best_d2 {
            best_d2 = d2;
            best = j;
        }
    }
    best
}

// helper: maximum centroid movement
fn max_centroid_move(prev: &Array2<f64>, next: &Array2<f64>) -> f64 {
    let mut max_move = 0f64;
    for (p_row, n_row) in prev.rows().into_iter().zip(next.rows()) {
        let diff = &p_row - &n_row;
        let d = diff.dot(&diff).sqrt();
        if d > max_move {
            max_move = d;
        }
    }
    max_move
}

