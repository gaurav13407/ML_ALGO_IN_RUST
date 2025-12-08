// src/utils/encoding.rs
use std::collections::HashMap;
use ndarray::{Array2, s};

#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    pub categories: Vec<String>,
    index_map: HashMap<String, usize>,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            index_map: HashMap::new(),
        }
    }

    /// Fit the encoder on a column of string values
    pub fn fit(&mut self, col: &[String]) {
        self.categories.clear();
        self.index_map.clear();
        for val in col.iter() {
            if !self.index_map.contains_key(val) {
                let idx = self.categories.len();
                self.categories.push(val.clone());
                self.index_map.insert(val.clone(), idx);
            }
        }
    }

    /// Transform: returns (n_samples, n_categories) dense matrix of 0.0/1.0
    /// Unknown categories produce a zero-vector row (handle_unknown="ignore")
    pub fn transform(&self, col: &[String]) -> Array2<f64> {
        let n = col.len();
        let k = self.categories.len();
        if k == 0 {
            // no known categories -> return zeros shape (n, 0)
            return Array2::<f64>::zeros((n, 0));
        }
        let mut out = Array2::<f64>::zeros((n, k));
        for (i, val) in col.iter().enumerate() {
            if let Some(&idx) = self.index_map.get(val) {
                out[[i, idx]] = 1.0;
            }
            // unknown -> remain zero
        }
        out
    }

    /// Convenience: fit + transform
    pub fn fit_transform(col: &[String]) -> Array2<f64> {
        let mut enc = OneHotEncoder::new();
        enc.fit(col);
        enc.transform(col)
    }
}

#[derive(Debug, Clone)]
pub struct OneHotEncoderMulti {
    pub encoders: Vec<OneHotEncoder>,
}

impl OneHotEncoderMulti {
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
        }
    }

    /// Fit on many columns. `cols` is Vec of columns; each column is Vec<String> of length n_samples.
    pub fn fit(&mut self, cols: &Vec<Vec<String>>) {
        self.encoders.clear();
        for col in cols.iter() {
            let mut enc = OneHotEncoder::new();
            enc.fit(col);
            self.encoders.push(enc);
        }
    }

    /// Transform many columns and horizontally concatenate encoded outputs.
    /// Returns (n_samples, total_categories)
    pub fn transform(&self, cols: &Vec<Vec<String>>) -> Array2<f64> {
        assert_eq!(
            cols.len(),
            self.encoders.len(),
            "cols and encoders length must match"
        );

        if cols.is_empty() {
            return Array2::<f64>::zeros((0, 0));
        }

        let n = cols[0].len();
        // total categories (sum of categories of each encoder)
        let total_k: usize = self.encoders.iter().map(|e| e.categories.len()).sum();

        let mut out = Array2::<f64>::zeros((n, total_k));
        let mut start = 0usize;
        for (enc, col) in self.encoders.iter().zip(cols.iter()) {
            let encoded = enc.transform(col); // (n, k_i)
            let k = encoded.ncols();
            if k == 0 {
                // nothing to assign for this column
                continue;
            }
            out.slice_mut(s![.., start..start + k]).assign(&encoded);
            start += k;
        }

        out
    }

    /// Fit + transform convenience
    pub fn fit_transform(cols: &Vec<Vec<String>>) -> Array2<f64> {
        let mut multi = OneHotEncoderMulti::new();
        multi.fit(cols);
        multi.transform(cols)
    }
}

