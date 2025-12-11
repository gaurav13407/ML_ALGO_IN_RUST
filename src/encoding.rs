// src/utils/encoding.rs
use crate::data;
use ndarray::{s, Array2};
use std::collections::HashMap;
use std::error::Error;

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

    /// Get feature names for encoded columns
    pub fn get_feature_names(&self, prefix: &str) -> Vec<String> {
        self.categories
            .iter()
            .map(|cat| format!("{}__{}", prefix, cat))
            .collect()
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

fn encode_categorical_and_build_matrix(
    path: &str,
) -> Result<(Array2<f64>, Vec<String>), Box<dyn Error>> {
    // load CSV raw
    let csv = data::load_csv_raw(path)?;

    // detect categorical columns (Vec<bool>) or you may already have cat_indices
    let cat_mask = data::detect_categorical_columns(&csv.data);
    let cat_indices: Vec<usize> = cat_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_cat)| if is_cat { Some(i) } else { None })
        .collect();

    let n_samples = csv.data.len();
    let n_cols = csv.headers.len();

    // extract categorical columns as Vec<Vec<String>> in the order of cat_indices
    let mut cat_cols: Vec<Vec<String>> = Vec::with_capacity(cat_indices.len());
    for &ci in &cat_indices {
        let col: Vec<String> = (0..n_samples).map(|r| csv.data[r][ci].clone()).collect();
        cat_cols.push(col);
    }

    // If there are categorical columns, fit_transform them
    let encoded_block: Option<Array2<f64>>;
    let encoded_feature_names: Vec<Vec<String>>; // per categorical col feature names
    if !cat_cols.is_empty() {
        // fit and transform
        let enc = {
            let mut e = OneHotEncoderMulti::new();
            e.fit(&cat_cols);
            e
        };
        let encoded = enc.transform(&cat_cols); // shape (n_samples, total_ohe_features)
                                                // build feature names for each categorical column
        encoded_feature_names = {
            let mut names: Vec<Vec<String>> = Vec::new();
            for (idx, &ci) in cat_indices.iter().enumerate() {
                let header = &csv.headers[ci];
                let enc_ref = &enc.encoders[idx];
                let names_i = enc_ref.get_feature_names(header);
                names.push(names_i);
            }
            names
        };
        encoded_block = Some(encoded);
    } else {
        encoded_block = None;
        encoded_feature_names = Vec::new();
    }

    // Build numeric columns (all non-categorical columns), but we need to preserve order later.
    // We'll create a Vec<Option<Array2<f64>>> of length n_cols where categorical slots will be None
    // and numeric slots will contain (n_samples, 1) column arrays.
    let mut per_col_arrays: Vec<Option<Array2<f64>>> = vec![None; n_cols];

    for c in 0..n_cols {
        if cat_mask[c] {
            // placeholder: will be filled by OHE block later
            per_col_arrays[c] = None;
        } else {
            // collect column values and parse to f64 (your other module might already do this,
            // but here we parse directly)
            let mut col_vals = Vec::with_capacity(n_samples);
            for r in 0..n_samples {
                let s = csv.data[r][c].trim();
                // if parse fails, we default to 0.0; you may want to handle errors differently
                let v = s.parse::<f64>().unwrap_or(0.0);
                col_vals.push(v);
            }
            // create (n_samples, 1) array
            let arr = Array2::from_shape_vec((n_samples, 1), col_vals)?;
            per_col_arrays[c] = Some(arr);
        }
    }

    // Now assemble final matrix by walking original columns left->right.
    // For categorical columns, insert the corresponding block from encoded_block (slice).
    // For numeric columns, insert the (n_samples,1) array.
    // First compute final number of features
    let mut total_features = 0usize;
    let mut encoded_col_start_indices: Vec<usize> = Vec::new(); // for mapping blocks
    let mut running = 0usize;
    if let Some(ref enc_block) = encoded_block {
        // we need to know how many OHE columns each cat column has
        for enc_names in &encoded_feature_names {
            encoded_col_start_indices.push(running);
            running += enc_names.len();
        }
    }
    // compute total features: sum numeric + encoded
    for c in 0..n_cols {
        if cat_mask[c] {
            // find index of this categorical column in cat_indices to get its OHE width
            let idx_in_cat = cat_indices.iter().position(|&x| x == c).unwrap();
            let ohe_width = encoded_feature_names[idx_in_cat].len();
            total_features += ohe_width;
        } else {
            total_features += 1;
        }
    }

    // create final matrix (n_samples, total_features)
    let mut final_mat = Array2::<f64>::zeros((n_samples, total_features));
    let mut final_feature_names: Vec<String> = Vec::with_capacity(total_features);

    let mut write_col = 0usize;
    for c in 0..n_cols {
        if cat_mask[c] {
            // categorical: take its block from encoded_block
            let idx_in_cat = cat_indices.iter().position(|&x| x == c).unwrap();
            let start = encoded_col_start_indices[idx_in_cat];
            let width = encoded_feature_names[idx_in_cat].len();
            if width > 0 {
                if let Some(ref enc_block) = encoded_block {
                    // slice encoded_block[.., start..start+width] into final_mat[.., write_col..]
                    final_mat
                        .slice_mut(s![.., write_col..write_col + width])
                        .assign(&enc_block.slice(s![.., start..start + width]));
                    // push feature names
                    for n in &encoded_feature_names[idx_in_cat] {
                        final_feature_names.push(n.clone());
                    }
                    write_col += width;
                }
            } else {
                // no categories (shouldn't happen) — skip
            }
        } else {
            // numeric: copy single column
            if let Some(ref num_col) = per_col_arrays[c] {
                final_mat
                    .slice_mut(s![.., write_col..write_col + 1])
                    .assign(&num_col.view());
                final_feature_names.push(csv.headers[c].clone());
                write_col += 1;
            }
        }
    }

    Ok((final_mat, final_feature_names))
}
