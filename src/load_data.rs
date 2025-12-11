// src/dataset.rs
use ndarray::{s, Array1, Array2};
use std::error::Error;

use crate::data;
use crate::encoding;

/// Final dataset structure
pub struct Dataset {
    pub x: Array2<f64>,
    pub y: Option<Array1<f64>>,
    pub feature_names: Vec<String>,
    pub label_map: Option<Vec<String>>, // Only when target is categorical
}

/// Loader builder
pub struct DatasetLoader {
    path: String,
    target_name: Option<String>,
    target_index: Option<usize>,
}

impl DatasetLoader {
    pub fn from_path(path: &str) -> Self {
        Self {
            path: path.to_string(),
            target_name: None,
            target_index: None,
        }
    }

    pub fn with_target(mut self, name: &str) -> Self {
        self.target_name = Some(name.to_string());
        self
    }

    pub fn with_target_index(mut self, idx: usize) -> Self {
        self.target_index = Some(idx);
        self
    }

    fn is_excel(&self) -> bool {
        self.path.ends_with(".xlsx") || self.path.ends_with(".xls")
    }

    fn load_raw(&self) -> Result<data::CsvData, Box<dyn Error>> {
        if self.is_excel() {
            return Err("Excel raw loader is not implemented yet for raw string loading.".into());
        }
        data::load_csv_raw(&self.path)
    }

    pub fn build(self) -> Result<Dataset, Box<dyn Error>> {
        // ================================================
        // 1) LOAD RAW STRING DATA
        // ================================================
        let raw = self.load_raw()?;
        let n_samples = raw.data.len();
        let n_cols = raw.headers.len();

        // ================================================
        // 2) DETECT CATEGORICAL COLUMNS
        // ================================================
        let cat_mask = data::detect_categorical_columns(&raw.data);

        // ================================================
        // 3) DETERMINE TARGET COLUMN INDEX
        // ================================================
        let target_idx = if let Some(name) = &self.target_name {
            raw.headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or("Target column not found")?
        } else if let Some(idx) = self.target_index {
            if idx >= n_cols {
                return Err("Target column index out of range".into());
            }
            idx
        } else {
            usize::MAX // Means "no target"
        };

        // ================================================
        // 4) EXTRACT TARGET COLUMN BEFORE ENCODING
        // ================================================
        let (y, label_map) = if target_idx != usize::MAX {
            let mut lookup: Vec<String> = Vec::new();
            let mut y_vals: Vec<f64> = Vec::with_capacity(n_samples);

            for r in 0..n_samples {
                let v = raw.data[r][target_idx].clone();

                if !lookup.contains(&v) {
                    lookup.push(v.clone());
                }
                let idx = lookup.iter().position(|a| a == &v).unwrap();
                y_vals.push(idx as f64);
            }

            (Some(Array1::from(y_vals)), Some(lookup))
        } else {
            (None, None)
        };

        // ================================================
        // 5) PREPARE CATEGORICAL + NUMERIC COLUMNS EXCEPT TARGET
        // ================================================

        // Collect categorical columns (excluding target)
        let mut cat_cols: Vec<Vec<String>> = Vec::new();
        let mut cat_col_indices: Vec<usize> = Vec::new();

        for c in 0..n_cols {
            if c == target_idx {
                continue;
            }
            if cat_mask[c] {
                cat_col_indices.push(c);
                let col = (0..n_samples).map(|r| raw.data[r][c].clone()).collect();
                cat_cols.push(col);
            }
        }

        // Collect numeric columns (excluding target)
        let mut num_arrays: Vec<Array2<f64>> = Vec::new();
        let mut num_names: Vec<String> = Vec::new();

        for c in 0..n_cols {
            if c == target_idx {
                continue;
            }
            if !cat_mask[c] {
                let mut vals = Vec::with_capacity(n_samples);
                for r in 0..n_samples {
                    vals.push(raw.data[r][c].parse::<f64>().unwrap_or(0.0));
                }
                num_arrays.push(Array2::from_shape_vec((n_samples, 1), vals)?);
                num_names.push(raw.headers[c].clone());
            }
        }

        // ================================================
        // 6) ONE-HOT ENCODE CATEGORICAL COLUMNS
        // ================================================
        let (cat_encoded, cat_feature_names) = if !cat_cols.is_empty() {
            let mut enc = encoding::OneHotEncoderMulti::new();
            enc.fit(&cat_cols);
            let block = enc.transform(&cat_cols);

            let mut names = Vec::new();
            for (i, &ci) in cat_col_indices.iter().enumerate() {
                let prefix = &raw.headers[ci];
                for cat in &enc.encoders[i].categories {
                    names.push(format!("{}__{}", prefix, cat));
                }
            }

            (Some(block), names)
        } else {
            (None, Vec::new())
        };

        // ================================================
        // 7) BUILD FINAL X MATRIX IN ORIGINAL ORDER
        // ================================================
        let total_features = num_names.len() + cat_feature_names.len();
        let mut x = Array2::<f64>::zeros((n_samples, total_features));
        let mut feature_names: Vec<String> = Vec::with_capacity(total_features);

        let mut write_pos = 0usize;
        let mut cat_cursor = 0usize;

        for c in 0..n_cols {
            if c == target_idx {
                continue; // Skip target from X
            }

            if !cat_mask[c] {
                // numeric column
                let idx = num_names.iter().position(|h| h == &raw.headers[c]).unwrap();
                x.slice_mut(s![.., write_pos..write_pos + 1])
                    .assign(&num_arrays[idx]);
                feature_names.push(raw.headers[c].clone());
                write_pos += 1;
            } else {
                // categorical column
                if let Some(ref block) = cat_encoded {
                    // count how many OHE columns belong to this categorical column
                    let enc = &cat_feature_names[cat_cursor..];

                    // detect next column prefix
                    let prefix = &raw.headers[c];
                    let cols_for_this_cat =
                        enc.iter().take_while(|n| n.starts_with(prefix)).count();

                    let start = cat_cursor;
                    let end = start + cols_for_this_cat;

                    x.slice_mut(s![.., write_pos..write_pos + cols_for_this_cat])
                        .assign(&block.slice(s![.., start..end]));

                    feature_names.extend_from_slice(&cat_feature_names[start..end]);

                    write_pos += cols_for_this_cat;
                    cat_cursor += cols_for_this_cat;
                }
            }
        }

        Ok(Dataset {
            x,
            y,
            feature_names,
            label_map,
        })
    }
}
