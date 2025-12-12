use calamine::{open_workbook, Data, Reader, Xlsx};
use csv::ReaderBuilder;
use csv::Writer;
use ndarray::{Array1, Array2};
use std::{error::Error, fs::File, result};
use crate::PCA::PCA;

/// Load Excel file (.xlsx) by column name
pub fn load_excel_by_name(
    path: &str,
    target_column_name: &str,
    sheet_name: Option<&str>,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut workbook: Xlsx<_> = open_workbook(path)?;

    // Get the specified sheet or the first one
    let sheet_name = match sheet_name {
        Some(name) => name.to_string(),
        None => workbook
            .sheet_names()
            .first()
            .ok_or("No sheets found in workbook")?
            .clone(),
    };

    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|_| format!("Sheet '{}' not found", sheet_name))?;

    // Get headers from first row
    let first_row = range.rows().next().ok_or("Excel file is empty")?;

    let headers: Vec<String> = first_row
        .iter()
        .map(|cell| cell.to_string().trim().to_string())
        .collect();

    // Find target column index
    let target_index = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case(target_column_name.trim()))
        .ok_or_else(|| {
            format!(
                "Column '{}' not found. Available: {:?}",
                target_column_name, headers
            )
        })?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    // Process data rows (skip header)
    for (_row_num, row) in range.rows().skip(1).enumerate() {
        let mut row_vals: Vec<f64> = Vec::new();

        for cell in row.iter() {
            let val = match cell {
                Data::Int(i) => *i as f64,
                Data::Float(f) => *f,
                Data::String(s) => {
                    let trimmed = s.trim();
                    if trimmed.is_empty() || trimmed == "?" || trimmed.eq_ignore_ascii_case("nan") {
                        f64::NAN
                    } else {
                        trimmed.parse::<f64>().unwrap_or(f64::NAN)
                    }
                }
                Data::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Data::Empty => f64::NAN,
                _ => f64::NAN,
            };
            row_vals.push(val);
        }

        if row_vals.len() <= target_index {
            continue; // Skip incomplete rows
        }

        let y_val = row_vals[target_index];
        row_vals.remove(target_index);

        targets.push(y_val);
        features.push(row_vals);
    }

    if features.is_empty() {
        return Err("No data rows found in Excel file".into());
    }

    let n_sample = features.len();
    let n_features = features[0].len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();

    let x = Array2::from_shape_vec((n_sample, n_features), flat)?;
    let y = Array1::from_vec(targets);

    Ok((x, y))
}

/// Load Excel file (.xlsx) by column index
pub fn load_excel(
    path: &str,
    target_column: usize,
    sheet_name: Option<&str>,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut workbook: Xlsx<_> = open_workbook(path)?;

    let sheet_name = match sheet_name {
        Some(name) => name.to_string(),
        None => workbook
            .sheet_names()
            .first()
            .ok_or("No sheets found in workbook")?
            .clone(),
    };

    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|_| format!("Sheet '{}' not found", sheet_name))?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    // Process all rows (skip header if present)
    let mut is_first_row = true;
    for row in range.rows() {
        if is_first_row {
            is_first_row = false;
            continue; // Skip header
        }

        let mut row_vals: Vec<f64> = Vec::new();

        for cell in row.iter() {
            let val = match cell {
                Data::Int(i) => *i as f64,
                Data::Float(f) => *f,
                Data::String(s) => {
                    let trimmed = s.trim();
                    if trimmed.is_empty() || trimmed == "?" || trimmed.eq_ignore_ascii_case("nan") {
                        f64::NAN
                    } else {
                        trimmed.parse::<f64>().unwrap_or(f64::NAN)
                    }
                }
                Data::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Data::Empty => f64::NAN,
                _ => f64::NAN,
            };
            row_vals.push(val);
        }

        if row_vals.len() <= target_column {
            continue;
        }

        let y_val = row_vals[target_column];
        row_vals.remove(target_column);

        targets.push(y_val);
        features.push(row_vals);
    }

    if features.is_empty() {
        return Err("No data rows found in Excel file".into());
    }

    let n_sample = features.len();
    let n_features = features[0].len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();

    let x = Array2::from_shape_vec((n_sample, n_features), flat)?;
    let y = Array1::from_vec(targets);

    Ok((x, y))
}

/// Load CSV by column name (string)
pub fn load_csv_by_name(
    path: &str,
    target_column_name: &str,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    // Read headers to find column index
    let headers = rdr.headers()?.clone();
    let target_index = headers
        .iter()
        .position(|h| h.trim().eq_ignore_ascii_case(target_column_name.trim()))
        .ok_or_else(|| {
            format!(
                "Column '{}' not found in headers. Available columns: {:?}",
                target_column_name, headers
            )
        })?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for (line_num, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                return Err(format!("CSV parser error at record {}: {}", line_num + 1, e).into());
            }
        };

        let mut row_vals: Vec<f64> = Vec::new();

        for field in record.iter() {
            let trimmed = field.trim();
            let val = if trimmed.is_empty() {
                f64::NAN
            } else {
                match trimmed.parse::<f64>() {
                    Ok(v) => v,
                    Err(_) => f64::NAN,
                }
            };
            row_vals.push(val);
        }

        if target_index >= row_vals.len() {
            return Err(format!(
                "Target column index {} out of range at line {}",
                target_index, line_num
            )
            .into());
        }
        let y_val = row_vals[target_index];
        row_vals.remove(target_index);

        targets.push(y_val);
        features.push(row_vals);
    }

    if features.is_empty() {
        return Err("No data rows found in CSV".into());
    }
    let n_sample = features.len();
    let n_features = features[0].len();

    let flat: Vec<f64> = features.into_iter().flatten().collect();

    let x = Array2::from_shape_vec((n_sample, n_features), flat)?;
    let y = Array1::from_vec(targets);

    Ok((x, y))
}

/// Load CSV by column index (backward compatibility)
pub fn load_csv(
    path: &str,
    target_column: usize,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for (line_num, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                return Err(format!("CSV parser error at record {}: {}", line_num + 1, e).into());
            }
        };

        let mut row_vals: Vec<f64> = Vec::new();

        for field in record.iter() {
            let trimmed = field.trim();
            let val = if trimmed.is_empty() {
                f64::NAN
            } else {
                match trimmed.parse::<f64>() {
                    Ok(v) => v,
                    Err(_) => f64::NAN,
                }
            };
            row_vals.push(val);
        }

        if target_column >= row_vals.len() {
            return Err(format!(
                "Target column {} out of range at line {}",
                target_column, line_num
            )
            .into());
        }
        let y_val = row_vals[target_column];
        row_vals.remove(target_column);

        targets.push(y_val);
        features.push(row_vals);
    }
    if features.is_empty() {
        return Err("No data rows found in CSV".into());
    }
    let n_sample = features.len();
    let n_features = features[0].len();

    let flat: Vec<f64> = features.into_iter().flatten().collect();

    let x = Array2::from_shape_vec((n_sample, n_features), flat)?;
    let y = Array1::from_vec(targets);

    Ok((x, y))
}

pub struct CsvData {
    pub data: Vec<Vec<String>>,
    pub headers: Vec<String>,
}

pub fn load_csv_raw(path: &str) -> Result<CsvData, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

    // Read headers
    let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();

    let mut rows: Vec<Vec<String>> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        rows.push(record.iter().map(|s| s.to_string()).collect());
    }

    if rows.is_empty() {
        return Err("CSV contains no rows".into());
    }

    Ok(CsvData {
        data: rows,
        headers,
    })
}

/// Detect categorical columns in CsvData.
/// A column is categorical if ANY value fails to parse as f64.
pub fn detect_categorical_columns(data: &Vec<Vec<String>>) -> Vec<bool> {
    if data.is_empty() {
        return Vec::new();
    }

    let n_cols = data[0].len();
    let n_rows = data.len();

    let mut is_cat = vec![false; n_cols];

    for col in 0..n_cols {
        for row in 0..n_rows {
            let cell = &data[row][col];
            // Try to parse as f64 — if fail → categorical
            if cell.parse::<f64>().is_err() {
                is_cat[col] = true;
                break;
            }
        }
    }

    is_cat
}

pub fn save_array2_to_csv(path: &str, arr: &Array2<f64>, headers: Option<Vec<String>>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;

    if let Some(h) = headers {
        wtr.write_record(h)?;
    }

    for row in arr.outer_iter() {
        let row_vals: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        wtr.write_record(row_vals)?;
    }

    wtr.flush()?;
    Ok(())
}



pub fn export_components_csv(pca: &PCA, feature_names: &[String]) -> Result<(), Box<dyn Error>> {
    let comps = pca.components.as_ref().unwrap(); // (k, d)
    let (n_components, n_features) = comps.dim();
    
    // Only use feature names up to n_features (in case there are more names than features)
    let mut headers = vec!["Component".to_string()];
    headers.extend(feature_names.iter().take(n_features).cloned());
    
    let mut wtr = Writer::from_path("rust_pca_components.csv")?;
    wtr.write_record(&headers)?;
    
    for (i, row) in comps.outer_iter().enumerate() {
        let mut rec = vec![format!("PC{}", i + 1)];
        rec.extend(row.iter().map(|v| v.to_string()));
        wtr.write_record(rec)?;
    }
    
    wtr.flush()?;
    Ok(())
}

pub fn export_explained_csv(pca: &PCA, x: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    let explained = pca.explained_variance.as_ref().unwrap();
    let ratio = pca.explained_variance_ratio(x);
    
    let mut cumulative = 0.0;
    let mut wtr = Writer::from_path("rust_pca_explained.csv")?;
    
    wtr.write_record(&["Component", "ExplainedVariance", "ExplainedVarianceRatio", "CumulativeVariance"])?;

    for i in 0..explained.len() {
        cumulative += ratio[i];
        wtr.write_record(&[
            format!("PC{}", i + 1),
            explained[i].to_string(),
            ratio[i].to_string(),
            cumulative.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

pub fn export_reduced_csv(path: &str, reduced: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;
    
    // Write header
    let n_components = reduced.ncols();
    let headers: Vec<String> = (1..=n_components).map(|i| format!("PC{}", i)).collect();
    wtr.write_record(&headers)?;
    
    // Write data rows
    for row in reduced.outer_iter() {
        let row_vals: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        wtr.write_record(row_vals)?;
    }
    
    wtr.flush()?;
    Ok(())
}
