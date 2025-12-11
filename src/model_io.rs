use std::error::Error;
use std::fs::File;
use std::path::Path;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SaveModel {
    coef: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
}

pub fn save_model<P: AsRef<Path>>(
    path: P,
    coef: &Array1<f64>,
    mean: &Array1<f64>,
    std: &Array1<f64>,
) -> Result<(), Box<dyn Error>> {
    let saved = SaveModel {
        coef: coef.to_vec(),
        mean: mean.to_vec(),
        std: std.to_vec(),
    };

    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &saved)?;
    Ok(())
}

pub fn load_model<P: AsRef<Path>>(
    path: P,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let saved: SaveModel = serde_json::from_reader(file)?;

    Ok((
        Array1::from(saved.coef),
        Array1::from(saved.mean),
        Array1::from(saved.std),
    ))
}
