use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use ndarray::{Array1, Array2, s};
use std::collections::{HashMap, HashSet};

use ML_ALGO_Rewite::{data, metrics, Kmeans, encoding,load_data};
fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_1.csv";

    // ----------- LOAD DATASET WITH TARGET ----------
    
    // Change the target column name based on your CSV
    let ds = load_data::DatasetLoader::from_path(path)
        .with_target("gdpp")        // or .with_target_index(0)
        .build()?;

    // Extract X and y
    let x = ds.x;              // Array2<f64>
    let y = ds.y.unwrap();     // Array1<f64>

    // ----------- PRINT BASIC INFO ----------
    println!("=== Loaded Dataset ===");
    println!("X shape = {:?}", x.dim());
    println!("y shape = {:?}", y.len());
    println!("First 5 rows of X:\n{:?}", x.slice(s![0..5, ..]));
    println!("First 10 y: {:?}", y.slice(s![0..10]));

    // Show label mapping for categorical target
    if let Some(label_map) = ds.label_map {
        println!("\nLabel mapping (index → class):");
        for (i, v) in label_map.iter().enumerate() {
            println!("{i} => {v}");
        }
    }

    Ok(())
}
