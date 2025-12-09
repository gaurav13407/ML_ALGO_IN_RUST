use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use ndarray::{Array1, Array2, s};
use std::collections::{HashMap, HashSet};

use ML_ALGO_Rewite::{data, metrics, Kmeans, encoding,load_data};

fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_1.csv";
         println!("=== Testing Dataset Loader ===");

    // ---------- 1️⃣ LOAD FULL DATASET (NO TARGET) ----------
    let ds = load_data::DatasetLoader::from_path(path)
        .with_target("gdpp")
        .build()?;   // no target set → X only
     let x =&ds.x;           // Array2<f64>
     let y = ds.y.as_ref().unwrap();

    println!("Test 1: Full encoded dataset (no target)");
    println!("X shape = {:?}", ds.x.dim());
    println!("Feature names = {:?}", ds.feature_names);
    println!("Y = {:?}", ds.y);  // should be None
    println!("-------------------------------------------------\n");


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

