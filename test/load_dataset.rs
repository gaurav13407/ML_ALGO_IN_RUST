use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use ndarray::{s, Array1, Array2};
use std::collections::{HashMap, HashSet};

use ML_ALGO_Rewite::{data, encoding, load_data, metrics, Kmeans};
fn main() -> Result<(), Box<dyn Error>> {
    let path = "examples/KMeans_1.csv";

    println!("=== Testing Dataset Loader ===");

    // ---------- 1️⃣ LOAD FULL DATASET (NO TARGET) ----------
    let ds = load_data::DatasetLoader::from_path(path).build()?; // no target set → X only

    println!("Test 1: Full encoded dataset (no target)");
    println!("X shape = {:?}", ds.x.dim());
    println!("Feature names = {:?}", ds.feature_names);
    println!("Y = {:?}", ds.y); // should be None
    println!("-------------------------------------------------\n");

    // ---------- 2️⃣ LOAD WITH TARGET COLUMN NAME ----------
    let ds2 = load_data::DatasetLoader::from_path(path)
        .with_target("gdpp") // <-- change to your real column name
        .build()?;

    println!("Test 2: Target by name");
    println!("X shape = {:?}", ds2.x.dim());
    println!("Y length = {:?}", ds2.y.as_ref().unwrap().len());
    println!(
        "First 10 labels = {:?}",
        ds2.y.as_ref().unwrap().slice(s![0..10])
    );
    println!("Label map = {:?}", ds2.label_map);
    println!("-------------------------------------------------\n");

    // ---------- 3️⃣ LOAD WITH TARGET COLUMN INDEX ----------
    let ds3 = load_data::DatasetLoader::from_path(path)
        .with_target_index(0) // column 0 as target
        .build()?;

    println!("Test 3: Target by index");
    println!("X shape = {:?}", ds3.x.dim());
    println!("Y length = {:?}", ds3.y.as_ref().unwrap().len());
    println!("Label map = {:?}", ds3.label_map);
    println!("-------------------------------------------------\n");

    println!("All dataset tests completed successfully!");

    Ok(())
}
