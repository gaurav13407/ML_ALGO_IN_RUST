# ML Algorithms from Scratch in Rust

A high-performance machine learning library implemented from scratch in Rust, featuring supervised learning (linear/logistic regression), unsupervised learning (K-Means clustering), and intelligent data preprocessing with automatic categorical feature handling.

## 🌟 What's New

- **🤖 K-Means Clustering** - Unsupervised learning with K-Means++ initialization
- **🧠 Smart Dataset Loader** - Automatically detects and encodes categorical features
- **📊 Multiple Datasets** - 15+ example datasets included for testing
- **🐍 Python Comparisons** - Side-by-side Python scripts for benchmarking
- **📈 Comprehensive Benchmarks** - Performance results for all algorithms in `results/`

## 🚀 Features

### Implemented Algorithms
- **Linear Regression** - Ordinary Least Squares (OLS) using normal equation
- **Logistic Regression** - Binary classification with gradient descent and L2 regularization
- **K-Means Clustering** - Unsupervised clustering with K-Means++ initialization
- **PCA (Principal Component Analysis)** - SVD-based dimensionality reduction
- **Standard Scaler** - Feature normalization (z-score standardization)
- **Missing Value Imputation** - Median-based imputation for handling NaN values
- **Train-Test Split** - Stratified data splitting with optional shuffling
- **One-Hot Encoding** - Categorical variable encoding for single and multiple columns

### Key Capabilities
- ✅ CSV and Excel (.xlsx) data loading with automatic missing value handling (NaN)
- ✅ **Smart Dataset Loader** - Automatic categorical detection and one-hot encoding
- ✅ Feature scaling and normalization
- ✅ One-hot encoding for categorical features
- ✅ L2 regularization support for logistic regression
- ✅ Model serialization (save/load to JSON)
- ✅ High-performance matrix operations using `ndarray` and `nalgebra`
- ✅ Comprehensive evaluation metrics:
  - **Regression**: MSE, RMSE, R²
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  - **Clustering**: Inertia, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

## 📋 Requirements

- Rust 1.70+ (2021 edition)
- Cargo (comes with Rust)

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/gaurav13408/ML_ALGO_IN_RUST.git
cd ML_ALGO_Rewite
```

2. **Build the project:**
```bash
cargo build --release
```

## 📊 Quick Start

### Running Examples

#### Linear Regression Example

The project includes a complete example using California housing data:

```bash
cargo run --bin linear_regression
```

**Expected Output:**
```
Loaded X Shaped:(20640, 8), y.len 20640
Train Rows:16512, Test rows:4128
Trained. Train time: ~25 ms
MSE: 4786221958.573119
R² : 0.646525
Predict time: ~1.2 ms
Saved model -> models/linear_model_rust.json
```

#### Logistic Regression Example

Binary classification with L2 regularization:

```bash
cargo run --bin logsitic_regression
```

**Expected Output:**
```
Loaded X shape: (768, 8), y.len: 768
Data sets — Train rows: 614, Test rows: 154
epoch 100/1000  log_loss = 0.xxxxx
...
Results:
ROC-AUC: 0.8xxx
Accuracy: 0.7xxx
Precision: 0.7xxx
Recall: 0.6xxx
F1: 0.6xxx
Confusion matrix:
[[tn, fp],
 [fn, tp]]
Train time: ~XXXms, Predict time: ~XXms
Saved weights to models/logistic_weights.csv
```

#### K-Means Clustering Example

Unsupervised clustering with automatic categorical encoding:

```bash
cargo run --bin Kmeans
```

**Expected Output:**
```
Fit time         = 24.0567ms
Predict time     = 3.1685ms
Per-sample pred  ≈ 0.000018973 seconds
First 10 labels (fit):     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
First 10 labels (predict): [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
Inertia                 = 36528388099.32208
Silhouette score        = 0.7256314858790945
Davies–Bouldin index    = 0.5190307948651756
Calinski–Harabasz score = 365.5694958970166
```

**Metrics Explained:**
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Silhouette Score**: [-1, 1], higher is better (measures cluster separation)
- **Davies–Bouldin Index**: Lower is better (measures cluster similarity)
- **Calinski–Harabasz Score**: Higher is better (ratio of between/within cluster dispersion)

#### Dataset Loader Examples

Test the smart dataset loader with automatic categorical detection:

```bash
# Load dataset without target (X only)
cargo run --bin load_data

# Load dataset with target (X, y)
cargo run --bin load_data_1
```

### Using as a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
ML_ALGO_Rewite = { path = "../path/to/ML_ALGO_Rewite" }
ndarray = "0.15"
```

#### Example 1: Traditional Workflow (Manual Feature Engineering)
```rust
use ML_ALGO_Rewite::{data, preprocess, split, linear};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from CSV or Excel
    let (x, y) = data::load_csv_by_name("data.csv", "target")?;
    // or from Excel:
    // let (x, y) = data::load_excel_by_name("data.xlsx", "target", None)?;
    
    // Split data
    let (x_train, x_test, y_train, y_test) = 
        split::train_test_split(&x, &y, 0.2, true, Some(42))?;
    
    // Handle missing values
    let medians = preprocess::column_median(&x_train);
    let x_train = preprocess::impute_median(&x_train, &medians);
    let x_test = preprocess::impute_median(&x_test, &medians);
    
    // Scale features
    let scaler = preprocess::StandardScaler::fit(&x_train);
    let x_train_scaled = scaler.transform(&x_train);
    let x_test_scaled = scaler.transform(&x_test);
    
    // Add bias term
    let xb_train = preprocess::add_bias(&x_train_scaled);
    let xb_test = preprocess::add_bias(&x_test_scaled);
    
    // Train model
    let coef = linear::train_ols(&xb_train, &y_train)?;
    
    // Predict and evaluate
    let y_pred = linear::predict(&xb_test, &coef);
    let mse = linear::mse(&y_test, &y_pred);
    let r2 = linear::r2(&y_test, &y_pred);
    
    println!("MSE: {:.6}, R²: {:.6}", mse, r2);
    Ok(())
}
```

#### Example 2: Smart Dataset Loader (Automatic Feature Engineering)
```rust
use ML_ALGO_Rewite::{load_data, split, Kmeans};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dataset with automatic categorical detection and encoding
    let ds = load_data::DatasetLoader::from_path("data.csv")
        .with_target("species")  // categorical target auto-encoded
        .build()?;
    
    let x = ds.x;  // Fully encoded feature matrix (numeric + one-hot)
    let y = ds.y.unwrap();  // Encoded target (0.0, 1.0, 2.0, ...)
    
    // Check label mapping for categorical targets
    if let Some(label_map) = ds.label_map {
        println!("Label mapping: {:?}", label_map);
        // e.g., ["setosa", "versicolor", "virginica"]
    }
    
    println!("Feature names: {:?}", ds.feature_names);
    // e.g., ["sepal_length", "sepal_width", "color__red", "color__blue", ...]
    
    // Use the data directly with any algorithm
    let mut kmeans = Kmeans::KMeans::new(3)
        .max_iter(300)
        .rng_seed(42);
    
    let labels = kmeans.fit(&x.view());
    println!("Cluster assignments: {:?}", labels);
    println!("Inertia: {:?}", kmeans.inertia);
    
    Ok(())
}
```

## 📁 Project Structure

```
ML_ALGO_Rewite/
├── src/
│   ├── lib.rs           # Library entry point, exports public modules
│   ├── main.rs          # Main binary (logistic regression demo)
│   ├── data.rs          # CSV/Excel loading with NaN handling + raw data loader
│   ├── load_data.rs     # Smart dataset loader with auto categorical detection
│   ├── preprocess.rs    # Scaling, normalization, imputation
│   ├── split.rs         # Train-test splitting
│   ├── linear.rs        # Linear regression (OLS)
│   ├── logistic.rs      # Logistic regression with L2 regularization
│   ├── Kmeans.rs        # K-Means clustering with K-Means++ init
│   ├── metrics.rs       # Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
│   ├── encoding.rs      # One-hot encoding for categorical variables
│   └── model_io.rs      # Model serialization/deserialization
├── test/
│   ├── linear_regression.rs   # Linear regression example
│   ├── logistic_regression.rs # Logistic regression example
│   ├── KMeans.rs              # K-Means clustering example
│   ├── PCA.rs                 # PCA dimensionality reduction example
│   ├── load_dataset.rs        # Dataset loader test (no target)
│   ├── load_xy_dataset.rs     # Dataset loader test (with target)
│   ├── linear_regression.py   # Python comparison script
│   ├── logstic_regression.py  # Python comparison script
│   ├── KMeans.py              # Python comparison script
│   └── PCA.py                 # Python PCA comparison script
├── examples/
│   ├── Linear_regression_1.csv       # California housing dataset
│   ├── Linear_refression_2.csv
│   ├── Linear_refression_3.csv
│   ├── Linear_refression_4.csv
│   ├── Linear_refression_5.csv
│   ├── Linear_refression_6.xlsx      # Excel format dataset
│   ├── Logistic_regression_1.csv
│   ├── Logistic_regression_2.csv
│   ├── Logistic_regression_3.csv
│   ├── Logistic_regression_4.csv
│   ├── logistic_regression_5.csv     # Diabetes dataset
│   ├── KMeans_1.csv                  # Clustering dataset
│   ├── KMeans_2.csv
│   ├── KMeans_3.csv
│   ├── KMeans_4.csv
│   └── KMeans_5.csv
├── models/              # Saved model files (JSON, CSV weights)
│   ├── linear_model_rust.json
│   ├── logistic_weights.csv
│   └── linear_model.joblib
├── results/             # Benchmark results
│   ├── Linear_regression_Result.csv.xlsx
│   ├── Logsitic_regression_Result.xlsx
│   ├── KMeans_Result.xlsx
│   ├── pca_results.xlsx
│   ├── PCA_Comparision.xlsx
│   ├── pca_components.csv            # Python PCA components
│   ├── pca_explained.csv             # Python explained variance
│   ├── pca_reduced.csv               # Python reduced data
│   ├── rust_pca_components.csv       # Rust PCA components
│   ├── rust_pca_explained.csv        # Rust explained variance
│   └── rust_pca_reduced.csv          # Rust reduced data
├── Cargo.toml          # Project configuration
└── README.md           # This file
```

## 🧪 Modules

### `data` - Data Loading
```rust
// CSV Loading
pub fn load_csv(path: &str, target_column: usize) 
    -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>>

pub fn load_csv_by_name(path: &str, target_column_name: &str) 
    -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>>

// Excel Loading (.xlsx)
pub fn load_excel(path: &str, target_column: usize, sheet_name: Option<&str>) 
    -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>>

pub fn load_excel_by_name(path: &str, target_column_name: &str, sheet_name: Option<&str>) 
    -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>>
```
- Loads CSV and Excel (.xlsx) files
- Automatically handles missing values (converts to NaN)
- Separates features and target variable
- Column selection by index or name
- For Excel: specify sheet name or use first sheet by default

### `preprocess` - Data Preprocessing
```rust
pub struct StandardScaler { pub mean: Array1<f64>, pub std: Array1<f64> }
pub fn column_median(X: &Array2<f64>) -> Array1<f64>
pub fn impute_median(X: &Array2<f64>, medians: &Array1<f64>) -> Array2<f64>
pub fn add_bias(X: &Array2<f64>) -> Array2<f64>
```
- Feature scaling (z-score normalization)
- Median imputation for missing values
- Bias term addition for regression

### `split` - Data Splitting
```rust
pub fn train_test_split(
    X: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>>
```
- Train-test splitting with optional shuffling
- Reproducible splits with seed

### `linear` - Linear Regression
```rust
pub fn train_ols(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>>
pub fn predict(x: &Array2<f64>, coef: &Array1<f64>) -> Array1<f64>
pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64
pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64
pub fn r2(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64
```
- OLS training using normal equation with ridge fallback
- Comprehensive evaluation metrics

### `logistic` - Logistic Regression
```rust
pub struct LogisticRegression { pub w: Array1<f64> }
pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>, epochs: usize, lr: f64, l2: f64, verbose: bool, print_every: usize)
pub fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64>
pub fn predict(&self, X: &Array2<f64>) -> Array1<f64>
pub fn log_loss(probs: &Array1<f64>, y: &Array1<f64>) -> f64
```
- Binary classification with gradient descent
- **L2 regularization** support (ridge penalty) to prevent overfitting
- Numerically stable sigmoid implementation
- Log loss (binary cross-entropy) for model evaluation

### `metrics` - Evaluation Metrics
```rust
// Classification Metrics
pub fn confusion_counts(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> (usize, usize, usize, usize)
pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn confusion_matrix_array(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> Array2<usize>
pub fn roc_auc_score(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64

// Clustering Metrics
pub fn inertia(x: &ArrayView2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> f64
pub fn silhouette_score(x: &ArrayView2<f64>, labels: &Array1<usize>) -> Option<f64>
pub fn davies_bouldin_score(x: &ArrayView2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> Option<f64>
pub fn calinski_harabasz_score(x: &ArrayView2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> Option<f64>
```
**Classification Metrics:**
- **Confusion Matrix**: True/False Positives/Negatives
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve using rank-based method

**Clustering Metrics:**
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Silhouette Score**: Measures cluster separation, range [-1, 1] (higher is better)
- **Davies–Bouldin Index**: Measures cluster similarity (lower is better)
- **Calinski–Harabasz Score**: Ratio of between/within cluster dispersion (higher is better)

### `encoding` - Categorical Encoding
```rust
pub struct OneHotEncoder { pub categories: Vec<String>, index_map: HashMap<String, usize> }
pub fn fit(&mut self, col: &[String])
pub fn transform(&self, col: &[String]) -> Array2<f64>
pub fn fit_transform(col: &[String]) -> Array2<f64>

pub struct OneHotEncoderMulti { pub encoders: Vec<OneHotEncoder> }
pub fn fit(&mut self, cols: &Vec<Vec<String>>)
pub fn transform(&self, cols: &Vec<Vec<String>>) -> Array2<f64>
pub fn fit_transform(cols: &Vec<Vec<String>>) -> Array2<f64>
```
- **OneHotEncoder**: Convert categorical column to binary (0/1) matrix
- **OneHotEncoderMulti**: Encode multiple categorical columns and concatenate
- Handles unknown categories gracefully (ignore mode)

### `Kmeans` - K-Means Clustering
```rust
pub struct KMeans { k: usize, max_iter: usize, tol: f64, pub centroids: Option<Array2<f64>>, pub inertia: Option<f64> }
pub fn new(k: usize) -> Self
pub fn max_iter(self, it: usize) -> Self
pub fn tol(self, t: f64) -> Self
pub fn rng_seed(self, seed: u64) -> Self
pub fn fit(&mut self, x: &ArrayView2<f64>) -> Array1<usize>
pub fn predict(&self, x: &ArrayView2<f64>) -> Array1<usize>
pub fn centroids(&self) -> Option<&Array2<f64>>
```
- **K-Means++ Initialization**: Smart centroid initialization for faster convergence
- **Configurable**: Set max iterations, tolerance, and random seed
- **Inertia**: Measures within-cluster sum of squares
- **Predict**: Assign new samples to nearest cluster

### `load_data` - Smart Dataset Loader
```rust
pub struct Dataset { pub x: Array2<f64>, pub y: Option<Array1<f64>>, pub feature_names: Vec<String>, pub label_map: Option<Vec<String>> }

pub struct DatasetLoader
pub fn from_path(path: &str) -> Self
pub fn with_target(self, name: &str) -> Self
pub fn with_target_index(self, idx: usize) -> Self
pub fn build(self) -> Result<Dataset, Box<dyn Error>>
```
- **Automatic Categorical Detection**: Detects non-numeric columns automatically
- **One-Hot Encoding**: Automatically encodes categorical features
- **Target Handling**: Supports categorical targets with label mapping
- **Feature Names**: Preserves original and encoded feature names
- **Flexible API**: Load with or without target, by name or index

**Example Usage:**
```rust
// Load dataset with target column by name
let ds = load_data::DatasetLoader::from_path("data.csv")
    .with_target("price")
    .build()?;

let x = ds.x;  // Fully encoded feature matrix
let y = ds.y.unwrap();  // Target vector
println!("Features: {:?}", ds.feature_names);
```

### `model_io` - Model Persistence
```rust
pub fn save_model(path: &str, coef: &Array1<f64>, mean: &Array1<f64>, std: &Array1<f64>) -> Result<(), Box<dyn Error>>
pub fn load_model(path: &str) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>>
```
- Save/load models to JSON format

### `PCA` - Principal Component Analysis
```rust
pub struct PCA {
    pub n_components: usize,
    pub components: Option<Array2<f64>>,         // Principal components (n_components x n_features)
    pub explained_variance: Option<Array1<f64>>, // Variance per component
    pub mean: Option<Array1<f64>>,               // Feature means
    pub singular_values: Option<Array1<f64>>,    // Singular values
}

pub fn new(n_components: usize) -> Self
pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>>
pub fn transform(&self, x: &Array2<f64>) -> Array2<f64>
pub fn inverse_transform(&self, scores: &Array2<f64>) -> Array2<f64>
pub fn explained_variance_ratio(&self, x: &Array2<f64>) -> Array1<f64>
pub fn reconstruction_mse(&self, x: &Array2<f64>) -> f64
```

**Features:**
- **SVD-based**: Uses Singular Value Decomposition for numerical stability
- **Dimensionality Reduction**: Project high-dimensional data to lower dimensions
- **Variance Analysis**: Calculate explained variance per component and cumulative ratios
- **Inverse Transform**: Reconstruct original data from principal components
- **Reconstruction Error**: Measure information loss with MSE metric

**Example:**
```rust
use ML_ALGO_Rewite::PCA::PCA;

// Create PCA with 2 components
let mut pca = PCA::new(2);

// Fit on training data
pca.fit(&x_train)?;

// Transform data to principal components
let x_transformed = pca.transform(&x_train);

// Check explained variance ratio
let var_ratio = pca.explained_variance_ratio(&x_train);
println!("Variance explained: {:?}", var_ratio);

// Reconstruct data
let x_recon = pca.inverse_transform(&x_transformed);
let mse = pca.reconstruction_mse(&x_train);
println!("Reconstruction MSE: {}", mse);
```

## 📈 Performance

### Linear Regression Benchmarks

**California Housing Dataset** (20,640 samples, 8 features):

| Operation | Time |
|-----------|------|
| Data Loading (CSV) | ~10 ms |
| Training (OLS) | ~25 ms |
| Prediction (4,128 samples) | ~1.2 ms |
| Model R² Score | 0.646 |

**Detailed results**: See `results/Linear_regression_Result.csv.xlsx`

### Logistic Regression Benchmarks

**Diabetes Dataset** (768 samples, 8 features):

| Metric | Value |
|--------|-------|
| Training Time (1000 epochs) | ~XXX ms |
| Prediction Time | ~XX ms |
| ROC-AUC Score | 0.8xxx |
| Accuracy | 0.7xxx |
| F1-Score | 0.6xxx |

**Detailed results**: See `results/Logsitic_regression_Result.xlsx`

### K-Means Clustering Benchmarks

**Dataset** (167 samples with categorical features, auto-encoded):

| Metric | Value |
|--------|-------|
| Fit Time | ~24 ms |
| Predict Time | ~3.2 ms |
| Per-sample Prediction | ~0.000019 seconds |
| Inertia | 36528388099.32 |
| Silhouette Score | 0.726 |
| Davies–Bouldin Index | 0.519 |
| Calinski–Harabasz Score | 365.57 |

**Detailed results**: See `results/KMeans_Result.xlsx`

### PCA (Principal Component Analysis) Benchmarks

**Dataset** (167 samples, multiple features):

| Metric | Value |
|--------|-------|
| Full PCA Fit Time | ~XX ms |
| Components for 95% Variance | Automatically determined |
| Final PCA Fit Time | ~XX ms |
| Transform Time | ~XX ms |

**Detailed results**: See `results/pca_results.xlsx`

## 🔬 Compare Python vs Rust PCA

To verify correctness, compare the outputs side-by-side in Excel:

### Python Outputs (scikit-learn)
- `results/pca_components.csv` - Principal components (eigenvectors)
- `results/pca_explained.csv` - Explained variance ratios and cumulative variance
- `results/pca_reduced.csv` - Transformed data (principal component scores)

### Rust Outputs (This Implementation)
- `results/rust_pca_components.csv` - Principal components (eigenvectors)
- `results/rust_pca_explained.csv` - Explained variance ratios and cumulative variance
- `results/rust_pca_reduced.csv` - Transformed data (principal component scores)

### Comparison Notes
- **Numerical Accuracy**: Values should match to high precision (typically within 1e-10)
- **Sign Ambiguity**: PCA components may flip sign between implementations - this is mathematically correct. Use absolute values for comparison.
- **Comparison File**: See `results/PCA_Comparision.xlsx` for side-by-side analysis

**How to Run:**
```bash
# Run Rust implementation
cargo run --bin PCA

# Run Python comparison
cd test
python PCA.py
```

### Key Performance Features
- ✅ Pure Rust implementation - no Python overhead
- ✅ SIMD-optimized matrix operations via `ndarray`
- ✅ Zero-copy data structures where possible
- ✅ Efficient memory layout with contiguous arrays
- ✅ K-Means++ initialization for faster convergence

## 🎯 Use Cases

- **Learning Rust** - See how ML algorithms work under the hood
- **High Performance ML** - Leverage Rust's speed for production systems
- **Custom ML Pipelines** - Build specialized preprocessing and modeling workflows
- **Research** - Experiment with algorithm modifications

## 🔬 Example: Loading Datasets

### Option 1: Basic CSV/Excel Loading (Numeric Data Only)

For datasets with all numeric features:

```rust
// CSV with column index
let (x, y) = data::load_csv("your_data.csv", 5)?;

// CSV with column name (recommended)
let (x, y) = data::load_csv_by_name("your_data.csv", "target_column_name")?;

// Excel with column name and default (first) sheet
let (x, y) = data::load_excel_by_name("housing_data.xlsx", "price", None)?;

// Excel with column name and specific sheet
let (x, y) = data::load_excel_by_name("housing_data.xlsx", "price", Some("Sheet1"))?;

// Excel with column index
let (x, y) = data::load_excel("housing_data.xlsx", 8, None)?;
```

### Option 2: Smart Dataset Loader (Automatic Categorical Handling)

**For datasets with mixed numeric and categorical features** - the smart loader automatically:
- Detects categorical columns (strings that can't parse as numbers)
- One-hot encodes categorical features
- Handles categorical targets with label mapping
- Preserves feature names

```rust
use ML_ALGO_Rewite::load_data;

// Load with categorical target (e.g., species: "setosa", "versicolor", "virginica")
let ds = load_data::DatasetLoader::from_path("iris.csv")
    .with_target("species")  
    .build()?;

let x = ds.x;              // Array2<f64> - fully encoded (numeric + one-hot)
let y = ds.y.unwrap();     // Array1<f64> - [0.0, 1.0, 2.0, ...] for categories
let features = ds.feature_names;  // Vec<String> - ["sepal_length", "petal_width", "color__red", ...]

// Check label mapping for categorical targets
if let Some(labels) = ds.label_map {
    println!("Classes: {:?}", labels);  // ["setosa", "versicolor", "virginica"]
}

// Load without target (unsupervised learning)
let ds2 = load_data::DatasetLoader::from_path("data.csv")
    .build()?;  // X only, no y

// Load with numeric target by column index
let ds3 = load_data::DatasetLoader::from_path("data.csv")
    .with_target_index(0)
    .build()?;
```

**When to use which:**
- **Basic loaders** (`load_csv`, `load_excel`): Pure numeric data, manual preprocessing
- **Smart loader** (`DatasetLoader`): Mixed data types, automatic feature engineering

## 🎯 What This Library Handles Automatically

### Data Loading & Preprocessing
- ✅ CSV and Excel (.csv, .xlsx) file formats
- ✅ Missing values (NaN detection and median imputation)
- ✅ Categorical feature detection and one-hot encoding
- ✅ Categorical target encoding with label mapping
- ✅ Feature name preservation

### Machine Learning
- ✅ Linear regression (OLS with ridge fallback)
- ✅ Logistic regression with L2 regularization
- ✅ K-Means clustering with K-Means++ initialization
- ✅ Feature scaling (z-score normalization)
- ✅ Train-test splitting with optional shuffling

### Evaluation & Persistence
- ✅ Regression metrics (MSE, RMSE, R²)
- ✅ Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Clustering metrics (Inertia)
- ✅ Model serialization (JSON, CSV)

## 🛠️ Development

### Run Tests
```bash
cargo test
```

### Build for Release (Optimized)
```bash
cargo build --release
```

### Check Code (Fast)
```bash
cargo check
```

### Clean Build Artifacts
```bash
cargo clean
```

## 📦 Dependencies

- **ndarray** (0.15) - N-dimensional arrays for numerical computing
- **nalgebra** (0.32) - Linear algebra (matrix inversion)
- **csv** (1.3) - CSV parsing
- **calamine** (0.24) - Excel file reading (.xlsx)
- **serde** (1.0) - Serialization framework
- **serde_json** (1.0) - JSON serialization
- **rand** (0.8) - Random number generation
- **rand_distr** (0.4) - Probability distributions (for K-Means++)
- **ndarray-rand** (0.14) - Random array generation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional algorithms (decision trees, SVM, neural networks, DBSCAN, etc.)
- More preprocessing techniques (PCA, polynomial features, min-max scaling)
- Cross-validation
- L1 regularization (Lasso) for linear regression
- Multi-class classification (softmax regression)
- Hierarchical clustering
- Performance optimizations
- GPU acceleration support

## 📝 License

This project is available for educational and research purposes.

## 👨‍💻 Author

**Gaurav**
- GitHub: [@gaurav13407](https://github.com/gaurav13407)
- Repository: [ML_ALGO_IN_RUST](https://github.com/gaurav13407/ML_ALGO_IN_RUST)

## 🙏 Acknowledgments

- Implemented from scratch for learning and understanding
- Inspired by scikit-learn's API design
- Built with Rust's performance and safety in mind

---

**Happy Machine Learning with Rust! 🦀📊**
