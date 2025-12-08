# ML Algorithms from Scratch in Rust

A high-performance machine learning library implemented from scratch in Rust, featuring linear regression, logistic regression, and essential data preprocessing utilities.

## 🚀 Features

### Implemented Algorithms
- **Linear Regression** - Ordinary Least Squares (OLS) using normal equation
- **Logistic Regression** - Binary classification with gradient descent and L2 regularization
- **Standard Scaler** - Feature normalization (z-score standardization)
- **Missing Value Imputation** - Median-based imputation for handling NaN values
- **Train-Test Split** - Stratified data splitting with optional shuffling
- **One-Hot Encoding** - Categorical variable encoding for single and multiple columns

### Key Capabilities
- ✅ CSV and Excel (.xlsx) data loading with automatic missing value handling (NaN)
- ✅ Feature scaling and normalization
- ✅ One-hot encoding for categorical features
- ✅ L2 regularization support for logistic regression
- ✅ Model serialization (save/load to JSON)
- ✅ High-performance matrix operations using `ndarray` and `nalgebra`
- ✅ Comprehensive evaluation metrics:
  - **Regression**: MSE, RMSE, R²
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

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
cargo run --bin logistic_regression
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

### Using as a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
ML_ALGO_Rewite = { path = "../path/to/ML_ALGO_Rewite" }
ndarray = "0.15"
```

Example usage:
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

## 📁 Project Structure

```
ML_ALGO_Rewite/
├── src/
│   ├── lib.rs           # Library entry point, exports public modules
│   ├── main.rs          # Main binary (logistic regression demo)
│   ├── data.rs          # CSV/Excel loading with NaN handling
│   ├── preprocess.rs    # Scaling, normalization, imputation
│   ├── split.rs         # Train-test splitting
│   ├── linear.rs        # Linear regression (OLS)
│   ├── logistic.rs      # Logistic regression with L2 regularization
│   ├── metrics.rs       # Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
│   ├── encoding.rs      # One-hot encoding for categorical variables
│   └── model_io.rs      # Model serialization/deserialization
├── test/
│   ├── linear_regression.rs   # Linear regression example
│   └── logistic_regression.rs # Logistic regression example
├── examples/
│   ├── Linear_regression_1.csv       # California housing dataset
│   ├── logistic_regression_5.csv     # Diabetes dataset
│   └── linear_regression_example.rs
├── models/              # Saved model files (JSON, CSV weights)
├── results/             # Benchmark results
│   ├── Linear_regression_Result.csv.xlsx
│   └── Logsitic_regression_Result.xlsx
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
pub fn confusion_counts(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> (usize, usize, usize, usize)
pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> f64
pub fn confusion_matrix_array(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> Array2<usize>
pub fn roc_auc_score(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64
```
- **Confusion Matrix**: True/False Positives/Negatives
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve using rank-based method

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

### `model_io` - Model Persistence
```rust
pub fn save_model(path: &str, coef: &Array1<f64>, mean: &Array1<f64>, std: &Array1<f64>) -> Result<(), Box<dyn Error>>
pub fn load_model(path: &str) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>>
```
- Save/load models to JSON format

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

### Key Performance Features
- ✅ Pure Rust implementation - no Python overhead
- ✅ SIMD-optimized matrix operations via `ndarray`
- ✅ Zero-copy data structures where possible
- ✅ Efficient memory layout with contiguous arrays

## 🎯 Use Cases

- **Learning Rust** - See how ML algorithms work under the hood
- **High Performance ML** - Leverage Rust's speed for production systems
- **Custom ML Pipelines** - Build specialized preprocessing and modeling workflows
- **Research** - Experiment with algorithm modifications

## 🔬 Example: Custom Dataset

### CSV Files
Replace the dataset by modifying the binary:

```rust
// Using column index
let (x, y) = data::load_csv("your_data.csv", 5)?;

// Using column name (recommended)
let (x, y) = data::load_csv_by_name("your_data.csv", "target_column_name")?;
```

### Excel Files (.xlsx)
Load data from Excel files:

```rust
// Using column name with default (first) sheet
let (x, y) = data::load_excel_by_name("housing_data.xlsx", "price", None)?;

// Using column name with specific sheet
let (x, y) = data::load_excel_by_name("housing_data.xlsx", "price", Some("Sheet1"))?;

// Using column index
let (x, y) = data::load_excel("housing_data.xlsx", 8, None)?;
```

The pipeline automatically handles:
- Missing values (median imputation)
- Feature scaling
- Model training and evaluation
- Model persistence
- CSV and Excel formats (.csv, .xlsx)

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
- **serde** (1.0) - Serialization framework
- **serde_json** (1.0) - JSON serialization
- **rand** (0.8) - Random number generation
- **ndarray-rand** (0.14) - Random array generation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional algorithms (decision trees, SVM, neural networks, etc.)
- More preprocessing techniques (PCA, polynomial features)
- Cross-validation
- L1 regularization (Lasso) for linear regression
- Multi-class classification (softmax regression)
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
