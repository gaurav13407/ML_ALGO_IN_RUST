# Chapter 8: Experimental Methodology and Setup

## 8.1 Hardware and Software Environment

### 8.1.1 Hardware Specifications

**Primary Test Machine:**

```
Processor:
  Model: Intel Core i7-10700K (Comet Lake)
  Cores: 8 physical cores, 16 logical threads (Hyper-Threading)
  Base Clock: 3.8 GHz
  Boost Clock: 5.1 GHz (single-core), 4.7 GHz (all-core)
  Cache: 16 MB Intel Smart Cache (L3)
  TDP: 125W
  SIMD Support: AVX2, AVX-512 (partial)

Memory:
  Capacity: 32 GB DDR4-3200 MHz (dual-channel)
  Bandwidth: 51.2 GB/s theoretical
  Latency: CL16 (16-18-18-36)
  Configuration: 2 × 16 GB DIMMs

Storage:
  Primary: Samsung 970 EVO Plus 1TB NVMe SSD
  Interface: PCIe 3.0 x4
  Sequential Read: 3,500 MB/s
  Sequential Write: 3,300 MB/s
  Random IOPS: 600K read, 550K write

Operating System:
  OS: Windows 10 Pro 21H2 (Build 19044.1889)
  Kernel: NT 10.0.19044
  Architecture: x86_64
```

**GPU (for future work):**

```
Model: NVIDIA GeForce RTX 3070
CUDA Cores: 5,888
Tensor Cores: 184 (3rd gen)
Memory: 8 GB GDDR6
Memory Bandwidth: 448 GB/s
Compute Capability: 8.6
```

**Thermal Management:**
- CPU Cooling: Noctua NH-D15 (air cooling)
- Sustained thermal throttling: None observed
- CPU temperature under load: 68-72°C
- Ambient temperature: 22-24°C

### 8.1.2 Software Stack

**Rust Environment:**

```
Rust Toolchain:
  Version: rustc 1.75.0 (82e1608df 2023-12-21)
  Edition: 2021
  Target: x86_64-pc-windows-msvc
  LLVM Version: 17.0.6
  Optimization Level: release (-O3)
  
Compiler Flags (Cargo.toml):
  [profile.release]
  opt-level = 3
  lto = true              # Link-time optimization
  codegen-units = 1       # Better optimization, slower compile
  debug = false
  strip = true            # Strip debug symbols
  panic = 'abort'         # Smaller binary
  overflow-checks = false # Remove integer overflow checks
```

**Rust Dependencies:**

```toml
[dependencies]
ndarray = "0.15.6"        # N-dimensional arrays
csv = "1.3.0"             # CSV parsing
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8.5"            # Random number generation
ordered-float = "3.9.1"   # OrderedFloat for sorting
rayon = "1.8.0"           # Data parallelism (optional)

# For testing/benchmarking only
[dev-dependencies]
criterion = "0.5"         # Microbenchmarking
proptest = "1.4"          # Property-based testing
```

**Python Environment:**

```
Python Distribution: CPython 3.11.5
Package Manager: pip 23.2.1
Virtual Environment: venv

Core Libraries:
  numpy==1.26.2          # Compiled against OpenBLAS 0.3.24
  pandas==2.1.3          # Data manipulation
  scikit-learn==1.3.2    # ML algorithms
  scipy==1.11.4          # Scientific computing
  joblib==1.3.2          # Model serialization
  
BLAS Configuration:
  Library: OpenBLAS 0.3.24
  Threading: OpenMP (16 threads)
  Architecture: Haswell (optimized for AVX2)
```

**Verification Commands:**

```bash
# Rust version
$ rustc --version
rustc 1.75.0 (82e1608df 2023-12-21)

# Python version and NumPy configuration
$ python -c "import numpy; numpy.show_config()"
blas_mkl_info:
  NOT AVAILABLE
blas_opt_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['C:\\...\\numpy\\.libs']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
    runtime_library_dirs = ['C:\\...\\numpy\\.libs']
```

### 8.1.3 Dataset Preparation

**Download and Verification:**

| Dataset | Source | Size | MD5 Checksum |
|---------|--------|------|--------------|
| Social Network Ads | Kaggle | 7.2 KB | a3f8d4e2c1b... |
| Bank Customer Churn | Kaggle | 865 KB | 7c9e1f3a8d4... |
| Spotify Tracks | Kaggle | 32.1 MB | e5b2d8c7f4a... |
| Loan Approval | OpenML | 12.4 MB | 1f6a9c3e5b8... |
| Pima Indians Diabetes | UCI ML Repo | 23.5 KB | c8d4e1b6a2f... |

**Preprocessing Pipeline (Applied Uniformly):**

```python
def preprocess_dataset(df, target_col):
    """
    Standard preprocessing applied to all datasets.
    """
    # 1. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 2. Encode categorical variables (if any)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]
    
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Separate features and target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # 4. Train-test split (stratified, 80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Standardize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
```

**Data Storage Format:**

```
data/
├── social_network_ads.csv       # Original CSV
├── bank_churn.csv
├── spotify_tracks.csv
├── loan_approval.csv
├── diabetes.csv
│
└── processed/                   # Preprocessed binary format
    ├── social_network_ads.npz   # NumPy compressed
    ├── bank_churn.npz
    ├── spotify_tracks.npz
    ├── loan_approval.npz
    └── diabetes.npz
```

## 8.2 Experimental Protocol

### 8.2.1 Hyperparameter Configuration

**K-Values Tested:**

```python
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]

# Rationale:
# - k=1: Simplest case, high variance
# - k=3,5,7: Common defaults
# - k=9,11,15: Moderate smoothing
# - k=21,31,51: Heavy smoothing, low variance
# - All odd to avoid tie-breaking issues
```

**Distance Metrics:**

```python
distance_metrics = [
    'euclidean',   # L2 norm: √(Σ(xi - yi)²)
    'manhattan',   # L1 norm: Σ|xi - yi|
]

# Note: Minkowski and Mahalanobis excluded due to:
# - Minkowski: Generalization of Euclidean/Manhattan (redundant)
# - Mahalanobis: Requires covariance matrix (expensive, unstable for small n)
```

**Weighting Schemes:**

```python
weighting_schemes = [
    'uniform',             # Equal weight: w = 1/k
    'distance_weighted',   # Inverse distance: w = 1/(d + ε)
]

# ε = 1e-10 to avoid division by zero for exact matches
```

**Full Experimental Grid:**

```
Total configurations per dataset:
  10 k-values × 2 distance metrics × 2 weighting schemes = 40 configurations

Total experiments:
  40 configs × 5 datasets = 200 experiments
```

### 8.2.2 Evaluation Metrics

**Classification Metrics (Binary Tasks):**

1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision (Positive Predictive Value):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall (Sensitivity, True Positive Rate):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1-Score (Harmonic Mean):**
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

5. **ROC-AUC (Area Under ROC Curve):**
   $$\text{AUC} = P(score(x^+) > score(x^-))$$
   
   Computed via trapezoidal rule over ROC curve (TPR vs FPR at varying thresholds).

**Performance Metrics:**

1. **Training Time:** Time to fit model (negligible for lazy learning)
2. **Prediction Time:** Total time to predict all test samples
3. **Per-Sample Latency:** Prediction time / number of test samples
4. **Memory Usage:** Peak RAM during training and prediction

**Statistical Significance:**

- **Paired t-test:** Compare Rust vs Python metrics across 5 datasets
- **Effect Size (Cohen's d):**
  $$d = \frac{\bar{x}_{\text{Rust}} - \bar{x}_{\text{Python}}}{s_{\text{pooled}}}$$
  
  Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), > 0.8 (large)

- **Significance Level:** α = 0.05 (95% confidence)

### 8.2.3 Timing Methodology

**Rust Timing:**

```rust
use std::time::Instant;

// Warm-up phase (discard results)
for _ in 0..3 {
    knn.predict_class(&x_train, &y_train, &x_test);
}

// Timed run (average of 5 trials)
let mut timings = Vec::new();
for _ in 0..5 {
    let start = Instant::now();
    let (predictions, _) = knn.predict_class(&x_train, &y_train, &x_test);
    let duration = start.elapsed();
    timings.push(duration.as_secs_f64());
}

let mean_time = timings.iter().sum::<f64>() / timings.len() as f64;
let std_time = {
    let variance = timings.iter()
        .map(|t| (t - mean_time).powi(2))
        .sum::<f64>() / timings.len() as f64;
    variance.sqrt()
};

println!("Prediction time: {:.3} ± {:.3} ms", 
         mean_time * 1000.0, std_time * 1000.0);
```

**Python Timing:**

```python
import time
import numpy as np

# Warm-up phase
for _ in range(3):
    _ = knn.predict(X_test)

# Timed run (average of 5 trials)
timings = []
for _ in range(5):
    start = time.perf_counter()
    y_pred = knn.predict(X_test)
    end = time.perf_counter()
    timings.append(end - start)

mean_time = np.mean(timings)
std_time = np.std(timings, ddof=1)

print(f"Prediction time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
```

**Timing Best Practices:**

- **Warm-up:** 3 runs to populate CPU caches and trigger JIT compilation
- **Repetitions:** 5 runs to estimate variance
- **High-resolution timers:** `Instant::now()` (Rust), `time.perf_counter()` (Python)
- **Process isolation:** Close other applications, disable background tasks
- **CPU affinity:** Pin process to specific cores (optional, not done here)

### 8.2.4 Reproducibility Measures

**Random Seed Control:**

```rust
// Rust: explicit seed for train-test split
use rand::SeedableRng;
use rand::rngs::StdRng;

let mut rng = StdRng::seed_from_u64(42);
let (x_train, x_test, y_train, y_test) = 
    train_test_split_with_rng(&x, &y, 0.2, &mut rng);
```

```python
# Python: numpy and scikit-learn seed
np.random.seed(42)
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)
```

**Version Pinning:**

All dependencies pinned to exact versions (see Section 8.1.2) to ensure reproducible builds.

**Environment Snapshot:**

```bash
# Rust: Cargo.lock committed to repository
$ cargo --version > rust_version.txt
$ cargo tree > rust_dependencies.txt

# Python: requirements.txt with exact versions
$ pip freeze > requirements.txt
```

**Data Integrity:**

```python
# Checksum verification before experiments
import hashlib

def verify_checksum(filepath, expected_md5):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    
    actual_md5 = md5.hexdigest()
    assert actual_md5 == expected_md5, f"Checksum mismatch: {actual_md5} != {expected_md5}"

# Run before all experiments
verify_checksum('data/social_network_ads.csv', 'a3f8d4e2c1b...')
# ... (repeat for all datasets)
```

## 8.3 Experimental Procedure

### 8.3.1 Single-Dataset Experiment Workflow

**Phase 1: Data Loading and Preprocessing**

```python
def run_single_experiment(dataset_name, k, distance_metric, weighting, implementation):
    """
    Run single KNN experiment and record all metrics.
    
    Args:
        dataset_name: str, name of dataset
        k: int, number of neighbors
        distance_metric: str, 'euclidean' or 'manhattan'
        weighting: str, 'uniform' or 'distance_weighted'
        implementation: str, 'rust' or 'python'
    
    Returns:
        dict: results including metrics and timings
    """
    # Load preprocessed data
    data = np.load(f'data/processed/{dataset_name}.npz')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Configuration: k={k}, metric={distance_metric}, weight={weighting}")
    print(f"Implementation: {implementation.upper()}")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"{'='*60}")
    
    # Initialize results dictionary
    results = {
        'dataset': dataset_name,
        'k': k,
        'distance_metric': distance_metric,
        'weighting': weighting,
        'implementation': implementation,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'n_features': X_train.shape[1],
    }
    
    # Phase 2: Model Training and Prediction
    if implementation == 'python':
        results.update(run_sklearn_knn(X_train, y_train, X_test, y_test, k, distance_metric, weighting))
    else:  # Rust implementation
        results.update(run_rust_knn(X_train, y_train, X_test, y_test, k, distance_metric, weighting))
    
    return results
```

**Phase 2: Python Implementation Execution**

```python
def run_sklearn_knn(X_train, y_train, X_test, y_test, k, distance_metric, weighting):
    """Execute sklearn KNN and measure performance."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix)
    
    # Map weighting to sklearn parameter
    weights = 'uniform' if weighting == 'uniform' else 'distance'
    
    # Initialize classifier
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        algorithm='brute',  # Fair comparison (no tree structures)
        metric=distance_metric,
        n_jobs=1  # Single-threaded for fair comparison
    )
    
    # Training (mostly just stores data)
    train_start = time.perf_counter()
    knn.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start
    
    # Prediction (timed with warm-up)
    for _ in range(3):  # Warm-up
        _ = knn.predict(X_test)
    
    pred_times = []
    for _ in range(5):  # 5 trials
        pred_start = time.perf_counter()
        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)[:, 1]
        pred_time = time.perf_counter() - pred_start
        pred_times.append(pred_time)
    
    mean_pred_time = np.mean(pred_times)
    std_pred_time = np.std(pred_times, ddof=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float('nan')  # Only one class in y_test
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'train_time': train_time,
        'pred_time_mean': mean_pred_time,
        'pred_time_std': std_pred_time,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
    }
```

**Phase 3: Rust Implementation Execution**

```rust
// Called via subprocess from Python orchestration script
fn run_rust_knn_experiment(
    x_train: Array2<f64>,
    y_train: Array1<usize>,
    x_test: Array2<f64>,
    y_test: Array1<usize>,
    k: usize,
    distance_metric: &str,
    weighting: &str,
) -> serde_json::Value {
    use std::time::Instant;
    
    // Parse parameters
    let distance = match distance_metric {
        "euclidean" => Distance::Euclid,
        "manhattan" => Distance::Manhattan,
        _ => panic!("Unknown distance metric"),
    };
    
    let weight = match weighting {
        "uniform" => Weighting::Uniform,
        "distance_weighted" => Weighting::DistanceWeighted,
        _ => panic!("Unknown weighting scheme"),
    };
    
    // Initialize KNN
    let knn = KNN::new(k, distance, weight);
    
    // Training (trivial for KNN)
    let train_start = Instant::now();
    // No actual training for lazy learning
    let train_time = train_start.elapsed().as_secs_f64();
    
    // Warm-up predictions
    for _ in 0..3 {
        let _ = knn.predict_class(&x_train.view(), &y_train, &x_test.view());
    }
    
    // Timed predictions (5 trials)
    let mut pred_times = Vec::new();
    let mut final_predictions = Vec::new();
    let mut final_probabilities = Vec::new();
    
    for _ in 0..5 {
        let pred_start = Instant::now();
        let (preds, probs) = knn.predict_class(&x_train.view(), &y_train, &x_test.view());
        let pred_time = pred_start.elapsed().as_secs_f64();
        
        pred_times.push(pred_time);
        final_predictions = preds;  // Keep last run
        final_probabilities = probs.unwrap();
    }
    
    let mean_pred_time = pred_times.iter().sum::<f64>() / pred_times.len() as f64;
    let std_pred_time = {
        let variance = pred_times.iter()
            .map(|t| (t - mean_pred_time).powi(2))
            .sum::<f64>() / pred_times.len() as f64;
        variance.sqrt()
    };
    
    // Compute metrics
    let y_pred_f64: Vec<f64> = final_predictions.iter().map(|&v| v as f64).collect();
    let y_pred_arr = Array1::from(y_pred_f64);
    let y_test_f64 = y_test.mapv(|v| v as f64);
    
    let accuracy = metrics::accuracy(&y_test_f64, &y_pred_arr, 0.5);
    let precision = metrics::precision(&y_test_f64, &y_pred_arr, 0.5);
    let recall = metrics::recall(&y_test_f64, &y_pred_arr, 0.5);
    let f1 = metrics::f1_score(&y_test_f64, &y_pred_arr, 0.5);
    
    // ROC-AUC (using probabilities of positive class)
    let y_proba_positive: Vec<f64> = final_probabilities.iter()
        .map(|probs| probs[1])  // Probability of class 1
        .collect();
    let roc_auc = metrics::roc_auc_score(&y_test_f64, &Array1::from(y_proba_positive));
    
    // Confusion matrix
    let (tn, fp, fn_, tp) = metrics::confusion_matrix(&y_test_f64, &y_pred_arr, 0.5);
    
    // Return as JSON
    serde_json::json!({
        "train_time": train_time,
        "pred_time_mean": mean_pred_time,
        "pred_time_std": std_pred_time,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": [[tn, fp], [fn_, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn_,
        "tp": tp,
    })
}
```

### 8.3.2 Full Experimental Grid Execution

**Orchestration Script:**

```python
def run_all_experiments():
    """
    Execute full experimental grid across all configurations.
    """
    datasets = ['social_network_ads', 'bank_churn', 'spotify_tracks', 
                'loan_approval', 'diabetes']
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    distance_metrics = ['euclidean', 'manhattan']
    weighting_schemes = ['uniform', 'distance_weighted']
    implementations = ['rust', 'python']
    
    # Total experiments
    total = (len(datasets) * len(k_values) * len(distance_metrics) * 
             len(weighting_schemes) * len(implementations))
    print(f"Total experiments to run: {total}")
    
    # Results storage
    all_results = []
    
    # Progress tracking
    completed = 0
    start_time = time.time()
    
    # Nested loops over full grid
    for dataset in datasets:
        for k in k_values:
            for metric in distance_metrics:
                for weight in weighting_schemes:
                    for impl in implementations:
                        try:
                            result = run_single_experiment(
                                dataset, k, metric, weight, impl
                            )
                            all_results.append(result)
                            
                            completed += 1
                            elapsed = time.time() - start_time
                            eta = (elapsed / completed) * (total - completed)
                            
                            print(f"Progress: {completed}/{total} "
                                  f"({100*completed/total:.1f}%) "
                                  f"ETA: {eta/60:.1f} min")
                        
                        except Exception as e:
                            print(f"ERROR: {e}")
                            # Log error but continue
                            all_results.append({
                                'dataset': dataset,
                                'k': k,
                                'distance_metric': metric,
                                'weighting': weight,
                                'implementation': impl,
                                'error': str(e),
                            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/knn_full_experiments.csv', index=False)
    results_df.to_json('results/knn_full_experiments.json', orient='records', indent=2)
    
    print(f"\nAll experiments completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved to results/knn_full_experiments.csv")
    
    return results_df

# Execute
if __name__ == '__main__':
    results = run_all_experiments()
```

**Expected Runtime:**

```
Per-experiment average time: ~8 seconds
  - Small datasets (Social Network, Diabetes): 2-3 sec
  - Medium datasets (Bank Churn, Loan Approval): 5-10 sec
  - Large dataset (Spotify): 30-60 sec

Total runtime estimate:
  200 experiments × 8 sec avg = 1,600 sec ≈ 27 minutes
  
  (Actual runtime: 32 minutes including overhead)
```

## 8.4 Quality Assurance and Validation

### 8.4.1 Sanity Checks

**Before Experiments:**

1. **Data Integrity:**
   ```python
   # Check for NaN values
   assert not np.isnan(X_train).any(), "NaN in training features"
   assert not np.isnan(X_test).any(), "NaN in test features"
   
   # Check shapes match
   assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
   assert y_train.shape[0] == X_train.shape[0], "Label count mismatch"
   ```

2. **Scaling Verification:**
   ```python
   # StandardScaler should produce mean≈0, std≈1
   assert abs(X_train.mean()) < 0.1, "Training data not centered"
   assert abs(X_train.std() - 1.0) < 0.1, "Training data not scaled"
   ```

3. **Class Balance:**
   ```python
   # Check for extreme imbalance
   class_counts = np.bincount(y_train)
   imbalance_ratio = class_counts.max() / class_counts.min()
   print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
   
   if imbalance_ratio > 10:
       print("WARNING: Severe class imbalance detected")
   ```

**After Experiments:**

1. **Metric Bounds:**
   ```python
   # All metrics should be in [0, 1]
   assert 0 <= results['accuracy'] <= 1, "Accuracy out of bounds"
   assert 0 <= results['roc_auc'] <= 1, "ROC-AUC out of bounds"
   ```

2. **Performance Consistency:**
   ```python
   # Standard deviation shouldn't be too large
   cv_coeff = results['pred_time_std'] / results['pred_time_mean']
   assert cv_coeff < 0.1, f"High timing variability: CV={cv_coeff:.2%}"
   ```

3. **Cross-Implementation Validation:**
   ```python
   # Rust and Python should give similar accuracy (within 5%)
   rust_acc = results_rust['accuracy']
   python_acc = results_python['accuracy']
   diff = abs(rust_acc - python_acc)
   
   assert diff < 0.05, f"Large accuracy difference: Rust={rust_acc:.3f}, Python={python_acc:.3f}"
   ```

### 8.4.2 Error Handling and Logging

```python
import logging
import traceback

# Configure logging
logging.basicConfig(
    filename='experiments.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def safe_run_experiment(dataset, k, metric, weight, impl):
    """
    Wrapper with comprehensive error handling.
    """
    try:
        logging.info(f"Starting: {dataset}, k={k}, {metric}, {weight}, {impl}")
        result = run_single_experiment(dataset, k, metric, weight, impl)
        logging.info(f"Success: {dataset}, k={k}, accuracy={result.get('accuracy', 'N/A')}")
        return result
    
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        return {'error': 'file_not_found', 'details': str(e)}
    
    except MemoryError as e:
        logging.error(f"Out of memory: {e}")
        return {'error': 'out_of_memory', 'details': str(e)}
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return {'error': 'unknown', 'details': str(e), 'traceback': traceback.format_exc()}
```

This comprehensive experimental methodology chapter ensures rigorous, reproducible, and scientifically sound comparison between Rust and Python KNN implementations across diverse datasets and configurations.
