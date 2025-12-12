# Chapter 6: System Architecture - Rust Implementation

## 6.1 High-Level Architecture Overview

The Rust KNN implementation follows a modular architecture separating concerns into distinct components with well-defined interfaces. This design philosophy aligns with Rust's emphasis on composition, encapsulation, and zero-cost abstractions.

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (test/KNN.rs - Main execution, benchmarking, evaluation)   │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼─────────┐    ┌─────────▼────────┐
│  Data Pipeline  │    │   Model Layer    │
│                 │    │                  │
│  ├─ data.rs     │    │  ├─ KNN.rs      │
│  ├─ preprocess  │    │  └─ distance    │
│  └─ split.rs    │    │     metrics     │
└─────────────────┘    └──────────────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Evaluation Layer    │
        │                       │
        │  ├─ metrics.rs        │
        │  └─ confusion matrix  │
        └───────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   Foundation Layer    │
        │                       │
        │  ├─ ndarray (arrays)  │
        │  ├─ rand (sampling)   │
        │  └─ csv (I/O)         │
        └───────────────────────┘
```

### 6.1.1 Design Principles

**Principle 1: Separation of Concerns**
Each module addresses a single responsibility:
- `data.rs`: File I/O, parsing, validation
- `preprocess.rs`: Scaling, normalization, imputation
- `split.rs`: Train-test partitioning
- `KNN.rs`: Core algorithm logic
- `metrics.rs`: Performance evaluation

**Principle 2: Immutability by Default**
Rust's ownership system encourages immutable data structures. Transformations return new arrays rather than mutating in-place:
```rust
let x_scaled = scaler.transform(&x);  // x unchanged
```
This prevents aliasing bugs and enables fearless parallelism.

**Principle 3: Zero-Copy Views**
`ndarray` provides array views (similar to NumPy) enabling efficient slicing without allocation:
```rust
let row_view: ArrayView1<f64> = array.row(i);  // No copy
```

**Principle 4: Compile-Time Guarantees**
Type system enforces correctness:
- Shape compatibility checked at compile time where possible
- Lifetimes prevent dangling references
- Ownership prevents data races

## 6.2 Module Details

### 6.2.1 data.rs - Data Loading and I/O

**Responsibility:** Load CSV files, parse numeric data, extract features and labels.

**Key Functions:**

```rust
pub fn load_csv_by_name(
    path: &str, 
    target_column_name: &str
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>>
```

**Implementation Strategy:**

1. **CSV Parsing:** Use `csv` crate's `ReaderBuilder` for robust parsing
2. **Header Detection:** First row interpreted as column names
3. **Target Extraction:** Locate target column by name, extract as separate array
4. **Feature Matrix Construction:** Remaining columns → 2D array
5. **Type Conversion:** All values parsed as `f64` (float64)

**Error Handling:**
```rust
// Rust's Result type for explicit error propagation
match csv::Reader::from_path(path) {
    Ok(reader) => { /* process */ },
    Err(e) => return Err(Box::new(e)),
}
```

**Memory Layout:**
```
ndarray uses row-major (C-order) contiguous layout by default:

Features matrix (n × d):
  [x₁₁, x₁₂, ..., x₁ᵈ, x₂₁, x₂₂, ..., x₂ᵈ, ..., xₙ₁, ..., xₙᵈ]
   ↑─────────────────↑  ↑─────────────────↑       ↑──────────↑
      Row 1 (d elems)      Row 2 (d elems)       Row n (d elems)

Cache-friendly for row-wise iteration (common in ML).
```

**Performance Characteristics:**
- **File I/O:** Dominated by disk read speed (~500 MB/s HDD, ~3 GB/s SSD)
- **Parsing:** ~100-200 MB/s (CSV parsing overhead)
- **Allocation:** O(n·d) for n samples, d features

### 6.2.2 preprocess.rs - Data Preprocessing

**Responsibility:** Feature scaling, missing value imputation, normalization.

#### StandardScaler Implementation

```rust
pub struct StandardScaler {
    pub mean: Option<Array1<f64>>,
    pub std: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn fit(x: &Array2<f64>) -> Self {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std = x.std_axis(Axis(0), 0.0);  // ddof=0 (biased estimator)
        StandardScaler { 
            mean: Some(mean), 
            std: Some(std) 
        }
    }
    
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();
        
        // Broadcast subtraction and division
        (x - mean) / std  // ndarray operator overloading
    }
}
```

**Mathematical Operations:**

**Mean Computation:**
```rust
// ndarray's mean_axis uses SIMD-optimized summation
// Equivalent to: μ_j = (1/n) Σᵢ xᵢⱼ
let mean = x.mean_axis(Axis(0));
```

**Standard Deviation:**
```rust
// Biased estimator: σ_j = √((1/n) Σᵢ (xᵢⱼ - μ_j)²)
let std = x.std_axis(Axis(0), 0.0);  // ddof=0
```

**Broadcasting:**
```rust
// x: (n, d), mean: (d,), result: (n, d)
// ndarray automatically broadcasts mean across rows
let centered = x - &mean;  // Borrow prevents move
```

**Memory Efficiency:**
- `fit`: O(d) for mean/std storage (independent of n)
- `transform`: O(n·d) temporary for result
- No in-place modification preserves original data

#### Median Imputation

```rust
pub fn column_median(x: &Array2<f64>) -> Array1<f64> {
    let (n, d) = x.dim();
    let mut medians = Array1::<f64>::zeros(d);
    
    for j in 0..d {
        let mut col = x.column(j).to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        medians[j] = if n % 2 == 0 {
            (col[n/2 - 1] + col[n/2]) / 2.0
        } else {
            col[n/2]
        };
    }
    medians
}

pub fn impute_median(x: &Array2<f64>, medians: &Array1<f64>) -> Array2<f64> {
    let mut x_imputed = x.clone();
    
    for ((i, j), val) in x_imputed.indexed_iter_mut() {
        if val.is_nan() {
            *val = medians[j];
        }
    }
    x_imputed
}
```

**Design Rationale:**
- **Median vs Mean:** Median robust to outliers (mean pulled by extreme values)
- **Column-wise:** Each feature imputed independently
- **Two-phase:** Compute medians from training data, apply to both train/test (prevents data leakage)

### 6.2.3 split.rs - Train-Test Splitting

**Responsibility:** Partition dataset into training and testing subsets with optional stratification.

```rust
pub fn train_test_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>>
```

**Implementation Steps:**

1. **Index Generation:** Create index array [0, 1, ..., n-1]
2. **Shuffling (if enabled):**
   ```rust
   use rand::seq::SliceRandom;
   use rand::SeedableRng;
   
   let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
   indices.shuffle(&mut rng);
   ```

3. **Split Point Calculation:**
   ```rust
   let n_test = (n as f64 * test_size).round() as usize;
   let n_train = n - n_test;
   ```

4. **Array Slicing:**
   ```rust
   let train_idx = &indices[0..n_train];
   let test_idx = &indices[n_train..];
   
   let x_train = x.select(Axis(0), train_idx);  // Rows selection
   let x_test = x.select(Axis(0), test_idx);
   ```

**Stratification Implementation:**
```rust
// Group indices by class label
let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
for (i, &label) in y.iter().enumerate() {
    class_indices.entry(label as i32).or_default().push(i);
}

// Sample proportionally from each class
let mut train_indices = Vec::new();
let mut test_indices = Vec::new();

for (_class, indices) in class_indices.iter_mut() {
    indices.shuffle(&mut rng);
    let split = (indices.len() as f64 * (1.0 - test_size)) as usize;
    train_indices.extend_from_slice(&indices[..split]);
    test_indices.extend_from_slice(&indices[split..]);
}
```

**Correctness Guarantees:**
- Partition property: train_indices ∩ test_indices = ∅
- Coverage: train_indices ∪ test_indices = {0, ..., n-1}
- Stratification: Class ratios preserved within floating-point precision

### 6.2.4 KNN.rs - Core Algorithm

**Responsibility:** KNN classifier with configurable distance metrics and weighting schemes.

**Structure Definition:**

```rust
pub struct KNN {
    pub k: usize,
    pub distance: Distance,
    pub weighting: Weighting,
}

pub enum Distance {
    Euclid,
    Manhattan,
}

pub enum Weighting {
    Uniform,
    DistanceWeighted,
}
```

**Constructor and Builder Pattern:**

```rust
impl KNN {
    pub fn new(k: usize) -> Self {
        KNN {
            k,
            distance: Distance::Euclid,
            weighting: Weighting::Uniform,
        }
    }
    
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }
    
    pub fn with_weighting(mut self, weighting: Weighting) -> Self {
        self.weighting = weighting;
        self
    }
}
```

**Usage:**
```rust
let knn = KNN::new(5)
    .with_distance(Distance::Euclid)
    .with_weighting(Weighting::Uniform);
```

**Prediction Method:**

```rust
pub fn predict_class(
    &self,
    x_train: &ArrayView2<f64>,
    y_train: &Array1<usize>,
    x_test: &ArrayView2<f64>,
) -> (Vec<usize>, Option<Vec<Vec<f64>>>)
```

**Implementation Algorithm:**

```
For each test instance x_test[i]:
  1. Compute distances to all training instances
     distances[j] = distance_metric(x_test[i], x_train[j])  for j=0..n_train
  
  2. Find k nearest neighbors
     Sort distances, keep indices of k smallest
     neighbors = argsort(distances)[0..k]
  
  3. Extract neighbor labels
     neighbor_labels = y_train[neighbors]
  
  4. Apply weighting scheme
     If Uniform: weight[j] = 1/k
     If DistanceWeighted: weight[j] = 1 / (distances[neighbors[j]] + ε)
  
  5. Vote for class
     For each class c:
       votes[c] = Σ weight[j] where neighbor_labels[j] == c
     predicted_class = argmax(votes)
  
  6. Compute class probabilities (optional)
     probs[c] = votes[c] / Σ votes
```

**Distance Computation:**

```rust
fn euclidean_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn manhattan_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .sum()
}
```

**Neighbor Selection (Min-Heap):**

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

// BinaryHeap is max-heap by default; Reverse for min-heap
let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> = BinaryHeap::new();

for (j, dist) in distances.iter().enumerate() {
    heap.push(Reverse((OrderedFloat(*dist), j)));
}

// Extract k smallest
let mut neighbors = Vec::with_capacity(self.k);
for _ in 0..self.k {
    if let Some(Reverse((_, idx))) = heap.pop() {
        neighbors.push(idx);
    }
}
```

**Complexity:**
- Distance computation: O(n·d) for n training samples, d features
- Heap operations: O(n log k) for k neighbors
- Voting: O(k·C) for C classes
- **Total per prediction:** O(n·d + n log k) ≈ O(n·d)

### 6.2.5 metrics.rs - Evaluation Metrics

**Responsibility:** Compute classification performance metrics.

**Confusion Matrix:**

```rust
pub fn confusion_matrix_array(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    threshold: f64,
) -> Array2<usize> {
    let (tn, fp, fn_, tp) = confusion_counts(y_true, y_pred, threshold);
    Array2::from_shape_vec((2, 2), vec![tn, fp, fn_, tp]).unwrap()
}
```

**ROC-AUC via Mann-Whitney:**

```rust
pub fn roc_auc_score(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64 {
    // Sort by predicted probability
    let mut pairs: Vec<(f64, f64)> = y_true.iter()
        .zip(y_proba.iter())
        .map(|(&yt, &yp)| (yp, yt))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Compute rank sum for positive class
    let mut rank_sum_pos = 0.0;
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    
    for (rank, (_, label)) in pairs.iter().enumerate() {
        if *label == 1.0 {
            rank_sum_pos += (rank + 1) as f64;  // Ranks are 1-indexed
            n_pos += 1.0;
        } else {
            n_neg += 1.0;
        }
    }
    
    // Mann-Whitney U statistic
    let u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0;
    let auc = u / (n_pos * n_neg);
    auc
}
```

## 6.3 Why No Bias Term for KNN?

Unlike linear models (y = w·x + b), KNN doesn't use a bias term. Understanding why reveals fundamental differences between parametric and non-parametric methods.

**Linear Models Require Bias:**
```
ŷ = w₁x₁ + w₂x₂ + ... + wᵈxᵈ + b

Without b, decision boundary forced through origin:
  w·x = 0  ⟹  Boundary passes through (0,0,...,0)

With b, boundary can be positioned anywhere:
  w·x + b = 0  ⟹  Boundary offset by -b/||w||
```

**KNN Uses Distances:**
```
KNN decision based on: d(x_test, x_train)

Adding constant feature x₀=1 to both:
  x_test = [1, x₁, x₂, ..., xᵈ]
  x_train = [1, x₁, x₂, ..., xᵈ]

Euclidean distance:
  d² = (1-1)² + Σ(x_test,i - x_train,i)²
     = 0 + Σ(x_test,i - x_train,i)²  [unchanged!]

Bias cancels out—provides no information.
```

**Worse: Bias Harms Performance:**
```
Adding x₀=1 increases dimensionality from d → d+1.

Every distance computation now includes:
  d² = (x₀,test - x₀,train)² + Σ(xᵢ,test - xᵢ,train)²
     =  ↑─────0²─────↑      + relevant differences
     
Wasted computation + slight numerical error accumulation.
```

**Contrast with Logistic Regression:**
```
Logistic: P(y=1|x) = σ(w·x + b)

Bias b shifts decision boundary (probability threshold).
Essential for handling class imbalance and position adjustment.
```

## 6.4 Memory Layout and Performance

**ndarray Memory Model:**

```
Array2<f64> uses contiguous row-major storage:

Physical memory layout for 3×4 array:
  [a₁₁, a₁₂, a₁₃, a₁₄, a₂₁, a₂₂, a₂₃, a₂₄, a₃₁, a₃₂, a₃₃, a₃₄]
   ↑─────────────────↑  ↑─────────────────↑  ↑─────────────────↑
       Row 1               Row 2               Row 3

Logical indexing: array[[i, j]] → memory[i * n_cols + j]
```

**Cache Performance:**
- Row-wise iteration: Sequential access → excellent cache hit rates
- Column-wise iteration: Strided access → cache misses every stride

**Distance Computation Access Pattern:**
```rust
// Good: Sequential access within each row
for i in 0..n_test {
    let test_row = x_test.row(i);  // Sequential scan
    for j in 0..n_train {
        let train_row = x_train.row(j);  // Sequential scan
        let dist = euclidean(test_row, train_row);  // Sequential ops
    }
}
```

**SIMD Opportunities:**
```rust
// Manual SIMD (unsafe, nightly Rust):
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe {
    // Load 4 f64 values at once (AVX)
    let x_vec = _mm256_loadu_pd(x_ptr);
    let y_vec = _mm256_loadu_pd(y_ptr);
    let diff = _mm256_sub_pd(x_vec, y_vec);
    let sq = _mm256_mul_pd(diff, diff);
    // accumulate...
}
```

**Current Implementation:**
- Relies on LLVM auto-vectorization (effectiveness varies)
- Explicit SIMD implementation deferred to future work
- CPU profile shows ~40% time in distance calculations (optimization target)

## 6.5 Rust Safety Guarantees vs Python

### 6.5.1 Memory Safety

**Python Issues:**
```python
# Possible segfault in C extension
import numpy as np
arr = np.array([1, 2, 3])
ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
del arr  # Free memory
ptr[0]  # Use-after-free → crash or silent corruption
```

**Rust Prevention:**
```rust
let arr = Array1::from(vec![1, 2, 3]);
let view = arr.view();  // Lifetime tied to arr
drop(arr);  // Compile error: arr borrowed by view
// view[0];  // Would be use-after-free, caught at compile time
```

### 6.5.2 Data Race Freedom

**Python GIL:**
```python
# GIL prevents true parallelism
import threading

def increment():
    global counter
    for _ in range(1000000):
        counter += 1  # Not atomic!

threads = [threading.Thread(target=increment) for _ in range(2)]
# Despite GIL, race conditions possible in C extensions
```

**Rust Type System:**
```rust
use std::sync::Arc;
use std::sync::Mutex;

let counter = Arc::new(Mutex::new(0));
let handles: Vec<_> = (0..2).map(|_| {
    let counter = Arc::clone(&counter);
    std::thread::spawn(move || {
        for _ in 0..1_000_000 {
            *counter.lock().unwrap() += 1;  // Mutex enforced by type system
        }
    })
}).collect();

// Compile error if trying to access counter without lock
```

### 6.5.3 Null Pointer Dereferences

**Python:**
```python
def get_value(d, key):
    return d[key]  # Runtime error if key missing

val = get_value({'a': 1}, 'b')  # KeyError at runtime
```

**Rust:**
```rust
fn get_value(map: &HashMap<String, i32>, key: &str) -> Option<i32> {
    map.get(key).copied()  // Returns Option<i32>
}

// Compile error: can't use value directly
// let val = get_value(&map, "b");

// Must handle None case explicitly
let val = match get_value(&map, "b") {
    Some(v) => v,
    None => 0,  // Default value
};
```

**Result:** Rust forces explicit error handling—no silent failures or crashes.
