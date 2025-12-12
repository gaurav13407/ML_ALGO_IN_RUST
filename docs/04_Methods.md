# Chapter 4: Mathematical Foundations and Methods

## 4.1 Formal Definition of K-Nearest Neighbors

### 4.1.1 Problem Formulation

Let (X, Y) denote a labeled dataset where:
- X = {x₁, x₂, ..., xₙ} ⊂ ℝᵈ represents n training samples in d-dimensional feature space
- Y = {y₁, y₂, ..., yₙ} ⊂ {1, 2, ..., C} represents corresponding class labels from C classes

For a new test instance x* ∈ ℝᵈ, the KNN classification task is to predict its label ŷ.

**Definition 1 (k-Nearest Neighbors Set):**
Given a distance metric d: ℝᵈ × ℝᵈ → ℝ₊, the set of k-nearest neighbors of x* is:

```
N_k(x*) = {x_(1), x_(2), ..., x_(k)} ⊆ X
```

Where x_(i) denotes the i-th nearest training sample satisfying:

```
d(x*, x_(1)) ≤ d(x*, x_(2)) ≤ ... ≤ d(x*, x_(k)) ≤ d(x*, x_(k+1)) ≤ ... ≤ d(x*, x_(n))
```

**Definition 2 (Majority Vote Classification):**
The predicted label ŷ is the most frequent class among k neighbors:

```
ŷ = argmax_{c ∈ {1,...,C}} Σ_{x_i ∈ N_k(x*)} 𝟙[y_i = c]
```

Where 𝟙[·] is the indicator function (1 if condition true, 0 otherwise).

### 4.1.2 Distance-Weighted Voting

Standard KNN treats all k neighbors equally, regardless of their proximity to x*. Distance-weighted voting assigns higher influence to closer neighbors.

**Definition 3 (Weighted KNN):**

```
ŷ = argmax_{c ∈ {1,...,C}} Σ_{x_i ∈ N_k(x*)} w_i · 𝟙[y_i = c]
```

Where weights w_i satisfy:
1. 0 ≤ w_i ≤ 1 (normalized weights)
2. w_i ≥ w_j if d(x*, x_i) ≤ d(x*, x_j) (monotonicity)
3. Σ w_i = 1 (sum to unity for probability interpretation)

**Common Weighting Schemes:**

**Uniform:**
```
w_i = 1/k  for all i ∈ {1,...,k}
```

**Inverse Distance:**
```
w_i = 1/d(x*, x_i) / Σ_{j=1}^k 1/d(x*, x_j)
```
(undefined for d=0; use small ε or assign w=1 for exact matches)

**Dudani's Weighting:**
```
w_i = (d_k - d_i) / (d_k - d_1)  if d_k ≠ d_1
w_i = 1/k                         otherwise
```
Where d_i = d(x*, x_i), ensuring linear interpolation between farthest and nearest neighbors.

**Gaussian Kernel:**
```
w_i = exp(-d²(x*, x_i) / 2σ²) / Σ_{j=1}^k exp(-d²(x*, x_j) / 2σ²)
```
Where σ controls bandwidth (larger σ → more uniform weighting).

## 4.2 Distance Metrics

### 4.2.1 Euclidean Distance (L2 Norm)

**Definition:**
```
d_2(x, y) = √(Σ_{i=1}^d (x_i - y_i)²) = ||x - y||_2
```

**Properties:**
- **Rotation invariant:** d(Rx, Ry) = d(x, y) for orthogonal matrix R
- **Translation invariant:** d(x+c, y+c) = d(x, y) for any vector c
- **Homogeneous:** d(αx, αy) = |α|·d(x, y) for scalar α
- **Triangle inequality:** d(x, z) ≤ d(x, y) + d(y, z)

**Computational Complexity:**
- Naive: O(d) with 2d memory accesses, d subtractions, d multiplications, d-1 additions, 1 square root
- Optimized: O(d) with SIMD parallelism, achieving 4-16× speedup on modern CPUs

**SIMD Optimization (AVX2):**
```rust
// Pseudo-Rust with AVX2 intrinsics
unsafe {
    let mut sum = _mm256_setzero_ps(); // 8×f32 accumulator
    for i in (0..d).step_by(8) {
        let x_vec = _mm256_loadu_ps(&x[i]);
        let y_vec = _mm256_loadu_ps(&y[i]);
        let diff = _mm256_sub_ps(x_vec, y_vec);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    // Horizontal sum and sqrt
    let result = horizontal_sum_and_sqrt(sum);
}
```

This processes 8 dimensions per cycle vs 1 for scalar code—8× theoretical speedup, 4-6× observed (memory bandwidth limits).

### 4.2.2 Manhattan Distance (L1 Norm)

**Definition:**
```
d_1(x, y) = Σ_{i=1}^d |x_i - y_i| = ||x - y||_1
```

**Properties:**
- **Grid metric:** Reflects city-block distance in Manhattan-style grids
- **Robust to outliers:** Linear growth vs quadratic for Euclidean
- **Not rotation invariant:** d(Rx, Ry) ≠ d(x, y) in general

**Use Cases:**
- High-dimensional spaces where L2 distances cluster near same value (curse of dimensionality mitigation)
- Count data or integer features where differences represent discrete quantities
- Sparse features where many dimensions are zero (L1 preserves sparsity better)

**Computational Advantage:**
No expensive square root operation; replace multiplication with absolute value (faster on most architectures).

### 4.2.3 Minkowski Distance (Lp Norm)

**Definition:**
```
d_p(x, y) = (Σ_{i=1}^d |x_i - y_i|^p)^(1/p)
```

**Special Cases:**
- p=1: Manhattan distance
- p=2: Euclidean distance
- p→∞: Chebyshev distance d_∞(x,y) = max_i |x_i - y_i|

**Fractional p (0 < p < 1):**
Technically not a metric (violates triangle inequality), but sometimes used empirically. Emphasizes smaller differences, useful for high-dimensional data where typical L2 distances saturate.

**Computational Cost:**
O(d·log p) due to power operations; expensive for non-integer p. Rarely used in practice except p ∈ {1, 2, ∞}.

### 4.2.4 Mahalanobis Distance

**Definition:**
```
d_M(x, y) = √((x-y)ᵀ Σ⁻¹ (x-y))
```

Where Σ is the covariance matrix of training data.

**Intuition:**
Mahalanobis distance accounts for feature correlations and different variances, effectively "whitening" the space. If features are independent and unit variance, Mahalanobis reduces to Euclidean.

**Computational Cost:**
- Precomputation: O(d² · n) to compute Σ, O(d³) to invert
- Per-distance: O(d²) for matrix-vector multiplication

**Practical Considerations:**
- High computational cost limits use to low-dimensional settings (d < 100)
- Requires sufficient data to estimate Σ reliably (rule of thumb: n > 10d)
- Sensitive to estimation errors when d is large relative to n (singular matrix issues)

## 4.3 Classification Metrics

### 4.3.1 Confusion Matrix

For binary classification, predictions fall into four categories:

```
                  Predicted Positive | Predicted Negative
Actual Positive:       TP            |        FN
Actual Negative:       FP            |        TN
```

- **True Positives (TP):** Correctly identified positive samples
- **False Positives (FP):** Negative samples incorrectly labeled positive (Type I error)
- **False Negatives (FN):** Positive samples incorrectly labeled negative (Type II error)
- **True Negatives (TN):** Correctly identified negative samples

### 4.3.2 Accuracy

**Definition:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN) = Correct / Total
```

**Interpretation:** Proportion of all predictions that are correct.

**Limitations:**
- **Misleading on imbalanced datasets:** 95% accuracy sounds impressive, but on 95:5 imbalanced dataset, always predicting majority class achieves same accuracy while being useless.
- **Ignores error types:** Treats false positives and false negatives equally, inappropriate when error costs differ (e.g., medical diagnosis where FN is far more costly than FP).

**Use Cases:**
Balanced datasets where errors are equally costly.

### 4.3.3 Precision

**Definition:**
```
Precision = TP / (TP + FP) = TP / (Predicted Positive)
```

**Interpretation:** Of all samples predicted positive, what proportion are actually positive?

**Alternative Names:** Positive Predictive Value (PPV)

**Use Cases:**
When false positives are costly:
- **Spam filtering:** FP means legitimate email goes to spam
- **Medical screening:** FP means unnecessary follow-up procedures, patient anxiety
- **Fraud detection:** FP means blocking legitimate transactions, customer frustration

**Mathematical Insight:**
Precision answers: "If the classifier says positive, how confident should I be?"

### 4.3.4 Recall (Sensitivity)

**Definition:**
```
Recall = TP / (TP + FN) = TP / (Actual Positive)
```

**Interpretation:** Of all actual positive samples, what proportion did we identify?

**Alternative Names:** Sensitivity, True Positive Rate (TPR), Hit Rate

**Use Cases:**
When false negatives are costly:
- **Cancer screening:** FN means missing disease, potentially fatal
- **Security intrusion detection:** FN means undetected attack
- **Search engines:** FN means relevant results not shown

**Mathematical Insight:**
Recall answers: "What fraction of positives are we catching?"

### 4.3.5 F1 Score

**Definition:**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall) = 2TP / (2TP + FP + FN)
```

**Derivation:** Harmonic mean of precision and recall.

**Why Harmonic Mean?**
Arithmetic mean treats precision and recall equally: (P+R)/2. But we want to penalize extreme imbalance—if one metric is very low, F1 should be low.

Harmonic mean properties:
- HM(x, y) ≤ AM(x, y) with equality iff x=y
- HM(ε, 1) ≈ 2ε as ε → 0 (severely penalizes low values)

**Example:**
- Precision=0.9, Recall=0.9 → F1=0.900 (balanced, excellent)
- Precision=0.99, Recall=0.5 → F1=0.662 (imbalanced, mediocre despite high precision)
- Precision=0.1, Recall=0.99 → F1=0.182 (imbalanced, poor despite high recall)

**Generalization: Fβ Score**
```
Fβ = (1+β²) · (Precision · Recall) / (β² · Precision + Recall)
```

Where β weights recall β times as important as precision:
- β=0: Fβ=Precision (recall ignored)
- β=1: Fβ=F1 (balanced)
- β=2: F2 weighs recall 2× higher (medical screening)
- β=0.5: F0.5 weighs precision 2× higher (search engines)

### 4.3.6 ROC-AUC Score

**ROC Curve Definition:**
Receiver Operating Characteristic curve plots True Positive Rate (Recall) vs False Positive Rate across all decision thresholds.

```
TPR = TP / (TP + FN)  [y-axis]
FPR = FP / (FP + TN)  [x-axis]
```

For KNN, thresholds correspond to k-neighbor vote proportions: 0/k, 1/k, 2/k, ..., k/k.

**AUC Definition:**
Area Under the ROC Curve—ranges [0, 1] where:
- 1.0 = perfect classifier (TPR=1, FPR=0 for some threshold)
- 0.5 = random classifier (diagonal line TPR=FPR)
- <0.5 = worse than random (invert predictions!)

**Interpretation via Mann-Whitney U Statistic:**
AUC equals probability that randomly chosen positive instance ranks higher than randomly chosen negative instance:

```
AUC = P(score(x⁺) > score(x⁻))  for x⁺~Positive, x⁻~Negative
```

**Computational Formula:**
For n₊ positive and n₋ negative samples with sorted scores:

```
AUC = (Σ rank_i - n₊(n₊+1)/2) / (n₊ · n₋)
```

Where rank_i is rank of i-th positive sample (1=lowest score).

**Advantages:**
- Threshold-independent: Single number summarizing performance across all operating points
- Handles class imbalance better than accuracy
- Interpretable as ranking quality metric

**Disadvantages:**
- May be overly optimistic if class distribution shifts in deployment
- Doesn't directly optimize operational metric (cost matrix unknown)
- Computational cost: O(n log n) for sorting

## 4.4 Computational Complexity Analysis

### 4.4.1 Training Complexity

**Naive Implementation:**
```
function fit(X_train, y_train):
    store X_train, y_train in memory
    return
```
**Time:** O(1)  
**Space:** O(n·d)

KNN is a **lazy learner**—no model parameters are learned during training. This contrasts sharply with eager learners like neural networks (backpropagation O(iterations · n · parameters)) or SVMs (quadratic programming O(n² · d) to O(n³ · d)).

**Implications:**
- Instant training: Ideal for online learning scenarios
- No opportunity for preprocessing optimizations during training
- Full dataset must be retained: Memory grows linearly with data

### 4.4.2 Prediction Complexity

**Naive Implementation:**
```
function predict(x_test):
    for each x_train in X_train:
        compute d(x_test, x_train)     // O(d) per distance
    select k smallest distances         // O(n log k) with heap
    vote on class labels                // O(k)
    return predicted_class
```

**Per-test-instance complexity:**
- Distance computation: O(n · d)
- Neighbor selection: O(n log k)
- Voting: O(k)
- **Total:** O(n · d + n log k) = O(n · d) since d typically >> log k

**Batch prediction (m test instances):**
```
Total = m · O(n · d) = O(m · n · d)
```

**Scaling Analysis:**
For n=10,000, m=1,000, d=50:
```
Operations = 1,000 × 10,000 × 50 = 500,000,000 floating-point operations
```

On modern CPU (1 GFLOP/s single-core): 500 milliseconds
With 8-core parallelism: 62.5 milliseconds  
With SIMD (4× vectorization): 15.6 milliseconds

**Comparison with Other Algorithms:**

| Algorithm | Training | Prediction (per instance) |
|-----------|----------|--------------------------|
| KNN | O(1) | O(n·d) |
| Decision Tree | O(n·d·log n) | O(log n) |
| Random Forest | O(n·d·log n·T) | O(T·log n) |
| SVM | O(n²·d) to O(n³·d) | O(n_sv·d) |
| Neural Net | O(epochs·n·params) | O(params) |

Where T=number of trees, n_sv=support vectors, params=network parameters.

**Key Insight:** KNN trades off training efficiency for prediction efficiency. Suitable when training data changes frequently but predictions are infrequent, or when explainability via local neighborhoods is valuable.

### 4.4.3 Curse of Dimensionality

**Phenomenon:** As dimensionality d increases, several pathologies emerge:

**1. Volume Concentration:**
In high dimensions, essentially all points lie near the boundary of any region.

**Proof Sketch:** Consider unit hypersphere in d dimensions. Volume ratio between outer shell (0.99 ≤ r ≤ 1.0) and full sphere:
```
Ratio = (1^d - 0.99^d) / 1^d = 1 - 0.99^d
```

| d | Ratio |
|---|-------|
| 2 | 0.0199 (2%) |
| 10 | 0.0956 (9.6%) |
| 100 | 0.6340 (63.4%) |
| 1000 | ~1.0 (99.99%+) |

In d=1000, virtually all volume is in the outermost 1% shell!

**2. Distance Concentration:**
All pairwise distances become similar as d grows.

**Theorem (Beyer et al. 1999):** For i.i.d. features with finite variance, as d→∞:
```
(d_max - d_min) / d_min → 0
```

Where d_max, d_min are maximum and minimum distances to a query point.

**Implication:** When all neighbors are equidistant, k-nearest neighbors provide no more information than random sample. KNN degrades to random guessing.

**3. Sample Complexity:**
To maintain constant density ρ in d dimensions requires n ~ ρ^d samples—exponential growth.

**Example:** 1D: 10 samples for 10 bins of width 0.1  
2D: 100 samples for 10×10 grid  
10D: 10^10 samples—larger than most real datasets!

**Mitigation Strategies:**
- **Feature selection:** Remove irrelevant dimensions
- **Dimensionality reduction:** PCA, random projections, autoencoders
- **Distance metric learning:** Learn optimal distance function for task
- **Local weighting:** Adaptive k or bandwidth based on local density

---

This chapter established mathematical rigor underlying KNN—formal definitions, distance metrics with properties and complexity, comprehensive metric definitions with derivations, and detailed computational complexity analysis including curse of dimensionality. Subsequent chapters apply these foundations to practical implementation and empirical evaluation.
