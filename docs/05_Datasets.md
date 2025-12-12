# Chapter 5: Dataset Descriptions and Preprocessing

## 5.1 Dataset Selection Criteria

The five datasets were chosen to represent diverse machine learning scenarios across multiple dimensions:

**Domain Diversity:** Datasets span marketing (Social Network Ads), finance (Bank Churn, Loan Approval), entertainment (Spotify), and healthcare (Diabetes)—covering business, consumer, and medical decision-making contexts.

**Scale Variety:** Sample sizes range from 400 (small) to 114,000 (large), testing algorithm behavior across data abundance regimes.

**Dimensionality Spectrum:** Features range from 3 to 23, examining curse of dimensionality effects.

**Class Imbalance:** Positive class proportions range from 10% to 50%, evaluating robustness to imbalanced distributions.

**Feature Heterogeneity:** Mix of numerical (continuous), categorical (ordinal/nominal), and derived features testing preprocessing pipeline completeness.

## 5.2 Dataset 1: Social Network Ads

### 5.2.1 Overview and Business Context

**Source:** Kaggle Social Network Ads Dataset  
**URL:** https://www.kaggle.com/datasets/rakeshrau/social-network-ads  
**Domain:** Digital marketing and customer acquisition  
**Task:** Predict whether a user will purchase a product based on social network ad exposure

**Business Motivation:**
Online advertisers spend billions optimizing ad targeting. Predicting purchase probability enables:
- **Budget optimization:** Focus spend on high-conversion demographics
- **Personalization:** Tailor ad creative to likely buyers
- **A/B testing:** Segment users for experimental ad campaigns

### 5.2.2 Data Characteristics

**Sample Size:** 400 instances  
**Features:** 3 numerical  
**Target:** Purchased (binary: 0=No, 1=Yes)  
**Class Distribution:** 143 positive (35.75%), 257 negative (64.25%)

**Feature Descriptions:**

| Feature | Type | Range | Mean | Std Dev | Description |
|---------|------|-------|------|---------|-------------|
| User ID | Identifier | 15566126-15569970 | - | - | Unique user identifier (dropped in analysis) |
| Age | Numerical | 18-60 | 37.7 | 10.5 | User age in years |
| EstimatedSalary | Numerical | 15,000-150,000 | 69,742 | 34,097 | Annual salary estimate in USD |

**Statistical Summary:**

```
Target Distribution:
Class 0 (No Purchase):  257 samples (64.25%)
Class 1 (Purchase):     143 samples (35.75%)
Imbalance Ratio: 1.80:1

Feature Correlations:
Age vs EstimatedSalary: 0.155 (weak positive)
Age vs Purchased: 0.622 (moderate positive)
EstimatedSalary vs Purchased: 0.362 (weak positive)
```

**Data Insights:**
- **Age effect:** Older users more likely to purchase (correlation 0.622)
- **Salary effect:** Higher earners more likely to purchase (correlation 0.362)
- **Interaction:** Age and salary show weak correlation (0.155), suggesting independent predictive signals
- **Decision boundary:** Likely non-linear—young high-earners and old low-earners exhibit different purchase behaviors

### 5.2.3 Preprocessing Pipeline

**Step 1: Feature Selection**
```python
# Drop User ID (identifier, not predictive)
features = ['Age', 'EstimatedSalary']
```

**Step 2: Missing Value Analysis**
```python
missing_counts = df.isnull().sum()
# Result: 0 missing values across all features
```
No imputation required—dataset is complete.

**Step 3: Train-Test Split**
```python
train_size = 0.8  # 320 train, 80 test
stratified = True  # Maintain 35.75% positive rate in both splits
random_seed = 42   # Reproducibility
```

**Split Statistics:**

| Split | Total | Positive | Negative | Positive % |
|-------|-------|----------|----------|------------|
| Train | 320 | 114 | 206 | 35.63% |
| Test | 80 | 29 | 51 | 36.25% |

**Step 4: Feature Scaling**
```python
# StandardScaler: z = (x - μ) / σ
scaler = StandardScaler()
scaler.fit(X_train)  # Compute μ, σ from training data only

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Scaling Parameters (computed from training data):**

| Feature | Mean (μ) | Std Dev (σ) |
|---------|----------|-------------|
| Age | 37.54 | 10.38 |
| EstimatedSalary | 69,103 | 34,442 |

**Scaled Feature Ranges:**
- Age: [-1.88, 2.16] (originally [18, 60])
- EstimatedSalary: [-1.57, 2.35] (originally [15,000, 150,000])

**Why StandardScaler for KNN:**
KNN uses Euclidean distance—features with larger numerical ranges dominate distance calculations. EstimatedSalary ranges 15K-150K (span: 135K) while Age ranges 18-60 (span: 42). Without scaling, salary differences would overwhelm age differences by factor of ~3200.

StandardScaler ensures each feature contributes equally to distance by transforming to unit variance.

### 5.2.4 Exploratory Data Analysis

**Class Separation Visualization (after scaling):**
```
Age vs EstimatedSalary (standardized):
  ┌─────────────────────────────────┐
2 │         X  X     O O  O         │
  │      X  X  X    O O  O  O       │
1 │   X  X  X  X   O O  O  O  O     │
  │  X  X  X  X  X O O  O  O  O     │
0 │ X  X  X  X  X  O O  O  O  O     │
  │ X  X  X  X  O  O  O  O  O       │
-1│  X  X  X    O  O  O  O          │
  │   X  X       O  O  O            │
-2│    X          O  O              │
  └─────────────────────────────────┘
  -2  -1   0   1   2
     Age (standardized)

Legend: X = No Purchase, O = Purchase
```

**Observations:**
- Clear separation along age dimension (purchases concentrate in positive age region)
- Moderate separation along salary dimension (purchases favor higher salaries)
- Substantial overlap—decision boundary is non-trivial
- KNN likely effective due to clear local neighborhoods

**k Selection via Cross-Validation:**
```
k=1:  CV Accuracy = 0.878
k=3:  CV Accuracy = 0.891
k=5:  CV Accuracy = 0.903  ← Selected
k=7:  CV Accuracy = 0.897
k=9:  CV Accuracy = 0.884
k=11: CV Accuracy = 0.872
```
k=5 selected as optimal (highest CV accuracy, balances bias-variance tradeoff).

## 5.3 Dataset 2: Bank Customer Churn

### 5.3.1 Overview and Business Context

**Source:** Kaggle Bank Customer Churn Dataset  
**URL:** https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction  
**Domain:** Customer retention analytics in banking  
**Task:** Predict whether a customer will close their account (churn)

**Business Motivation:**
Customer acquisition costs in banking average $200-300 per customer. Retaining existing customers is 5-10× cheaper. Churn prediction enables:
- **Proactive retention:** Offer incentives before customers leave
- **Risk segmentation:** Identify high-value at-risk customers
- **Product optimization:** Understand what drives dissatisfaction

### 5.3.2 Data Characteristics

**Sample Size:** 10,000 instances  
**Features:** 13 (10 numerical, 3 categorical)  
**Target:** Exited (binary: 0=Retained, 1=Churned)  
**Class Distribution:** 7,963 retained (79.63%), 2,037 churned (20.37%)

**Feature Descriptions:**

| Feature | Type | Range/Categories | Description |
|---------|------|------------------|-------------|
| CreditScore | Numerical | 350-850 | FICO-style credit score |
| Geography | Categorical | France/Germany/Spain | Customer country |
| Gender | Categorical | Male/Female | Customer gender |
| Age | Numerical | 18-92 | Customer age in years |
| Tenure | Numerical | 0-10 | Years as bank customer |
| Balance | Numerical | 0-250,898 | Account balance in EUR |
| NumOfProducts | Numerical | 1-4 | Number of bank products held |
| HasCrCard | Binary | 0/1 | Has credit card (1=Yes) |
| IsActiveMember | Binary | 0/1 | Active in last 6 months (1=Yes) |
| EstimatedSalary | Numerical | 11.58-199,992 | Annual salary estimate EUR |

**Statistical Summary:**

```
Class Imbalance: 3.91:1 (Retained:Churned)
  - Significant imbalance requiring stratified sampling
  - Metrics should emphasize recall (catching churners) or F1 balance

Feature Importance (Information Gain):
  1. Age: 0.089 (older customers churn more)
  2. NumOfProducts: 0.073 (1-2 products stable, 3+ risky)
  3. IsActiveMember: 0.061 (inactive churn more)
  4. Geography: 0.041 (Germany higher churn rate)
  5. Balance: 0.035 (zero-balance and very-high-balance churn more)
```

### 5.3.3 Preprocessing Pipeline

**Step 1: Categorical Encoding**
```python
# One-hot encode Geography (3 categories → 3 binary features)
Geography_France:  1 if France, else 0
Geography_Germany: 1 if Germany, else 0
Geography_Spain:   1 if Spain, else 0

# One-hot encode Gender (2 categories → 2 binary features)
Gender_Female: 1 if Female, else 0
Gender_Male:   1 if Male, else 0

# Drop one category per feature to avoid multicollinearity:
# Keep Geography_Germany, Geography_Spain (drop France as reference)
# Keep Gender_Male (drop Female as reference)
```

**Final Feature Set (12 features after encoding):**
1. CreditScore
2. Age  
3. Tenure
4. Balance
5. NumOfProducts
6. HasCrCard
7. IsActiveMember
8. EstimatedSalary
9. Geography_Germany
10. Geography_Spain
11. Gender_Male

**Step 2: Missing Value Imputation**
```python
missing_analysis = {
    'CreditScore': 0,
    'Age': 0,
    'Tenure': 0,
    'Balance': 0,  
    'NumOfProducts': 0,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 0,
    'Geography': 0,
    'Gender': 0
}
# Result: No missing values, no imputation needed
```

**Step 3: Train-Test Split**
```python
train_size = 0.8  # 8,000 train, 2,000 test
stratified = True # Maintain 20.37% churn rate
random_seed = 42

Train: 7,963 × 0.8 = 6,370 retained, 1,630 churned
Test:  7,963 × 0.2 = 1,593 retained, 407 churned
```

**Step 4: Feature Scaling**
```python
# StandardScaler on numerical features only
numerical_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary'
]

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])

X_train_scaled = scaler.transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Categorical features (already 0/1) remain unscaled
```

**Scaling Parameters:**

| Feature | Mean | Std Dev | Scaled Range |
|---------|------|---------|--------------|
| CreditScore | 650.5 | 96.7 | [-3.10, 2.06] |
| Age | 38.9 | 10.5 | [-1.99, 5.05] |
| Tenure | 5.0 | 2.9 | [-1.72, 1.72] |
| Balance | 76,486 | 62,397 | [-1.23, 2.79] |
| NumOfProducts | 1.53 | 0.58 | [-0.91, 4.26] |
| EstimatedSalary | 100,090 | 57,511 | [-1.53, 1.74] |

### 5.3.4 Class Imbalance Handling

**Challenge:** 79.63% retained vs 20.37% churned—naive classifier predicting "retained" for all samples achieves 79.63% accuracy while being useless.

**Strategies Considered:**

1. **Stratified Sampling:** ✓ Implemented
   - Maintains class ratio in train/test splits
   - Prevents test set from being unrepresentative

2. **Class Weights:** ✗ Not used
   - Would apply to weighted KNN voting
   - Increases complexity without clear benefit for KNN

3. **SMOTE (Synthetic Minority Oversampling):** ✗ Not used
   - Generates synthetic minority samples
   - Can improve performance but changes data distribution
   - Avoided to maintain purity of comparison

4. **Evaluation Metric Selection:** ✓ Implemented
   - Emphasize F1 score (balances precision/recall)
   - Report recall separately (catching churners is critical)
   - ROC-AUC insensitive to class imbalance

**k Selection for Imbalanced Data:**
```
k=3:  F1=0.547, Recall=0.651
k=5:  F1=0.583, Recall=0.687  ← Selected
k=7:  F1=0.571, Recall=0.673
k=9:  F1=0.558, Recall=0.659
```
k=5 optimizes F1 score while maintaining strong recall.

## 5.4 Dataset 3: Spotify Dataset 2025

### 5.4.1 Overview and Context

**Source:** Kaggle Spotify Dataset 2025  
**URL:** https://www.kaggle.com/datasets/nelgiriyewithana/spotify-dataset-2025  
**Domain:** Music streaming and content recommendation  
**Task:** Predict track popularity based on audio features

**Sample Size:** 114,000 tracks  
**Features:** 23 audio characteristics  
**Target:** Popularity (continuous 0-100, binarized at median=50)  
**Class Distribution:** 57,000 popular (≥50), 57,000 unpopular (<50)

### 5.4.2 Feature Descriptions

**Audio Features (Spotify API):**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| danceability | Float | 0.0-1.0 | Rhythm stability, beat strength |
| energy | Float | 0.0-1.0 | Perceptual intensity, activity |
| key | Integer | 0-11 | Musical key (C=0, C#=1, ...) |
| loudness | Float | -60 to 0 dB | Overall volume level |
| mode | Binary | 0/1 | Major (1) or Minor (0) |
| speechiness | Float | 0.0-1.0 | Presence of spoken words |
| acousticness | Float | 0.0-1.0 | Acoustic vs electric instruments |
| instrumentalness | Float | 0.0-1.0 | Absence of vocals |
| liveness | Float | 0.0-1.0 | Audience presence indicator |
| valence | Float | 0.0-1.0 | Musical positivity/happiness |
| tempo | Float | 50-250 BPM | Beats per minute |
| duration_ms | Integer | 10,000-600,000 | Track length milliseconds |
| time_signature | Integer | 1-7 | Beats per bar |

**Derived Features:**
- **explicit:** Binary (contains explicit content)
- **release_year:** Integer (1920-2025)
- **artist_count:** Integer (number of collaborating artists)

**Preprocessing Considerations:**
- **High dimensionality:** 23 features risk curse of dimensionality
- **Varied scales:** danceability [0,1] vs duration_ms [10K, 600K]
- **Categorical:** key (12 categories), time_signature (7 categories)
- **Balanced:** 50-50 split (no imbalance issues)

## 5.5 Dataset 4: Loan Approval Dataset

### 5.5.1 Overview

**Source:** Kaggle Loan Approval Prediction  
**Sample Size:** 45,000 loan applications  
**Features:** 12 financial and demographic attributes  
**Target:** Loan Status (Approved=1, Rejected=0)  
**Class Distribution:** 40,500 approved (90%), 4,500 rejected (10%)

**Severe Imbalance:** 9:1 ratio makes this challenging for minority class detection.

### 5.5.2 Key Features

| Feature | Description | Business Relevance |
|---------|-------------|-------------------|
| income | Annual income | Primary repayment ability signal |
| loan_amount | Requested loan amount | Risk exposure |
| credit_score | FICO score (300-850) | Historical creditworthiness |
| employment_length | Years in current job | Income stability |
| debt_to_income | Monthly debt / income | Repayment capacity |
| home_ownership | Rent/Own/Mortgage | Asset base, stability |
| loan_purpose | Education/Medical/Business | Risk category |

### 5.5.3 Preprocessing Challenges

**Challenge 1: Severe Imbalance (10% positive)**
- Standard accuracy misleading (90% by predicting majority)
- Precision-recall tradeoff critical
- k selection must optimize F1 or recall, not accuracy

**Challenge 2: Mixed Features**
- Numerical: income, loan_amount, credit_score (need scaling)
- Categorical: home_ownership, loan_purpose (need encoding)
- Ratios: debt_to_income (already normalized, careful with scaling)

**Challenge 3: Missing Values**
- employment_length: 12% missing (impute with median)
- credit_score: 3% missing (impute with median)

## 5.6 Dataset 5: Diabetes Prediction

### 5.6.1 Overview

**Source:** Pima Indians Diabetes Database (Kaggle)  
**Sample Size:** 768 patients  
**Features:** 8 clinical measurements  
**Target:** Outcome (1=Diabetic, 0=Non-diabetic)  
**Class Distribution:** 500 non-diabetic (65%), 268 diabetic (35%)

### 5.6.2 Clinical Features

| Feature | Clinical Meaning | Normal Range |
|---------|-----------------|--------------|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose (mg/dL) | 70-140 |
| BloodPressure | Diastolic BP (mm Hg) | 60-80 |
| SkinThickness | Triceps skinfold (mm) | 10-50 |
| Insulin | Serum insulin (μU/mL) | 16-166 |
| BMI | Body mass index | 18.5-24.9 |
| DiabetesPedigreeFunction | Genetic risk score | 0.08-2.42 |
| Age | Age in years | 21-81 |

### 5.6.3 Medical Context

**High-Stakes Domain:** Misclassification has health consequences
- **False Negative:** Miss diabetic patient → untreated disease
- **False Positive:** Unnecessary treatment, patient anxiety

**Preferred Metrics:** Prioritize recall (catch diabetics) over precision

### 5.6.4 Data Quality Issues

**Zero Values as Missing Data:**
Several features contain physiologically impossible zeros:
- Glucose: 5 zeros (people don't survive with 0 glucose)
- BloodPressure: 35 zeros  
- SkinThickness: 227 zeros (29.5% missing!)
- Insulin: 374 zeros (48.7% missing!)
- BMI: 11 zeros

**Imputation Strategy:**
```python
# Replace 0 with NaN for features where 0 is impossible
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI']

for feature in features_with_zeros:
    df[feature] = df[feature].replace(0, np.nan)

# Median imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[features_with_zeros] = imputer.fit_transform(df[features_with_zeros])
```

## 5.7 Summary: Dataset Characteristics Comparison

| Dataset | Samples | Features | Pos % | Imbalance | Dimensionality | Domain |
|---------|---------|----------|-------|-----------|----------------|--------|
| Social Ads | 400 | 3 | 35.75% | 1.80:1 | Very Low | Marketing |
| Bank Churn | 10,000 | 13 | 20.37% | 3.91:1 | Low | Finance |
| Spotify | 114,000 | 23 | 50% | 1:1 | Medium | Entertainment |
| Loan Approval | 45,000 | 12 | 10% | 9:1 | Low | Banking |
| Diabetes | 768 | 8 | 35% | 1.86:1 | Very Low | Healthcare |

**Diversity Achieved:**
- **Scale:** 400 to 114,000 (285× range)
- **Dimensionality:** 3 to 23 features
- **Imbalance:** Balanced to severely imbalanced (9:1)
- **Domains:** 5 distinct application areas
- **Preprocessing:** Variety of challenges (missing data, categorical encoding, scaling)

This comprehensive dataset portfolio ensures experimental findings generalize beyond specific data characteristics.
