# TrueKPT Prediction System

**A multi-stage machine learning pipeline for predicting restaurant Kitchen Preparation Time (KPT) using merchant intelligence, operational signals, and temporal features.**

## Overview

Food delivery platforms depend on accurate Kitchen Preparation Time (KPT) estimates to optimize rider dispatch and minimize wait times.

Many systems rely on a merchant-provided **Food Order Ready (FOR)** signal, which is often noisy due to:

* Delayed readiness marking
* Operational inconsistencies
* Kitchen congestion
* Merchant-specific behavior

This project develops a **two-stage machine learning pipeline** that estimates the true preparation time (**TrueKPT**) by combining restaurant behavior signals, operational workload metrics, and temporal features.

---

## Pipeline Architecture

```text
Restaurant Signals
        │
        ▼
Enhanced Merchant Intelligence
        │
        ├── FOR Reliability Score
        ├── Kitchen Clustering Score
        ├── Acceptance Proxy Score
        ▼
Enhanced Merchant Score
        │
        ▼
Stage 1: Enhanced FOR Predictor
(Random Forest)
        │
        ▼
Predicted Enhanced FOR
        │
        ▼
Stage 2: TrueKPT Predictor
(Random Forest + Time Features)
        │
        ▼
Final Kitchen Preparation Time
```

---

## Model Evolution

The project progressively improves prediction quality through multiple modeling stages.

### 1. Raw Merchant FOR

Uses the merchant-provided readiness estimate directly.

**Features**

* merchant_FOR_time

**Purpose**

Operational benchmark.

---

### 2. Baseline Model

Random Forest model using basic restaurant context.

**Features**

* merchant_FOR_time
* food_item
* order_hour
* peak_hour

---

### 3. Advanced Model

Introduces operational workload signals.

**Additional Features**

* total_active_orders
* competitor_load
* merchant_intelligence_score
* restaurant_avg_prep
* restaurant_std_prep

**Architecture**

```text
Stage 1
Predict Clean FOR
        ↓
Stage 2
Predict True KPT
```

---

### 4. Enhanced Model

Introduces merchant behavior intelligence.

#### FOR Reliability Score

Measures historical accuracy of merchant FOR reporting.

Captures:

* reporting bias
* delayed readiness marking
* consistency

#### Kitchen Clustering Score

Models kitchen congestion using:

* active orders
* orders within the same hour

#### Acceptance Proxy Score

Measures merchant attentiveness and operational consistency.

#### Enhanced Merchant Score

```text
0.25 × Merchant Intelligence Score
0.35 × FOR Reliability Score
0.20 × Clustering Score
0.20 × Acceptance Proxy Score
```

---

### 5. TrueKPT Final Model

Final production-style prediction system.

#### Additional Features

**Temporal Features**

* order_weekday
* order_minute

**Recency Weighting**

Recent orders receive greater influence during training to simulate changing kitchen conditions and queue dynamics.

#### Final Architecture

```text
Enhanced Signals
        ↓
Stage 1 RF
        ↓
Predicted Enhanced FOR
        ↓
Time Features
        ↓
Recency-Weighted RF
        ↓
TrueKPT Prediction
```

---

## Results

| Model            | MAE (min) | P50 (min) | P90 (min) |
| ---------------- | --------: | --------: | --------: |
| Raw Merchant FOR |     3.503 |     3.464 |     5.629 |
| Baseline ML      |     0.593 |     0.428 |     1.249 |
| Advanced ML      |     0.314 |     0.144 |     0.858 |
| Enhanced ML      |     0.312 |     0.139 |     0.839 |
| TrueKPT Final    |     0.312 |     0.139 |     0.839 |

---

## Key Improvements

Compared to raw merchant readiness estimates:

* ~91% reduction in MAE
* Significant reduction in P90 error
* More stable prediction performance
* Better modeling of merchant behavior
* Improved handling of operational congestion

---

## Feature Engineering Highlights

### Merchant Intelligence Signals

* FOR Reliability Score
* Acceptance Proxy Score
* Enhanced Merchant Score

### Operational Signals

* Total Active Orders
* Competitor Load
* Orders This Hour
* Kitchen Clustering Score

### Temporal Signals

* Order Weekday
* Order Minute
* Recency Weight

---

## Technologies Used

### Core Libraries

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

### Models

* Random Forest Regressor
* Feature Engineering Pipelines
* Two-Stage Prediction Architecture

---

## Repository Structure

```text
truekpt-prediction/
│
├── README.md
├── requirements.txt
├── TrueKPT_Prediction_System.ipynb
│
├── data/
│   └── sample_data.csv
│
└── images/
    ├── mae_comparison.png
    └── feature_importance.png
```

---

## Running the Project

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Open:

```text
TrueKPT_Prediction_System.ipynb
```

Run all notebook cells sequentially.

The notebook will:

* Train all model variants
* Generate prediction metrics
* Produce comparison tables
* Generate feature importance visualizations

---

## Future Improvements

* Gradient Boosting / XGBoost comparison
* Real timestamp-based temporal features
* Cross-validation evaluation
* Online learning for continuously changing restaurant behavior
* Deployment as a real-time prediction API
