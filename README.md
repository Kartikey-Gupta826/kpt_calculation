# TrueKPT: Machine Learning Pipeline for Accurate Kitchen Preparation Time Prediction

## Overview

Food delivery platforms rely heavily on **Kitchen Preparation Time (KPT)** predictions to schedule rider dispatch and reduce wait times. In many systems, restaurants manually mark **Food Order Ready Time (FOR)**, which is often unreliable due to delayed marking or operational bias.

This project builds a **multi-stage machine learning pipeline** to estimate the **true preparation time (TrueKPT)** using operational signals from restaurants and order dynamics.

The system progressively improves prediction accuracy through:

* Baseline ML modeling
* Operational signal engineering
* Merchant reliability scoring
* Sequential kitchen congestion modeling
* Time-aware recency weighting

The final system reduces prediction error significantly compared to raw merchant signals.

---

# Project Pipeline

The model architecture follows a **two-stage prediction pipeline**.

```
Restaurant Signals
        │
        ▼
Stage 1: Clean FOR Prediction
(Random Forest + Feature Engineering)
        │
        ▼
Predicted Clean Preparation Time
        │
        ▼
Stage 2: TrueKPT Prediction
(Random Forest + Enhanced Signals)
        │
        ▼
Final Kitchen Preparation Time Prediction
```

---

# Models Implemented

The project compares multiple approaches to quantify improvements.

## 1. Raw Merchant Prediction

Uses the restaurant provided **merchant_FOR_time** directly.

This represents the **current operational baseline**.

MAE: **3.50 minutes**

---

## 2. Baseline Machine Learning Model

Features used:

* merchant_FOR_time
* food_item
* order_hour
* peak_hour

Model:

* Random Forest Regressor

MAE: **0.59 minutes**

---

## 3. Advanced Model

Adds operational signals:

* total_active_orders
* competitor_load
* merchant_intelligence_score
* restaurant_avg_prep
* restaurant_std_prep

Pipeline:

1. Predict **clean FOR**
2. Use prediction to estimate **true KPT**

MAE: **0.31 minutes**

---

## 4. Enhanced Model (TrueKPT)

Introduces **new operational signals** to model restaurant behavior more accurately.

### New Signals

#### 1. FOR Reliability Score

Measures how reliable a restaurant’s FOR signal is.

Penalizes:

* delayed marking
* large deviation from true KPT
* inconsistent reporting

#### 2. Sequential Order Clustering

Captures kitchen congestion.

Uses:

* active orders
* orders within the same hour

#### 3. Acceptance Proxy Score

Measures merchant attentiveness by comparing observed KPT vs average preparation time.

#### 4. Enhanced Merchant Intelligence Score

Weighted combination:

```
Enhanced Score =
0.25 * merchant_intelligence_score
0.35 * FOR reliability score
0.20 * clustering score
0.20 * acceptance proxy score
```

---

## 5. Time-Aware Model

Adds temporal dynamics.

New features:

* order_weekday
* order_minute
* recency_weight

Recency weighting gives **more importance to recent orders**, simulating kitchen queue priority.

---

# Final Model Results

| Model              | MAE       | P50       | P90       |
| ------------------ | --------- | --------- | --------- |
| Raw Merchant FOR   | 3.503     | 3.464     | 5.629     |
| Baseline ML        | 0.593     | 0.428     | 1.249     |
| Advanced ML        | 0.314     | 0.144     | 0.858     |
| TrueKPT (Enhanced) | **0.312** | **0.139** | **0.839** |

### Improvements

* **91% reduction in MAE vs raw merchant signals**
* **85% reduction in P90 error**
* Significant reduction in prediction variance

---

# Rider Wait Time Simulation

The model also simulates rider arrival timing.

Dispatch buffer: **1.5 minutes**

| Model            | Avg Rider Wait |
| ---------------- | -------------- |
| Raw Merchant FOR | 0.00 min       |
| Baseline ML      | 0.025 min      |
| TrueKPT Model    | **0.015 min**  |

This indicates **better alignment between rider arrival and food readiness**.

---

# Feature Importance

The most influential feature is:

```
predicted_enhanced_FOR
```

This validates the **two-stage prediction pipeline**, where accurate preparation estimation drives final prediction quality.

Other contributing signals:

* enhanced_merchant_score
* for_reliability_score
* clustering_score
* recency_weight

---

# Visualizations Generated

The project automatically produces **9 analytical visualizations**:

1. FOR Bias Distribution
2. MAE / P50 / P90 Error Comparison
3. Error Distribution Curves
4. Prediction Error by Hour of Day
5. Feature Importance
6. Merchant Reliability vs Error
7. Rider Wait Time Simulation
8. Recency Weight Impact
9. Merchant Intelligence Score Components

These help explain the behavior of the model and validate feature design.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

Models:

* Random Forest Regressor
* Feature Engineering Pipelines
* Two-stage prediction architecture

---

# Key Contributions

This project demonstrates:

* Building **multi-stage ML pipelines**
* Designing **operational intelligence signals**
* Improving **prediction reliability for logistics systems**
* Modeling **restaurant behavior and kitchen congestion**

The system shows how **domain-aware feature engineering** can significantly improve real-world ML prediction performance.

---

# Running the Project

Install dependencies:

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

Run the notebook or script containing the training pipeline.

The script will:

1. Train all models
2. Generate prediction metrics
3. Produce visualization plots
4. Save comparison figures

---
