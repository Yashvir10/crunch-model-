# Startup Outcome & Funding Analysis

An end-to-end data science pipeline that turns raw Crunchbase-style startup data into actionable insight — spanning data cleaning, predictive modeling, and unsupervised learning to understand what drives startup success and funding behavior.

## Overview

This project explores three questions about startup outcomes:

1. **Will a startup survive, get acquired, or shut down?**
2. **How much total funding is a company likely to raise?**
3. **Are there natural groupings of startups based on funding behavior?**

Each question is tackled with a dedicated modeling approach, tied together by shared data cleaning and exploratory analysis.

## Methods

### 1. Classification — Predicting Startup Outcome
Predicts whether a startup is **operating, acquired, or closed**.
- **Models:** Logistic Regression, Random Forest
- **Class imbalance:** addressed with SMOTE (Synthetic Minority Oversampling)
- **Goal:** identify which company attributes are most predictive of survival vs. failure vs. acquisition

### 2. Regression — Estimating Total Funding Raised
Estimates a startup's total funding based on its attributes.
- **Models:** Linear Regression, Random Forest Regressor
- **Evaluation metrics:** RMSE, R²

### 3. Clustering — Uncovering Funding Profiles
Applies **KMeans clustering** to segment startups into distinct funding profiles, with the optimal number of clusters (three) selected via the elbow method.

## Pipeline

```
Raw Data → Cleaning & Preprocessing → EDA → Classification / Regression / Clustering → Insights
```

## Key Takeaways

- Combines supervised (classification, regression) and unsupervised (clustering) learning in one workflow
- Handles real-world messiness: missing values, class imbalance, mixed feature types
- Surfaces interpretable startup funding profiles alongside predictive performance

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, KMeans, SMOTE via imbalanced-learn)
- matplotlib / seaborn for visualization

## Getting Started

```bash
pip install -r requirements.txt
```

*(Add specific run instructions, e.g. notebook name or script entry point, once finalized.)*

## Results

*(Add key metrics, cluster descriptions, or top predictive features here once available.)*
