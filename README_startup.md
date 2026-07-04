# Startup Crunch — Funding & Status Analysis

Exploratory data analysis and machine learning on a Crunchbase-style startup funding dataset. The project covers data cleaning, EDA, classification, regression, and clustering.

## 📊 Overview

This notebook analyzes startup company data (funding rounds, status, category, region, etc.) to answer:
- **Classification** — Can we predict a startup's `status` (operating / acquired / closed) from its features?
- **Regression** — Can we predict `funding_total_usd` from company attributes?
- **Clustering** — Do startups group into natural funding-behavior segments (KMeans)?

## 🗂️ Project Structure

```
startupcrunch/
├── startupcrunch.ipynb     # Main analysis notebook
├── data/
│   └── README.md           # Notes on dataset source (data not committed)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## 🧪 Workflow

1. **Data Cleaning** — handled nulls, duplicates, encoded categorical variables with `LabelEncoder`
2. **EDA** — histograms, count plots, correlation heatmap
3. **Classification** — Logistic Regression & Random Forest to predict `status`, with SMOTE for class imbalance, evaluated via precision/recall/F1
4. **Regression** — Linear Regression & Random Forest Regressor to predict `funding_total_usd`, evaluated via RMSE and R²
5. **Clustering** — KMeans (elbow method to choose k) to segment startups by funding profile

## 🛠️ Tools & Libraries

`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn` (SMOTE)

## ▶️ Running Locally

```bash
git clone https://github.com/<Yashvir10>/startupcrunch.git
cd startupcrunch
pip install -r requirements.txt
jupyter notebook startupcrunch.ipynb

```This project analyzes a Crunchbase-style startup dataset to explore funding patterns and outcomes. After cleaning the data (handling nulls, encoding categorical features), it builds three types of models: classification models (Logistic Regression, Random Forest) to predict whether a startup is operating, acquired, or closed; regression models (Linear Regression, Random Forest Regressor) to predict total funding raised; and a KMeans clustering model to segment startups into distinct funding-behavior groups. The notebook includes exploratory visualizations (histograms, correlation heatmaps) and evaluates models using precision/recall/F1 for classification and RMSE/R² for regression
