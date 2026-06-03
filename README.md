# Breast Cancer Classification — SVM + Bagging Ensemble

A complete machine learning pipeline to classify breast cancer tumors as **Malignant** or **Benign** using an ensemble of Support Vector Machines with exhaustive hyperparameter tuning.

---

## Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate classification of tumors as malignant or benign is critical for treatment decisions. This project builds a high-accuracy classifier on 30 clinical features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

---

## Dataset

- **Source:** Breast Cancer Wisconsin Dataset
- **Samples:** 569 patients
- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Diagnosis — Malignant (M → 1) or Benign (B → 0)
- **Class distribution:** ~37% Malignant, ~63% Benign

---

## What This Project Covers

### 1. Exploratory Data Analysis (EDA)
- Inspected data types, null values, and feature distributions
- Removed irrelevant columns (`id`, unnamed index columns)
- Encoded target variable: `M → 1`, `B → 0`
- Visualized all 30 feature distributions using histograms
- Detected outliers across all features using boxplots
- Computed feature correlation matrix to identify multicollinearity

### 2. Key Insights from EDA
- Features like `radius_mean`, `perimeter_mean`, and `area_mean` are highly correlated — expected since they measure size
- Malignant tumors consistently show higher values in size-related features
- Several features contain significant outliers, particularly `area_worst` and `perimeter_worst`
- The dataset is moderately imbalanced but manageable without resampling

### 3. Preprocessing
- Dropped `id` column (non-informative identifier)
- Split data: 75% training / 25% test (`random_state=42`)
- Applied `StandardScaler` — fit on training data only, transformed both sets to prevent data leakage

### 4. Model — SVM with Bagging Ensemble
Instead of a single SVM, a **BaggingClassifier** wrapping an **SVC** was used:
- Bagging trains multiple SVMs on random subsets of data and features
- Reduces variance and improves generalization over a single SVM
- Each base estimator sees a different view of the data, making the ensemble more robust

### 5. Hyperparameter Tuning — GridSearchCV
Exhaustive search over:

| Parameter | Values Tried |
|-----------|-------------|
| `n_estimators` | 10, 25 |
| `max_samples` | 0.6, 0.8 |
| `max_features` | 0.6, 0.8 |
| `estimator__C` | 0.1, 1, 10 |
| `estimator__kernel` | RBF |
| `estimator__gamma` | scale, auto |

- 4-fold cross-validation on all combinations
- Scored by accuracy, selected best configuration automatically

### 6. Evaluation
- Test accuracy reported
- Full classification report (Precision, Recall, F1-score per class)
- Confusion matrix to visualize true/false positives and negatives

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Preprocessing, modeling, evaluation |
| SVC | Base classifier |
| BaggingClassifier | Ensemble method |
| GridSearchCV | Hyperparameter tuning |

---

## How to Run

1. Clone the repo
```bash
git clone https://github.com/Nour-saber57/breast-cancer-svm-bagging
cd breast-cancer-svm-bagging
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Load the dataset — replace the CSV path with either:
   - Download the [Breast Cancer Wisconsin dataset from Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
   - Or load directly from sklearn:
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
```

4. Run the notebook in Jupyter or Google Colab

---

## Project Structure

```
breast-cancer-svm-bagging/
│
├── breast_cancer_prediction.py   # Full pipeline: EDA → preprocessing → model → evaluation
└── README.md
```

---

## Author

**Nour Mohammed**  
AI & Machine Learning Engineer | Computer Engineering Student  
[LinkedIn](https://www.linkedin.com/in/nour-mohammed-06341a204/) | [GitHub](https://github.com/Nour-saber57)
