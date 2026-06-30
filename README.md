# Iris Species Classification

A supervised machine learning project that classifies Iris flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements, comparing three classification algorithms.

## Problem Statement

A botanical research centre currently identifies Iris species manually by measuring sepal and petal dimensions — a process that is time-consuming, error-prone, and not scalable. This project builds and compares machine learning classifiers to automate species identification.

## Dataset

- **Source:** Iris.csv (150 samples, 3 balanced classes — 50 samples each)
- **Features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species (Iris-setosa, Iris-versicolor, Iris-virginica)

No missing values were found in the dataset.

## Methodology

To avoid the unrealistically perfect scores typical of this dataset, models were trained on only **50% of the data** and tested on the **full 100%** (150 samples), rather than a standard train/test split.

Steps:
1. Split data (50% train, stratified by species)
2. Scale features using `StandardScaler`
3. Train and evaluate three classifiers
4. Compare results using accuracy, precision, recall, F1-score, and confusion matrices

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model, `C=1.0`, `lbfgs` solver |
| Naive Bayes | Gaussian Naive Bayes |
| K-Nearest Neighbours | Tuned via `GridSearchCV` (5-fold CV) over `k = [3, 5, 7, 9]`; built using a `Pipeline` (scaler + KNN) |

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.973 | 0.960 |
| Naive Bayes | 0.973 | 0.960 |
| KNN (best k=5) | — | 0.960 |

All three models achieved the same test accuracy (0.96). Errors were consistently concentrated between Iris-versicolor and Iris-virginica, while Iris-setosa was classified perfectly across all models — consistent with the known overlap in petal/sepal measurements between those two species. Logistic Regression and Naive Bayes produced identical predictions on every test sample.

## Best Model

**Logistic Regression** is recommended as the best overall choice. While all three models tied on accuracy, Logistic Regression offers better interpretability, lower computational cost, and simpler deployment compared to Naive Bayes (which relies on a feature-independence assumption that doesn't hold for these measurements) and KNN (which requires storing the full training set and is slower at prediction time).

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
2. Place `Iris.csv` in the project directory
3. Run the notebook cells in order

## Author

Sam — Software Development student, Haaga-Helia University of Applied Sciences
