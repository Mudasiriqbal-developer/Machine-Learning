# Project 1: House Price Prediction - Technical Report

**Course Assignment** | **Due Date:** 3 November 2026  
**Status:** In Progress (Structure & Baseline Scaffolded)  

---

## 1. Introduction
- **Problem Statement:** Accurately predicting the sale prices of residential homes using physical, temporal, and spatial property characteristics.
- **Dataset Context:** The Ames Housing dataset provides 79 explanatory features across 1,460 residential transactions in Ames, Iowa (2006–2010). Unlike the older Boston Housing dataset, Ames offers modernized variables, detailed categorical quality assessments, and extensive basement/garage specifications.
- **Evaluation Objectives:**
  - Establish a baseline with Multiple Linear Regression.
  - Implement an ensemble tree model (Random Forest Regressor) to capture non-linear interactions.
  - Optionally explore gradient boosting (XGBoost).
  - Assess predictions using Mean Squared Error (MSE) and Root Mean Squared Logarithmic Error (RMSLE).
  - Deploy the top-performing pipeline into an end-user application.

*[Placeholder: Add formal background, dataset citation, and specific problem boundaries during implementation.]*

---

## 2. Preprocessing / Methodology
- **Handling Missing Values:**
  - Imputation strategy for numerical columns (median imputation based on neighborhood or global median).
  - Categorical missing value imputation (e.g., distinguishing true missingness from 'None' indicating absence of amenities like pools, fireplaces, or alleys).
- **Distribution Transformation:**
  - Analysis of target variable `SalePrice` skewness. Application of logarithmic transformation (`log1p`) to conform to Gaussian error distribution assumptions.
  - Identification and transformation of right-skewed predictor features (`LotArea`, `GrLivArea`, `TotalBsmtSF`).
- **Feature Encoding:**
  - Ordinal encoding for quality variables with clear hierarchies (`Ex` > `Gd` > `TA` > `Fa` > `Po`).
  - One-hot encoding / Frequency encoding for nominal variables (`Neighborhood`, `HouseStyle`, `SaleType`).
- **Data Splitting & Cross-Validation:**
  - 80/20 train-validation split using `random_state=42` to guarantee strict reproducibility.

*[Placeholder: Document final feature set, exact imputation values, and feature correlation findings during implementation.]*

---

## 3. Model or Algorithm Comparison
- **Multiple Linear Regression (Baseline):**
  - Formulation: \( \hat{y} = \beta_0 + \sum_{i=1}^p \beta_i x_i \)
  - Strengths: Highly explainable coefficients, fast training.
  - Assumptions & Weaknesses: Assumes linear relationships, sensitive to multicollinearity and unhandled outliers.
- **Random Forest Regressor:**
  - Formulation: Bagging ensemble of \( B \) decision trees with randomized feature subsampling.
  - Strengths: Robust to outliers, models complex non-linear feature interactions without manual interaction terms.
  - Hyperparameters to track: `n_estimators=100`, `max_depth`, `min_samples_split`, `random_state=42`.
- **XGBoost Regressor (Optional / Candidate):**
  - Formulation: Gradient boosted tree ensemble optimizing pseudo-residuals sequentially.
  - Strengths: Regularization terms (\(L_1, L_2\)) to curb overfitting, high empirical accuracy.

*[Placeholder: Document selected hyperparameters, convergence behavior, and training times during implementation.]*

---

## 4. Evaluation Results
- **Quantitative Performance Table:**

| Metric | Linear Regression | Random Forest Regressor | XGBoost Regressor (Opt.) |
| :--- | :---: | :---: | :---: |
| **Validation MSE** | *[Pending]* | *[Pending]* | *[Pending]* |
| **Validation RMSLE** | *[Pending]* | *[Pending]* | *[Pending]* |
| **Training R²** | *[Pending]* | *[Pending]* | *[Pending]* |
| **Validation R²** | *[Pending]* | *[Pending]* | *[Pending]* |

- **Visual Diagnostics:**
  - Predicted vs Actual scatter plots (`outputs/actual_vs_predicted.png`).
  - Residual distribution and homoscedasticity inspection (`outputs/residual_plot.png`).
- **Feature Importance:**
  - Top predictive features identified (e.g., `OverallQual`, `GrLivArea`, `TotalBsmtSF`).

*[Placeholder: Populate exact metric values, error analysis tables, and diagnostic chart discussions during implementation.]*

---

## 5. Limitations
- **Geographic Specificity:** The Ames dataset reflects real estate dynamics of Ames, Iowa between 2006 and 2010 and does not generalize directly to modern or metropolitan real estate markets.
- **High Cardinality & Missingness:** Several attributes have high proportions of missing values (e.g., `PoolQC`, `MiscFeature`) requiring heuristic pruning or imputation.
- **Linearity Constraints:** Linear regression struggles with non-linear compounding value effects (e.g., luxury finishes that multiply overall property value).

*[Placeholder: Expand on model error edge cases, underpredicted luxury homes, and outlier impacts during implementation.]*

---

## 6. Improvements
- **Advanced Ensembling / Stacking:** Blending predictions from Ridge regression, Random Forest, and Gradient Boosting.
- **Systematic Hyperparameter Tuning:** Running Bayesian optimization or GridSearchCV over tree depth and learning rate.
- **Target Encoding & Domain Features:** Engineering domain-specific composite metrics (e.g., `TotalLivingSpace = GrLivArea + TotalBsmtSF`, `TotalBaths = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`).

*[Placeholder: Document experimental validation of potential improvements during implementation.]*
