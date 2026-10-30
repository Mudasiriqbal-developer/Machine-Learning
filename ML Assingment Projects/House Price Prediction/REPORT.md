# Project 1: House Price Prediction - Technical Report

**Course Assignment** | **Due Date:** 3 November 2026  
**Status:** Completed & Evaluated (Final Pipeline)  
**Random State:** `42` (Strict Reproducibility Across All Modules)

---

## 1. Introduction
- **Problem Statement:** The objective of this project is to develop, evaluate, and compare supervised regression models to predict the final transaction sale prices (`SalePrice`) of residential properties based on architectural, dimensional, temporal, geographic, and amenity features.
- **Dataset Context:** We utilized the Ames Housing dataset compiled by Dean De Cock (Truman State University). The dataset records 1,460 residential home transactions in Ames, Iowa between 2006 and 2010. It contains 79 explanatory features detailing structural attributes, build quality, basement specifications, garage capacity, and neighborhood geography.
- **Evaluation Objectives:**
  - Establish a transparent linear baseline via **Multiple Linear Regression**.
  - Implement an ensemble tree model (**Random Forest Regressor**) to capture non-linear relationships.
  - Evaluate both models on an out-of-fold validation set using **Mean Squared Error (MSE)** and **Root Mean Squared Logarithmic Error (RMSLE)**.
  - Identify and serialize the model with the lowest validation RMSLE via `joblib`.
  - Deploy the top-performing pipeline into an end-user **Streamlit** web application.

---

## 2. Preprocessing & Methodology

### 2.1 Outlier Mitigation
Following the original Ames dataset author recommendations, two extreme outliers were identified during Exploratory Data Analysis: properties with `GrLivArea > 4,000` sq ft that sold for under $300,000 (abnormal commercial/partial sales). These instances were pruned from the training set, reducing training records from 1,460 to 1,458 and preventing distorted regression slopes.

### 2.2 Target Transformation
- The original target variable `SalePrice` was heavily right-skewed with a skewness coefficient of **+1.88**.
- We applied natural logarithmic transformation via `np.log1p(SalePrice)`.
- The transformed target demonstrated a skewness coefficient of **+0.12**, conforming closely to a normal (Gaussian) distribution and stabilizing regression variance.

### 2.3 Feature Selection & Quality Mappings
In accordance with assignment requirements, the pipeline utilizes the following 10 feature groups:
1. **Overall Quality (`OverallQual`):** Rating of overall material and finish (1 to 10).
2. **Living Area (`GrLivArea`):** Above-ground finished living area in sq ft.
3. **Basement Area (`TotalBsmtSF`):** Total square feet of basement area.
4. **Year Built (`YearBuilt`):** Original construction year.
5. **Neighborhood (`Neighborhood`):** 25 distinct spatial locations in Ames, Iowa.
6. **Garage Capacity (`GarageCars`):** Size of garage in car capacity.
7. **Kitchen Quality (`KitchenQual`):** Ordinal mapping applied:
   - `Ex` (Excellent) = 5
   - `Gd` (Good) = 4
   - `TA` (Typical/Average) = 3
   - `Fa` (Fair) = 2
   - `Po` (Poor) = 1
8. **Basement Quality (`BsmtQual`):** Ordinal mapping applied:
   - `Ex` = 5, `Gd` = 4, `TA` = 3, `Fa` = 2, `Po` = 1
   - Missing values filled with `0` (indicating absence of a basement).
9. **Bathrooms:**
   - `FullBath`: Full bathrooms above grade.
   - `HalfBath`: Half bathrooms above grade (missing filled with 0).
   - `BsmtFullBath`: Basement full bathrooms (missing filled with 0).
   - `BsmtHalfBath`: Basement half bathrooms (missing filled with 0).
10. **Lot Area (`LotArea`):** Lot size in sq ft.

### 2.4 Handling Missing Values
- Missing numerical values in both training and test partitions were imputed using median values computed exclusively from the training set (e.g., `TotalBsmtSF` median = 990 sq ft, `GarageCars` median = 2).
- Bathrooms and `BsmtQual` were imputed with `0` where amenities are not present.
- This strictly avoided data leakage from validation and test sets.

### 2.5 Distribution Transformation on Predictors
Continuous spatial features with long positive tails (`GrLivArea`, `LotArea`, `TotalBsmtSF`) were transformed using `np.log1p()` to compress scale disparities and linearize relationships.

### 2.6 Categorical Encoding
- The `Neighborhood` variable was encoded using one-hot dummy encoding (`pd.get_dummies(..., drop_first=True)`).
- The training and test feature matrices were aligned to guarantee identical feature schema across all splits (36 engineered features total).

### 2.7 Validation Split
The preprocessed training set was partitioned into an 80% training split (1,166 samples) and a 20% validation split (292 samples) using `random_state=42`.

---

## 3. Model Comparison & Training

### 3.1 Multiple Linear Regression (Baseline)
- **Formulation:**  
  $$\hat{y}_{log} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
- **Characteristics:** Highly transparent, computationally efficient, and stable. With log-transformed target and continuous features, linear coefficients model exponential percentage-multiplier price effects naturally.

### 3.2 Random Forest Regressor (Ensemble Bagging)
- **Formulation:** An ensemble of $B = 100$ decorrelated decision trees:  
  $$\hat{y}_{log} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)$$
- **Hyperparameters:** `n_estimators=100`, `random_state=42`, `n_jobs=1`.
- **Characteristics:** Non-linear tree ensemble capable of capturing high-order feature interactions, but prone to overfitting without depth regularization.

---

## 4. Evaluation Results

Both models were evaluated on the held-out validation set ($N = 292$). Predicted values in log space were converted back to actual currency scale using `np.expm1()` before computing MSE and RMSLE.

### 4.1 Quantitative Performance Table

| Evaluation Metric | Multiple Linear Regression | Random Forest Regressor | Performance Verdict |
| :--- | :---: | :---: | :--- |
| **Validation MSE** | **564,647,736.19** | 609,691,473.71 | **Linear Regression wins** (lower error by ~45M) |
| **Validation RMSLE** | **0.1440** | 0.1534 | **Linear Regression wins** (~14.4% vs 15.3% error) |
| **Validation R²** | **0.8978 (89.8%)** | 0.8896 (89.0%) | **Linear Regression wins** (higher variance explained) |
| **Training R²** | **0.8968** | **0.9814** | **Random Forest overfits significantly** |

### 4.2 Overfitting Diagnosis
A critical analytical finding in this project is the comparative generalization behavior of the two algorithms:
- **Random Forest Regressor displays pronounced overfitting:**
  - On the training data, Random Forest reaches a near-perfect $R^2$ of **0.9814**.
  - However, on unseen validation data, its $R^2$ drops to **0.8896** (a **9.18% drop** in performance). The unconstrained decision trees memorize specific training feature quirks that fail to generalize out-of-fold.
- **Multiple Linear Regression demonstrates remarkable generalization stability:**
  - Training $R^2$: **0.8968**
  - Validation $R^2$: **0.8978**
  - The generalization gap is virtually zero (**+0.10%**), proving that the linear model captures the true underlying economic relationships without memorizing training noise.

### 4.3 Best Model Selection
In strict accordance with the objective evaluation metric (**lowest Validation RMSLE**):
- **Multiple Linear Regression** was programmatically selected as the **BEST model** ($RMSLE = 0.1440$, $MSE = 564.65 \times 10^6$, $R^2 = 0.8978$).
- The model was serialized to [`models/best_house_price_model.joblib`](file:///d:/Machine%20Learning/ML%20Assingment%20Projects/House%20Price%20Prediction/models/best_house_price_model.joblib).

### 4.4 Diagnostic Visual Analysis
All diagnostic evaluation figures are generated and stored in `outputs/`:
- **Actual vs. Predicted Plot (`outputs/actual_vs_predicted.png`):** Both models track the 45-degree ideal fit line closely across typical price points ($100k to $350k). Linear Regression maintains tighter alignment across mid and upper-mid price tiers.
- **Residual Distribution Plot (`outputs/residual_plot.png`):** Residuals for Linear Regression are symmetrically distributed around zero with balanced variance across predicted price tiers, demonstrating homoscedastic behavior.
- **Target Distribution Plot (`outputs/eda_target_distribution.png`):** Confirms successful normalization from +1.88 to +0.12 skewness.

---

## 5. Artifact Serialization & Deployment

1. **Serialized Model & Artifacts:**
   - Best model saved to: `models/best_house_price_model.joblib` (Multiple Linear Regression).
   - Preprocessing parameters (medians, quality mappings, bathroom columns, skewed column lists, feature column schemas) saved to: `models/preprocessor.joblib`.
   - Neighborhood list saved to: `models/neighborhood_encoder.joblib`.
2. **Batch Test Predictions:**
   - Generated predictions for all 1,459 instances in `data/raw/test.csv` using the serialized best model.
   - Exported to: `outputs/submission_predictions.csv` with zero missing values (Min: $48,637.75, Median: $156,243.58, Max: $620,233.02).
3. **Streamlit Web Application (`app.py`):**
   - Interactive UI with inputs for all 10 feature groups: Overall Quality, Kitchen Quality, Basement Quality, Year Built, Living Area, Basement Area, Lot Size, Garage Capacity, Bathrooms (Full, Half, Bsmt Full, Bsmt Half), and Neighborhood.
   - Automatically executes the preprocessing transform and computes the estimated valuation along with an expected confidence interval ($\pm 14.4\%$).

---

## 6. Limitations

1. **Random Forest Complexity:** Without tuning `max_depth` or `min_samples_leaf`, default Random Forest ensembles overfit the training data ($R^2 = 0.981$ vs $0.890$), limiting their out-of-fold generalization compared to regularized or linear models.
2. **Geographic Specificity:** Data reflects Ames, Iowa real estate between 2006 and 2010 and does not capture macroeconomic inflation or modern urban housing markets.
3. **Upper-End Luxury Variance:** Extreme luxury properties (over $450k) exhibit idiosyncratic features (e.g. custom architectural finishes) that carry higher prediction variance.

---

## 7. Future Improvements

1. **Random Forest Regularization:** Implementing cross-validated hyperparameter tuning over `max_depth` (e.g., 8–15), `min_samples_leaf` (e.g., 2–5), and `max_features` to curb tree overfitting.
2. **Regularized Linear Models (Ridge & Lasso):** Applying $L_1$ and $L_2$ regularization to eliminate any remaining multicollinearity among spatial features.
3. **Composite Living Metric:** Formulating aggregate features such as Total Usable SF (`GrLivArea + TotalBsmtSF`) and Total Bath equivalents.
