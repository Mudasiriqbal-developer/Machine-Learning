# Project 1: House Price Prediction (Ames Housing Dataset)

## 📌 Project Overview
This project predicts residential house prices using the Ames Housing dataset. It applies supervised machine learning regression algorithms—specifically **Multiple Linear Regression (Baseline)** and **Random Forest Regressor**—to predict property sale prices based on architectural, quality, temporal, spatial, and amenity features.

### Assignment Required Features:
1. **Overall Quality:** `OverallQual` (material and finish quality rated 1–10)
2. **Living Area:** `GrLivArea` (above-grade living area in sq ft, log-transformed)
3. **Basement Area:** `TotalBsmtSF` (total basement area in sq ft, log-transformed)
4. **Year Built:** `YearBuilt` (original construction year)
5. **Neighborhood:** `Neighborhood` (one-hot encoded spatial location)
6. **Garage Capacity:** `GarageCars` (size of garage in car capacity)
7. **Kitchen Quality:** `KitchenQual` (ordinal mapping: Ex=5, Gd=4, TA=3, Fa=2, Po=1)
8. **Basement Quality:** `BsmtQual` (ordinal mapping: Ex=5, Gd=4, TA=3, Fa=2, Po=1, missing=0 for no basement)
9. **Bathrooms:** `FullBath`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath` (missing filled with 0)
10. **Lot Area:** `LotArea` (lot size in sq ft, log-transformed)

Key evaluation metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Logarithmic Error (RMSLE)**
- **Coefficient of Determination (R²)**

The top-performing model (**Multiple Linear Regression**) and feature transformers are serialized using `joblib` and served through a clean, interactive **Streamlit** web application.

---

## 📁 Folder Structure

```text
House Price Prediction/
├── data/
│   └── raw/
│       ├── train.csv                      # Training dataset (1,460 rows, 81 columns)
│       ├── test.csv                       # Kaggle test dataset (1,459 rows, 80 columns)
│       ├── sample_submission.csv          # Benchmark submission template
│       └── data_description.txt           # Detailed feature definitions
├── models/
│   ├── best_house_price_model.joblib      # Serialized best model (Multiple Linear Regression)
│   ├── preprocessor.joblib                # Medians, mappings, skewed columns & encoded feature names
│   └── neighborhood_encoder.joblib        # Serialized neighborhood categorical list
├── outputs/
│   ├── eda_missing_values.png             # Missing value percentage chart
│   ├── eda_target_distribution.png        # Target distribution before/after log1p
│   ├── eda_grlivarea_vs_saleprice.png     # Living area vs. sale price scatter with outliers
│   ├── actual_vs_predicted.png            # Actual vs. predicted prices comparison
│   ├── residual_plot.png                  # Residual error distribution & homoscedasticity
│   └── submission_predictions.csv         # Final test predictions (1,459 records)
├── app.py                                 # Streamlit interactive web application
├── house_price_prediction.ipynb           # Fully executed end-to-end Jupyter notebook
├── README.md                              # Project documentation & execution guide
└── REPORT.md                              # Formal analytical report for oral evaluation
```

---

## 📊 Dataset Link & Location

- **Local Path:** `data/raw/` (`train.csv`, `test.csv`, `data_description.txt`)
- **Dataset Origin:** Ames Housing Dataset compiled by Dean De Cock (Truman State University).
- **Public Reference:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Dataset Properties:** 79 explanatory features across 1,460 residential transactions in Ames, Iowa (2006–2010).

---

## ⚙️ Installation & Setup

1. Navigate to the assignment folder:
   ```bash
   cd "D:/Machine Learning/ML Assingment Projects"
   ```

2. Activate virtual environment:
   ```bash
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Running the Jupyter Notebook
Open and execute all cells sequentially:
```bash
cd "House Price Prediction"
jupyter notebook house_price_prediction.ipynb
```

### 2. Running the Streamlit Prediction App
Launch the interactive web UI:
```bash
cd "House Price Prediction"
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📈 Evaluation Results & Comparison

| Evaluation Metric | Multiple Linear Regression | Random Forest Regressor (100 Trees) | Winner / Analysis |
| :--- | :---: | :---: | :--- |
| **Validation MSE** | **564,647,736.19** | 609,691,473.71 | **Linear Regression** (Lower error by ~45M) |
| **Validation RMSLE** | **0.1440** | 0.1534 | **Linear Regression** (Lowest logarithmic error: ~14.4%) |
| **Validation R²** | **0.8978 (89.8%)** | 0.8896 (89.0%) | **Linear Regression** explains more variance on unseen data |
| **Training R²** | **0.8968** | **0.9814** | **Random Forest overfits** (0.981 train vs 0.889 val) |
| **Status** | **Selected Best Model** | Candidate Comparison | Linear Regression generalizes without overfitting |

### Key Findings & Overfitting Diagnosis:
1. **Best Model Selection:** **Multiple Linear Regression** was selected as the best model and serialized to `models/best_house_price_model.joblib` because it achieved the **lowest Validation RMSLE (0.1440)**, lowest Validation MSE ($564.65 \times 10^6$), and highest Validation $R^2$ ($0.8978$).
2. **Overfitting Analysis:** 
   - **Random Forest Regressor** overfits the training partition: its training $R^2$ is **0.9814**, but drops to **0.8896** on the validation set (a ~9.2% generalization gap).
   - **Multiple Linear Regression** shows exceptional stability: its training $R^2$ is **0.8968** and its validation $R^2$ is **0.8978** (virtually zero generalization gap).
3. **Logarithmic Target Transformation (`np.log1p`):** Reduced target skewness from **+1.88** to **+0.12**, enabling the linear model to capture multiplicative price effects cleanly.
4. **Outlier Pruning:** Removing the two documented atypical properties (> 4,000 sq ft sold for unusually low prices) prevented regression line distortion.

---

## 👥 Reproducibility & Seed
All splits, shuffles, and stochastic estimators strictly adhere to `random_state=42`.
