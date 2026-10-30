# Project 1: House Price Prediction (Ames Housing Dataset)

## 📌 Project Overview
This project tackles residential house price prediction using the well-known Ames Housing dataset. It applies supervised regression techniques—specifically **Multiple Linear Regression** and **Random Forest Regressor** (with optional **XGBoost Regressor**)—to estimate the final sale price of properties based on architectural, quality, and spatial features.

Key evaluation metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Logarithmic Error (RMSLE)**

The top-performing model and feature transformations are serialized using `joblib` and served through a lightweight, interactive **Streamlit** web application.

---

## 📁 Folder Structure

```text
House Price Prediction/
├── data/
│   └── raw/
│       ├── train.csv                # Training dataset (features + target SalePrice)
│       ├── test.csv                 # Test dataset for out-of-sample predictions
│       ├── sample_submission.csv    # Submission format benchmark
│       └── data_description.txt     # Detailed feature definitions
├── models/
│   ├── .gitkeep
│   ├── best_house_price_model.joblib  # [Placeholder: serialized best model]
│   └── neighborhood_encoder.joblib    # [Placeholder: serialized categorical encoder]
├── outputs/
│   ├── .gitkeep
│   ├── eda_target_distribution.png    # [Placeholder: EDA plot]
│   ├── eda_correlation_heatmap.png    # [Placeholder: Correlation heatmap]
│   ├── actual_vs_predicted.png        # [Placeholder: Model evaluation plot]
│   └── residual_plot.png              # [Placeholder: Residual distribution plot]
├── app.py                           # Streamlit interactive prediction app
├── house_price_prediction.ipynb     # Main Jupyter notebook (training & evaluation)
├── README.md                        # Project documentation
└── REPORT.md                        # Formal analytical report
```

---

## 📊 Dataset Link & Location

- **Local Path:** `data/raw/` (`train.csv`, `test.csv`, `data_description.txt`)
- **Dataset Origin:** Ames Housing Dataset compiled by Dean De Cock (Truman State University).
- **Public Reference:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Dataset Properties:** 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa, across 1,460 training records.

---

## ⚙️ Installation Steps

1. Navigate to the project root or parent assignment directory:
   ```bash
   cd "ML Assingment Projects"
   ```

2. Activate your virtual environment:
   ```bash
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Running the Jupyter Notebook
Open and run all cells sequentially:
```bash
cd "House Price Prediction"
jupyter notebook house_price_prediction.ipynb
```

### 2. Running the Streamlit Prediction App
Launch the interactive web user interface:
```bash
cd "House Price Prediction"
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📈 Evaluation Results & Comparison (Placeholder)

> *Results will be populated during model training and evaluation.*

| Model | Validation MSE | Validation RMSLE | Notes / Observations |
| :--- | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | *[TBD]* | *[TBD]* | *[TBD]* |
| **Random Forest Regressor** | *[TBD]* | *[TBD]* | *[TBD]* |
| **XGBoost Regressor (Optional)** | *[TBD]* | *[TBD]* | *[TBD]* |

**Best Model Selected:** *[To be determined based on minimum validation RMSLE]*

---

## 👥 Reproducibility & Seed
All splits and stochastic algorithms use `random_state=42`.
