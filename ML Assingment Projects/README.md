# Machine Learning Assignment Projects

**Submission Deadline:** 3 November 2026  
**Environment:** Python 3.10+ | Jupyter Notebook | Streamlit  

---

## 📌 Project Overview

This repository contains three self-contained machine learning and computer vision projects developed for the Machine Learning course assignment. Each project is located in its dedicated directory and can be executed, packaged, or submitted independently.

```text
ML Assingment Projects/
├── House Price Prediction/     # Project 1: Tabular Regression & Web App
├── Image Classification/       # Project 2: Feature Engineering, Classical ML & Web App
├── Computer Vision/            # Project 3: OpenCV Feature Detection & Geometric Transforms
├── requirements.txt            # Shared dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Top-level assignment overview & submission checklist
```

---

### 🏠 Project 1: House Price Prediction (Regression)
- **Dataset:** Ames Housing dataset (`train.csv`, `test.csv`).
- **Goal:** Predict residential house sale prices using regression modeling.
- **Algorithms:**
  - Multiple Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor (optional / exploratory)
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Logarithmic Error (RMSLE).
- **Deliverables:**
  - Jupyter Notebook (`house_price_prediction.ipynb`) covering EDA, preprocessing, model training, evaluation, and serialization.
  - Serialized best model (`models/`) using `joblib`.
  - Interactive prediction web application (`app.py`) built with Streamlit.
  - Project documentation (`README.md`) and technical report (`REPORT.md`).

---

### 🖼️ Project 2: Image Classification (Cars, Cricket Balls, Ice-Cream Cones)
- **Dataset:** 3-class image dataset consisting of Cars, Cricket Balls, and Ice-Cream Cones.
- **Goal:** Multiclass image classification comparing raw visual features against engineered descriptors.
- **Feature Representations:**
  - Raw flattened pixel representations (baseline)
  - Histogram of Oriented Gradients (HOG) feature descriptors
- **Algorithms:**
  - Multinomial Logistic Regression
  - Support Vector Machine (SVM)
- **Evaluation:** Training vs testing accuracy comparison, confusion matrices, precision/recall/F1-score, and qualitative sample prediction analysis.
- **Deliverables:**
  - Jupyter Notebook (`image_classification.ipynb`) executing all 4 experimental setups.
  - Serialized top model and feature extractor (`models/`).
  - Interactive image classification web application (`app.py`) built with Streamlit.
  - Project documentation (`README.md`) and technical report (`REPORT.md`).

---

### 👁️ Project 3: Computer Vision with OpenCV
- **Toolkit:** OpenCV (`cv2`), NumPy, Matplotlib.
- **Goal:** Implement and evaluate classical computer vision pipelines across detection, matching, and feature extraction.
- **Core Tasks:**
  1. **Face & Eye Detection:** Haar Cascade classifiers (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`) applied to single and multi-face images.
  2. **Feature Detection & Matching:** Comparative analysis of ORB (Oriented FAST and Rotated BRIEF) vs SIFT (Scale-Invariant Feature Transform) using BFMatcher / FLANN.
  3. **Edge Detection:** Canny Edge Detection with gradient calculation and hysteresis thresholding analysis.
  4. **Line Detection:** Standard Hough Transform (`HoughLines`) vs Probabilistic Hough Transform (`HoughLinesP`).
- **Deliverables:**
  - Jupyter Notebook (`computer_vision.ipynb`) demonstrating step-by-step implementations and parameter comparisons.
  - Output visualizations and annotated images saved in `outputs/`.
  - Project documentation (`README.md`) and technical report (`REPORT.md`).

---

## ⚙️ Installation & Environment Setup

1. **Clone or open the workspace:**
   ```bash
   cd "ML Assingment Projects"
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📋 General Rules & Development Guidelines

1. **Reproducibility:** A fixed random seed (`random_state=42`) is strictly enforced across all data splits, model initializations, and stochastic processes.
2. **Explainability:** All code sections are documented and commented clearly to support oral and technical project defenses.
3. **Visualization Standards:** Every plot includes clear titles, labeled axes, legends where appropriate, and is saved to the respective project's `outputs/` directory.
4. **Self-Contained Modularity:** Each of the three folders maintains its own data, scripts, models, and documentation so it can be zipped and submitted independently.

---

## ✅ Final Submission Checklist

- [ ] **Shared Setup:**
  - [x] Shared `requirements.txt` configured and verified.
  - [x] Clean `.gitignore` ignoring virtual environments, caches, and large archives.
  - [x] Top-level `README.md` created.
- [ ] **Project 1 (House Price Prediction):**
  - [ ] Complete notebook `house_price_prediction.ipynb` executed end-to-end.
  - [ ] Best model saved to `models/` with `joblib`.
  - [ ] Streamlit prediction app `app.py` tested and functional.
  - [ ] All diagnostic plots saved in `outputs/`.
  - [ ] `README.md` and `REPORT.md` fully populated with final numbers and findings.
- [ ] **Project 2 (Image Classification):**
  - [ ] Complete notebook `image_classification.ipynb` executed end-to-end.
  - [ ] Best classifier serialized in `models/`.
  - [ ] Streamlit image prediction app `app.py` tested and functional.
  - [ ] Confusion matrices and prediction grids saved in `outputs/`.
  - [ ] `README.md` and `REPORT.md` fully populated with comparative results.
- [ ] **Project 3 (Computer Vision):**
  - [ ] Complete notebook `computer_vision.ipynb` executed end-to-end.
  - [ ] Haar cascade detections, ORB vs SIFT matches, Canny edges, and Hough lines verified.
  - [ ] Result visualizations saved in `outputs/`.
  - [ ] `README.md` and `REPORT.md` completed.
