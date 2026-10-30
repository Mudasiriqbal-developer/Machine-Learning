# Project 2: Image Classification (Cars, Cricket Balls, Ice Cream Cones)

## Overview
This project classifies images into three classes:
- **Cars**
- **Cricket ball**
- **Ice cream cone**

We compare two feature representations:
1. **Raw Pixels** (flattened 64x64 grayscale images)
2. **HOG Features** (Histogram of Oriented Gradients using `skimage.feature.hog`)

Each feature type is trained using two models:
- **Logistic Regression**
- **Support Vector Machine (SVM)**

All experiments use `random_state=42` and an 80/20 stratified train-test split.

---

## Dataset
- **Location:** `data/images/`
- **Class Distribution:**
  - Cars: 366 images
  - Cricket ball: 345 images
  - Ice cream cone: 276 images
  - Total: 987 images
- **Preprocessing:**
  - Images converted to grayscale using OpenCV (`cv2.IMREAD_GRAYSCALE`)
  - Resized to 64x64 pixels
  - Scaled pixel values to range [0, 1]

---

## Model Evaluation Results

| Experiment | Feature Pipeline | Model | Train Accuracy | Test Accuracy |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Raw Pixels | Logistic Regression | 100.00% | 83.33% |
| 2 | HOG Descriptors | Logistic Regression | 100.00% | 93.43% |
| 3 | Raw Pixels | Support Vector Machine (SVM) | 96.96% | 93.94% |
| 4 | HOG Descriptors | Support Vector Machine (SVM) | 100.00% | **94.95%** |

### Best Model Selection
- **Best Model:** Support Vector Machine on HOG features (`SVM HOG`).
- **Test Accuracy:** 94.95%
- **Saved Model File:** `models/best_model.pkl`

---

## Project Structure
```text
Image Classification/
├── data/
│   └── images/
│       ├── Cars/
│       ├── Cricket ball/
│       └── Ice cream cone/
├── models/
│   └── best_model.pkl
├── outputs/
│   ├── sample_images.png
│   ├── confusion_matrices.png
│   └── test_predictions.png
├── app.py
├── image_classification.ipynb
├── README.md
└── REPORT.md
```

---

## How to Run

### 1. Run the Notebook
Open and run `image_classification.ipynb` to view all data loading, training, evaluations, and plots.

### 2. Run the Streamlit Web App
Launch the interactive web application to upload and classify new images:
```bash
streamlit run app.py
```
