# Project 2: Image Classification - Technical Report

## 1. Introduction
This project focuses on multiclass image classification to identify three distinct object categories:
- **Cars**
- **Cricket ball**
- **Ice cream cone**

The dataset consists of 987 total images:
- **Cars:** 366 images
- **Cricket ball:** 345 images
- **Ice cream cone:** 276 images

The main goal is to compare **Raw Pixels** (flattened pixel intensities) against **HOG Features** (Histogram of Oriented Gradients) using two classical machine learning classifiers: **Logistic Regression** and **Support Vector Machine (SVM)**.

---

## 2. Methodology & Preprocessing

### Image Pipeline
1. **Grayscale Conversion:** Images are converted to grayscale using OpenCV (`cv2.IMREAD_GRAYSCALE`) to focus on shape, edges, and gradients rather than color variations.
2. **Resizing:** All images are standardized to a uniform size of $64 \times 64$ pixels.
3. **Normalization:** Pixel values are divided by 255.0 to scale intensity values into the range $[0, 1]$.
4. **Dataset Splitting:** The data is split into 80% training (789 samples) and 20% testing (198 samples) using stratified sampling with `random_state=42`.

### Feature Representations
- **Raw Pixels:** Flattened 1D vector of length $64 \times 64 = 4096$ values per image.
- **HOG Features:** Extracted using `skimage.feature.hog`, generating a structural edge descriptor vector of 2916 values per image.

---

## 3. Experimental Setup & Model Training
We trained four distinct model configurations using `random_state=42`:
1. **Experiment 1:** Logistic Regression on Raw Pixels (`max_iter=1000`)
2. **Experiment 2:** Logistic Regression on HOG Features (`max_iter=1000`)
3. **Experiment 3:** Support Vector Machine (SVM) on Raw Pixels
4. **Experiment 4:** Support Vector Machine (SVM) on HOG Features

---

## 4. Evaluation Results

### Accuracy Comparison Table

| Experiment | Feature Pipeline | Model | Training Accuracy | Testing Accuracy |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Raw Pixels | Logistic Regression | 100.00% | 83.33% |
| 2 | HOG Features | Logistic Regression | 100.00% | 93.43% |
| 3 | Raw Pixels | Support Vector Machine (SVM) | 96.96% | 93.94% |
| 4 | HOG Features | Support Vector Machine (SVM) | 100.00% | **94.95%** |

### Key Findings & Observations
- **HOG Features Outperform Raw Pixels:** HOG improved Logistic Regression accuracy from 83.33% to 93.43% (+10.10%) and SVM accuracy from 93.94% to 94.95%.
- **Robustness of SVM:** SVM achieved strong performance even on raw pixels (93.94%), showing that the maximum margin decision boundary handles high-dimensional vectors well.
- **Confusion Matrix Analysis:**
  - Car images have strong horizontal and rectangular edge contours, leading to high classification accuracy.
  - Cricket balls are circular and ice-cream cones have triangular or conical shapes, making HOG edge gradients highly effective at distinguishing them.
  - The confusion matrix plot is saved in `outputs/confusion_matrices.png`.
- **Qualitative Test Predictions:**
  - The sample test predictions plot saved in `outputs/test_predictions.png` confirms that the best model accurately predicts all classes on unseen test samples.

---

## 5. Best Model Selection
- **Selected Model:** Support Vector Machine on HOG features (`SVM HOG`).
- **Test Accuracy:** 94.95%
- **Explanation:** Support Vector Machine trained on HOG features achieved the highest test accuracy of 94.95%. HOG effectively captures structural edge gradients, allowing SVM to form optimal decision boundaries across classes.
- **Serialization:** Saved to `models/best_model.pkl` using `joblib`.

---

## 6. Web Application
A lightweight Streamlit web application (`app.py`, 15 lines) is provided. Users can upload an image of a car, cricket ball, or ice cream cone, and the app will preprocess the image, compute HOG descriptors, and display the predicted class.

---

## 7. Limitations & Future Improvements
- **Dataset Size:** With 987 total images, further expanding the dataset could improve generalization.
- **Color Information:** Some classes (e.g. red cricket balls) could benefit from color histograms combined with HOG descriptors.
- **Deep Learning:** Modern Convolutional Neural Networks (CNNs) could be evaluated to automatically learn spatial hierarchy features without manual feature engineering.
