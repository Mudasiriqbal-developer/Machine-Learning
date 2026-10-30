# Project 2: Image Classification - Technical Report

**Course Assignment** | **Due Date:** 3 November 2026  
**Status:** In Progress (Structure & Baseline Scaffolded)  

---

## 1. Introduction
- **Problem Statement:** Multiclass visual recognition to classify images into one of three distinct categories: **Cars**, **Cricket Balls**, and **Ice-Cream Cones**.
- **Motivation & Significance:** Raw pixel values are notoriously susceptible to variations in illumination, translation, viewpoint, and background clutter. Comparing raw flattened pixel inputs against a hand-crafted spatial edge descriptor—Histogram of Oriented Gradients (HOG)—illustrates how intermediate feature representations affect classical machine learning classifiers (Logistic Regression and Support Vector Machines).
- **Core Research Questions:**
  1. Does HOG provide significant accuracy gains over raw pixel values across linear and kernelized classifiers?
  2. How do Logistic Regression and SVM handle high-dimensional feature spaces?
  3. Which visual category exhibits the highest rate of confusion and why?

*[Placeholder: Add formal background, dataset statistics, and research scope during implementation.]*

---

## 2. Preprocessing / Methodology
- **Image Pipeline:**
  - Standardized resizing: Rescaling images to uniform dimensions (e.g., \(64 \times 64\) or \(100 \times 100\)).
  - Grayscale conversion: Transforming 3-channel RGB to single-channel intensity maps for gradient orientation calculation.
  - Normalization: Scaling pixel intensities to \([0, 1]\).
- **Feature Representations:**
  - **Raw Pixels:** 1D vector formed by flattening normalized 2D grayscale pixel arrays (e.g., \(100 \times 100 = 10,000\) dimensions).
  - **HOG Features:** Computing local image gradients, binning gradient orientations into histograms per cell, and performing block normalization to achieve local photometric invariance (using `skimage.feature.hog`).
- **Data Partitioning:**
  - Stratified 80/20 train/test split preserving class distribution using `random_state=42`.

*[Placeholder: Document final image dimensions, HOG hyperparameter settings (orientations, pixels_per_cell, cells_per_block), and feature vector lengths during implementation.]*

---

## 3. Model or Algorithm Comparison
- **Multinomial Logistic Regression:**
  - Formulation: Models class probabilities using the softmax function:
    \[
    P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
    \]
  - Optimization: Cross-entropy loss with \(L_2\) regularization (ridge).
  - Strengths: Fast convergence, probabilistic outputs, highly interpretable feature weights.
- **Support Vector Machine (SVM):**
  - Formulation: Maximizes the geometric margin between decision boundaries and closest support vectors:
    \[
    \min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i
    \]
  - Kernels evaluated: Linear / RBF (Radial Basis Function).
  - Strengths: Effective in high-dimensional feature spaces; robust to moderate feature collinearity.

*[Placeholder: Document convergence iterations, regularization parameter C, and kernel choice during implementation.]*

---

## 4. Evaluation Results
- **Quantitative Comparison Across 4 Configurations:**

| Experiment | Feature Pipeline | Model | Train Accuracy | Test Accuracy | Macro F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Raw Pixels | Logistic Regression | *[Pending]* | *[Pending]* | *[Pending]* |
| 2 | HOG Features | Logistic Regression | *[Pending]* | *[Pending]* | *[Pending]* |
| 3 | Raw Pixels | Support Vector Machine | *[Pending]* | *[Pending]* | *[Pending]* |
| 4 | HOG Features | Support Vector Machine | *[Pending]* | *[Pending]* | *[Pending]* |

- **Confusion Matrix Analysis:**
  - Analysis of inter-class confusion (e.g., cricket balls confused with circular elements of cars or round ice-cream scoops).
  - Visual confusion matrix plots saved to `outputs/confusion_matrices.png`.
- **Qualitative Predictions:**
  - Discussion of sample correct predictions and notable failure cases (`outputs/sample_predictions.png`).

*[Placeholder: Populate exact metric values, precision/recall per class, and detailed diagnostic discussions during implementation.]*

---

## 5. Limitations
- **Dataset Size & Class Variation:** Relatively small training sample size per category increases risk of overfitting on raw pixels.
- **Background Clutter & Perspective Shifts:** Classical HOG assumes consistent structural edges; heavily cluttered or non-canonical views degrade descriptor effectiveness.
- **Manual Feature Engineering:** Handcrafted HOG features are not end-to-end learnable compared to modern Convolutional Neural Networks (CNNs).

*[Placeholder: Add detailed breakdown of misclassified test images and contextual failure factors during implementation.]*

---

## 6. Improvements
- **Data Augmentation:** Introducing random rotations, slight translations, and horizontal flips to increase variance.
- **Color-Aware Descriptors:** Evaluating Color HOG or combined Color Histogram + HOG to distinguish objects where color is a dominant cue (e.g., red cricket ball).
- **Deep Feature Extraction:** Utilizing pre-trained CNN backbones (e.g., ResNet18 / MobileNetV2) as feature extractors for SVM.

*[Placeholder: Document experimental validation of potential improvements during implementation.]*
