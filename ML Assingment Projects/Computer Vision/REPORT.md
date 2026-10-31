# Project 3: Computer Vision with OpenCV - Technical Report

**Course Assignment** | **Due Date:** 3 November 2026  
**Status:** Completed  

---

## 1. Introduction
Classical computer vision provides the geometric and mathematical foundation for modern visual perception. Unlike deep learning models that require extensive training data and GPU acceleration, classical algorithms rely on analytical principles—such as intensity gradients, multi-scale feature pyramids, spatial accumulators, and scale-space extrema.

This project implements, analyzes, and benchmarks four classical computer vision operations using OpenCV (`cv2`):
1. **Face and Eye Detection:** Object localization using pre-trained **Haar Feature-based Cascade Classifiers** with Region of Interest (ROI) processing.
2. **Feature Detection & Matching:** Comparative analysis of **ORB** (Oriented FAST and Rotated BRIEF) versus **SIFT** (Scale-Invariant Feature Transform) on a rotated and scaled landmark image pair.
3. **Edge Detection:** Multi-stage **Canny Edge Detection** examining parameter sensitivity across dual hysteresis threshold ratios.
4. **Line Detection:** Geometric line extraction comparing the **Standard Hough Line Transform** (`HoughLines`) with the **Probabilistic Hough Line Transform** (`HoughLinesP`).

---

## 2. Preprocessing / Methodology

### 2.1 Color Space Conversions
- **BGR to Grayscale:** Converted 3-channel BGR input images to 1-channel grayscale using `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`. Grayscale representation reduces computational overhead by 66% while retaining all structural luminance information required for Haar feature evaluation, gradient estimation, and corner detection.
- **BGR to RGB:** Matplotlib requires images in RGB order. Annotated images were transformed using `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` prior to visualization and saving.

### 2.2 Hierarchical Region of Interest (ROI) Processing
- For facial feature detection, face coordinates `(x, y, w, h)` were first extracted using the frontal face cascade.
- The eye cascade was applied strictly within the sub-image region `gray[y:y+h, x:x+w]`. Restricting the search space to the face ROI prevents false eye detections across the wider background.

### 2.3 Feature Matching Pipeline
- **SIFT Pipeline:** Detected keypoints and computed 128-dimensional floating-point descriptors. Matched descriptors using Brute-Force Matcher (`cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)`) to ensure reciprocal nearest neighbors.
- **ORB Pipeline:** Detected FAST corner points and computed 32-byte binary descriptors. Matched descriptors using Brute-Force Matcher with Hamming distance (`cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`).
- For both methods, matches were sorted by descriptor distance and the top 50 strongest matches were visualized using `cv2.drawMatches()`.

---

## 3. Model and Algorithm Comparison

### 3.1 Haar Cascade Classifiers (Viola-Jones)
- **Mathematical Basis:** Evaluates rectangular Haar-like features (edge, line, and four-rectangle features) rapidly using the **Integral Image** representation ($O(1)$ computation per rectangular region).
- **Cascade Structure:** Employs an attentional cascade of AdaBoost-selected classifiers. Negative sub-windows are rejected in the earliest stages, reserving computational resources for candidate face regions.
- **Parameters:**
  - `scaleFactor`: Image pyramid scale reduction step (e.g., 1.1 = 10% reduction per scale).
  - `minNeighbors`: Number of overlapping candidate rectangles required to retain a detection.
  - `minSize`: Minimum bounding box dimension $(w, h)$ to search.

### 3.2 SIFT vs. ORB Feature Detection & Matching
- **SIFT (Scale-Invariant Feature Transform):**
  - Detects extrema across Difference-of-Gaussians (DoG) scale-space.
  - Assigns dominant orientation based on local image gradient directions.
  - Generates 128-dimensional floating-point feature vectors normalized to provide illumination, scale, and rotation invariance.
  - Distance Metric: Euclidean ($L_2$) distance.
- **ORB (Oriented FAST and Rotated BRIEF):**
  - Detects keypoints using FAST corner detection and ranks them with the Harris corner metric.
  - Computes orientation using intensity centroid vectors.
  - Generates 256-bit (32-byte) binary descriptors using rotated BRIEF patterns.
  - Distance Metric: Hamming distance computed via fast bitwise XOR hardware instructions.

### 3.3 Canny Edge Detection
A 5-stage optimal edge detection algorithm:
1. **Gaussian Smoothing:** Suppresses high-frequency noise.
2. **Gradient Calculation:** Applies Sobel kernels to compute magnitude $G = \sqrt{G_x^2 + G_y^2}$ and direction $\theta = \arctan(G_y / G_x)$.
3. **Non-Maximum Suppression (NMS):** Thin edges by preserving only local gradient maxima along the gradient direction.
4. **Double Thresholding:** Classifies pixels into strong edges ($G \ge T_{\text{high}}$), weak edges ($T_{\text{low}} \le G < T_{\text{high}}$), and non-edges ($G < T_{\text{low}}$).
5. **Hysteresis Tracking:** Retains weak edges only if connected to a strong edge.

### 3.4 Standard Hough vs. Probabilistic Hough Transform
- **Standard Hough (`HoughLines`):**
  - Maps edge points $(x_i, y_i)$ into sinusoidal curves in polar accumulator space $\rho = x \cos\theta + y \sin\theta$.
  - Cells exceeding the accumulator threshold represent infinite lines.
  - Limitations: Requires extensive memory, votes for all collinear points, and returns infinite lines rather than line segments.
- **Probabilistic Hough (`HoughLinesP`):**
  - Applies a progressive probabilistic approach, voting with a random subset of edge pixels.
  - Follows connected edge paths to output explicit finite segment endpoints $(x_1, y_1, x_2, y_2)$ governed by `minLineLength` and `maxLineGap`.

---

## 4. Evaluation Results

### 4.1 Quantitative Results Table

| Task | Method | Parameters | Key Metric | Elapsed Time | Visual Output File |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Face Detection** | Haar Frontal Face | `scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)` | 1 Face / 3 Eyes | ~0.02s | `outputs/face_eye_detection_single.png` |
| **Crowd Face Detection** | Haar Frontal Face | `scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)` | 23 Faces / 0 Eyes | ~0.04s | `outputs/face_eye_detection_multi.png` |
| **Haar Sensitivity (Loose)**| Haar Frontal Face | `scaleFactor=1.05, minNeighbors=2, minSize=(15, 15)` | 27 Faces (False Positives) | ~0.06s | `outputs/face_detection_parameter_comparison.png` |
| **Haar Sensitivity (Strict)**| Haar Frontal Face | `scaleFactor=1.3, minNeighbors=8, minSize=(40, 40)` | 5 Faces (Missed Detections) | ~0.02s | `outputs/face_detection_parameter_comparison.png` |
| **Feature Matching** | SIFT + BFMatcher | `NORM_L2, crossCheck=True, top=50` | 424 kp1, 335 kp2 | ~0.041s | `outputs/sift_matches.png` |
| **Feature Matching** | ORB + BFMatcher | `NORM_HAMMING, crossCheck=True, top=50` | 415 kp1, 441 kp2 | ~0.015s | `outputs/orb_matches.png` |
| **Edge Detection (Low)** | Canny | `thresholds=(50, 150)` | Dense edge map | ~0.005s | `outputs/canny_edge_comparisons.png` |
| **Edge Detection (High)**| Canny | `thresholds=(100, 200)` | Filtered grid edges | ~0.005s | `outputs/canny_edge_comparisons.png` |
| **Line Detection** | Standard Hough | `threshold=150` vs `250` | 163 lines vs 31 lines | ~0.010s | Visualized in notebook |
| **Line Detection** | Probabilistic Hough | `minLineLength=50` vs `150` | 249 segs vs 91 segs | ~0.008s | `outputs/hough_line_comparison.png` |

---

## 5. Limitations & Failure Cases

### 5.1 Haar Cascade Failure Modes
1. **False Positive Clutter (Loose Settings):**
   - Setting `minNeighbors=2` caused the classifier to misidentify textured clothing folds and background shadows as faces.
2. **Missed Detections (Strict Settings):**
   - Setting `minNeighbors=8` and `scaleFactor=1.3` caused the detector to miss 18+ small and turned faces in the crowd photograph.
3. **Resolution Mismatch for Eyes:**
   - In `multiplefaces.jpg`, individual face bounding boxes averaged $25 \times 25$ pixels. The eye cascade window required features larger than the physical eye pixels present, causing complete detection failure on the crowd image.

### 5.2 SIFT vs. ORB Trade-offs
1. **Speed vs. Discriminability:**
   - ORB was approximately **2.8× faster** than SIFT (~0.015s vs. ~0.041s) due to binary bitwise matching.
   - However, SIFT's 128-dimensional floating-point vectors provide higher descriptor uniqueness in scenes with repetitive textures.
2. **Scale Invariance Limits:**
   - ORB uses a simple image pyramid and brief pairwise pixel tests, which exhibits lower scale invariance compared to SIFT's continuous scale-space Difference-of-Gaussians formulation.

### 5.3 Hough Transform Limitations
1. **Standard Hough Line Extent:**
   - Standard `HoughLines` returns infinite mathematical lines across the entire image coordinate plane, cluttering the background beyond the actual Sudoku grid.
2. **Over-Segmentation in Probabilistic Hough:**
   - When `minLineLength` is set too low (50 px), individual strokes of printed numbers are detected as lines, disrupting structured grid parsing.

---

## 6. Recommendations & Future Improvements

1. **Lightweight Deep Learning Face Detectors:**
   - Replace Haar cascades with OpenCV's deep neural network face detector (`cv2.dnn.readNetFromCaffe`) or MediaPipe Face Mesh, which provide robust detection across $360^\circ$ yaw/pitch angles and varying lighting.
2. **Learned Local Features:**
   - Integrate deep learned feature matchers such as **SuperPoint** and **SuperGlue** for environments with extreme viewpoint angles and low illumination.
3. **Adaptive Thresholding for Edge and Line Pipelines:**
   - Implement Otsu's thresholding or median-based dynamic threshold calculation:
     $$\text{lower} = \max(0, 0.67 \times \text{median}), \quad \text{upper} = \min(255, 1.33 \times \text{median})$$
     to automatically calibrate Canny thresholds across varying lighting conditions.
