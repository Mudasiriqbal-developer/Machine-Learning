# Project 3: Computer Vision with OpenCV - Technical Report

**Course Assignment** | **Due Date:** 3 November 2026  
**Status:** In Progress (Structure & Baseline Scaffolded)  

---

## 1. Introduction
- **Scope & Objectives:** Classical computer vision techniques remain foundational for understanding visual geometry, low-level image processing, and real-time embedded applications. This project implements and benchmarks four classical vision algorithms in OpenCV:
  1. Object and facial feature localization with **Haar Cascade Classifiers**.
  2. Local invariant feature detection and correspondence matching comparing **ORB** and **SIFT**.
  3. Structural gradient analysis through **Canny Edge Detection**.
  4. Geometric line modeling comparing **Standard** versus **Probabilistic Hough Transforms**.
- **Key Questions Explored:**
  - How do handcrafted Haar wavelets perform under variable pose and lighting?
  - Does ORB's binary descriptor speed compromise matching accuracy relative to SIFT's scale-space extrema?
  - How sensitive is Canny edge detection to non-optimal threshold ratios?
  - Why is the Probabilistic Hough Transform preferred over the Standard Hough Transform in practical engineering pipelines?

*[Placeholder: Add formal background, historical context, and mathematical citations during implementation.]*

---

## 2. Preprocessing / Methodology
- **Color Spaces & Conversions:**
  - BGR to Grayscale transformation (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`) for gradient-based detectors and Haar cascades.
  - Color channel preservation for annotated visual overlays (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` for Matplotlib).
- **Filtering & Noise Suppression:**
  - Application of Gaussian blurs (\(5 \times 5\) kernel) prior to gradient estimation in Canny and Hough workflows to suppress high-frequency noise.
- **Region of Interest (ROI) Localization:**
  - Hierarchical bounding box extraction: detecting primary face regions before initiating search windows for eyes to minimize false positives.
- **Feature Descriptor Matching Strategy:**
  - SIFT: Euclidian (\(L_2\)) distance matching combined with Lowe's ratio test (\(\text{threshold} = 0.75\)) to filter ambiguous nearest neighbors.
  - ORB: Hamming distance matching leveraging bitwise XOR for real-time binary descriptor comparison.

*[Placeholder: Document exact kernel sizes, smoothing parameters, and scale factor selections during implementation.]*

---

## 3. Model or Algorithm Comparison
- **Haar Cascade Classifiers:**
  - Principle: Viola-Jones framework computing rectangular Haar features rapidly using the Integral Image representation, filtered through an AdaBoost attentional cascade.
  - Characteristics: Rapid rejection of non-face background windows; sensitive to in-plane/out-of-plane head rotations.
- **SIFT vs. ORB Feature Matching:**
  - **SIFT:** Operates on Difference-of-Gaussians (DoG) scale-space; generates 128-dimensional floating point descriptors invariant to scale, rotation, and illumination changes. Computationally intensive.
  - **ORB:** Combines FAST corner detection (with Harris corner ranking) and rotated BRIEF binary descriptors. Computationally efficient, hardware friendly, open source (patent-free).
- **Canny Edge Detection:**
  - Five-stage optimal edge detection: Gaussian smoothing \(\to\) Sobel gradient magnitude/orientation \(\to\) Non-maximum suppression \(\to\) Double thresholding \(\to\) Hysteresis edge tracking.
- **Standard Hough vs. Probabilistic Hough Transform:**
  - **Standard (`HoughLines`):** Maps every edge pixel to a sinusoid in \((\rho, \theta)\) parameter space. Yields infinite lines defined by normal parameters; high accumulator memory footprint.
  - **Probabilistic (`HoughLinesP`):** Samples a random subset of edge points, connecting collinear points into finite line segments with `minLineLength` and `maxLineGap` constraints. Significantly faster and directly yields segment endpoints.

*[Placeholder: Document mathematical derivations, runtime benchmarks, and parameter tuning details during implementation.]*

---

## 4. Evaluation Results
- **Quantitative & Qualitative Assessment Table:**

| Task | Technique / Variant | Primary Parameter(s) | Detection / Match Quality | Processing Time |
| :--- | :--- | :--- | :---: | :---: |
| **Face Detection** | Haar Frontal Face | `scaleFactor=1.1, minNeighbors=5` | *[Pending]* | *[Pending]* |
| **Eye Detection** | Haar Eye (Within ROI) | `scaleFactor=1.1, minNeighbors=3` | *[Pending]* | *[Pending]* |
| **Feature Matching** | SIFT + BFMatcher (L2) | `ratio_test=0.75` | *[Pending]* | *[Pending]* |
| **Feature Matching** | ORB + BFMatcher (Hamming)| `cross_check=True` | *[Pending]* | *[Pending]* |
| **Edge Detection** | Canny | `thresholds=(50, 150)` vs `(100, 200)` | *[Pending]* | *[Pending]* |
| **Line Detection** | Standard Hough | `threshold=150` | *[Pending]* | *[Pending]* |
| **Line Detection** | Probabilistic Hough | `threshold=50, minLine=50, maxGap=10`| *[Pending]* | *[Pending]* |

- **Visual Results:**
  - Annotated face and eye detection boxes (`outputs/face_eye_detection_single.png`, `outputs/face_eye_detection_multi.png`).
  - SIFT vs ORB keypoint matching correspondences (`outputs/sift_matches.png`, `outputs/orb_matches.png`).
  - Canny edge threshold variations (`outputs/canny_edge_comparisons.png`).
  - Road lane and geometric line segment overlays (`outputs/hough_line_comparison.png`).

*[Placeholder: Populate exact metric values, execution times, match counts, and visual findings during implementation.]*

---

## 5. Limitations
- **Haar Cascades:** Inability to handle occluded faces, non-frontal poses, extreme facial expressions, or dramatic lighting variations. High false positive rate in complex backgrounds.
- **SIFT & ORB Invariance Bounds:** SIFT struggles with extreme 3D perspective changes; ORB's binary descriptors exhibit lower matching discriminability compared to dense 128-D descriptors when textures repeat.
- **Canny Edge Sensitivity:** Manual threshold selection is brittle across varying scene illuminations; requires adaptive or Otsu-based thresholding for dynamic environments.
- **Hough Line Over-Segmentation:** Standard Hough identifies infinite lines regardless of actual scene geometry; Probabilistic Hough requires careful tuning of gap and length parameters to avoid broken line segments.

*[Placeholder: Add visual examples of failure cases and boundary condition analyses during implementation.]*

---

## 6. Improvements
- **Deep Learning-Based Face Detection:** Replacing Haar cascades with modern lightweight detectors such as OpenCV's DNN Face Detector (Single Shot MultiBox Detector with ResNet-10) or MediaPipe.
- **Learned Local Features:** Employing deep learned feature descriptors (e.g., SuperPoint, SuperGlue) for challenging low-texture or extreme viewpoint correspondences.
- **Adaptive Edge & Lane Detection:** Integrating adaptive Gaussian thresholding, bilateral filtering, or semantic segmentation models (e.g., U-Net / LaneNet) for robust real-world autonomous navigation.

*[Placeholder: Document experimental validation of potential improvements during implementation.]*
