# Project 3: Computer Vision with OpenCV

## 📌 Project Overview
This project implements and evaluates classical, geometry-based, and feature-based computer vision algorithms using **OpenCV (`cv2`)**. The study covers four core vision tasks:
1. **Face and Eye Detection:** Multi-scale object detection using pre-trained **Haar Feature-based Cascade Classifiers** with Region of Interest (ROI) processing.
2. **Feature Detection & Matching:** A comparative benchmark of **ORB** (Oriented FAST and Rotated BRIEF) versus **SIFT** (Scale-Invariant Feature Transform) evaluating keypoints, matching quality, and execution speed.
3. **Edge Detection:** Multi-stage **Canny Edge Detection** analyzing parameter sensitivity across low and high hysteresis threshold values.
4. **Line Detection:** Geometric line detection contrasting the **Standard Hough Line Transform** (`HoughLines`) with the **Probabilistic Hough Line Transform** (`HoughLinesP`).

---

## 📁 Folder Structure

```text
Computer Vision/
├── data/
│   ├── images/                     # Test images for detection and matching
│   │   ├── contours.jpg
│   │   ├── DownlaodImage.jpg
│   │   ├── Effiel1.png             # Pair 1 for feature matching
│   │   ├── Effiel2.png             # Pair 2 for feature matching
│   │   ├── filtered_road.jpg       # Lane detection test image
│   │   ├── multiplefaces.jpg       # Multi-subject face detection
│   │   ├── scan.jpg                # Document edge detection
│   │   ├── soduku.jpg              # Grid / line detection
│   │   └── Trump.jpg               # Single portrait face detection
│   └── haarcascades/               # Pre-trained Haar model XML files
│       ├── haarcascade_frontalface_default.xml
│       └── haarcascade_eye.xml
├── outputs/
│   ├── .gitkeep
│   ├── face_eye_detection_single.png          # Single face & eye detection result
│   ├── face_eye_detection_multi.png           # Multi-face detection result
│   ├── face_detection_parameter_comparison.png# Haar parameter variation comparison
│   ├── sift_matches.png                       # SIFT top 50 feature matches
│   ├── orb_matches.png                        # ORB top 50 feature matches
│   ├── canny_edge_comparisons.png             # Canny edge threshold comparison
│   └── hough_line_comparison.png              # Hough line detection result
├── computer_vision.ipynb           # Main Jupyter notebook (all executed cells)
├── README.md                       # Project documentation
└── REPORT.md                       # Formal technical report
```

---

## 📊 Dataset & Asset Details

- **Local Images Path:** `data/images/`
- **Haar Cascades Path:** `data/haarcascades/`
- **Key Assets Used:**
  - `Trump.jpg`: High-resolution single frontal portrait used for single face and localized eye detection.
  - `multiplefaces.jpg`: Group photograph (280x640) with dense, small faces used for multi-scale crowd detection and parameter tuning.
  - `Effiel1.png`, `Effiel2.png`: Eiffel Tower image pair undergoing scale and viewpoint change used for SIFT and ORB descriptor matching.
  - `soduku.jpg`: Structured Sudoku grid (576x768) used for Canny edge filtering and Hough line transforms.

---

## ⚙️ Installation & Setup

1. Open PowerShell and navigate to the project directory:
   ```bash
   cd "D:\Machine Learning\ML Assingment Projects"
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Verify dependencies:
   ```bash
   pip install opencv-python matplotlib numpy pandas
   ```

---

## 🚀 How to Run

Launch Jupyter and open the notebook:
```bash
cd "Computer Vision"
jupyter notebook computer_vision.ipynb
```
All cells are fully pre-executed with visible outputs, and annotated figures are saved to `outputs/`.

---

## 📈 Evaluation Results & Comparison

### 1. Face & Eye Detection (Haar Cascades)

| Test Image | Parameters | Faces Found | Eyes Found | Observations & Failure Cases |
| :--- | :--- | :---: | :---: | :--- |
| **Single Face (`Trump.jpg`)** | `scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)` | 1 | 3 | Face cleanly detected; 2 true eyes plus 1 slight duplicate bounding box. |
| **Multi-Face (`multiplefaces.jpg`)** | `scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)` | 23 | 0 | 23 of ~28 faces detected; eye detection failed because eyes are smaller than cascade window. |

#### Parameter Sensitivity on Multi-Face Image:
- **Loose (`scaleFactor=1.05, minNeighbors=2, minSize=15`):** 27 faces detected. Successfully detects smaller peripheral faces, but creates false positive boxes on textured clothing and background artifacts.
- **Standard (`scaleFactor=1.1, minNeighbors=5, minSize=30`):** 23 faces detected. Balanced detection with zero false alarms on background.
- **Strict (`scaleFactor=1.3, minNeighbors=8, minSize=40`):** 5 faces detected. Misses 18+ valid faces due to aggressive scaling and neighbor thresholds.

---

### 2. Feature Matching: ORB vs. SIFT

Matches computed on `Effiel1.png` and `Effiel2.png` keeping top 50 matches using `cv2.BFMatcher`:

| Algorithm | Keypoints (Img 1) | Keypoints (Img 2) | Matches Kept | Matcher Distance | Time (s) | Relative Speed | Invariance & Quality |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **SIFT** | 424 | 335 | 50 | `NORM_L2` | ~0.041s | 1.0× (Baseline) | High discriminability; robust to scale, rotation, and lighting changes. |
| **ORB** | 415 | 441 | 50 | `NORM_HAMMING` | ~0.015s | ~2.8× Faster | Fast binary descriptors; rotation invariant; ideal for real-time applications. |

---

### 3. Edge Detection: Canny Algorithm

Evaluated on `soduku.jpg` across two hysteresis threshold pairs:
- **Lower Thresholds (50, 150):** Detects fine structural details, including inner printed numbers, grid cell borders, and faint paper texture.
- **Higher Thresholds (100, 200):** Filters out weaker gradient noise and digit textures, preserving only strong, continuous grid borders.

---

### 4. Line Detection: Standard vs. Probabilistic Hough Transform

Evaluated on the Canny edge map of `soduku.jpg`:

| Algorithm | Parameters Tested | Lines Detected | Endpoint Output | Behavior & Recommended Use Case |
| :--- | :--- | :---: | :--- | :--- |
| **Standard Hough (`HoughLines`)** | `threshold=150`<br>`threshold=250` | 163 lines<br>31 lines | Infinite lines $(\rho, \theta)$ | Threshold 150 produces heavy line clutter; threshold 250 isolates primary grid lines. Lines extend infinitely across image. |
| **Probabilistic Hough (`HoughLinesP`)** | `minLineLength=50`<br>`minLineLength=150` | 249 segments<br>91 segments | Finite segments $(x_1, y_1, x_2, y_2)$ | `minLineLength=50` captures short number strokes; `minLineLength=150` isolates continuous grid border segments. |

---

## 🔍 Summary of Successful & Unsuccessful Detection Cases

### Successful Detection Cases:
1. **Frontal Single Face:** Haar cascade reliably located the single frontal face in `Trump.jpg` with standard settings.
2. **Eiffel Tower Matching:** Both SIFT and ORB matched 50 high-confidence correspondences across significant perspective and scale shifts.
3. **Sudoku Grid Isolation:** Probabilistic Hough Transform with `minLineLength=150` accurately extracted the clean borders of the Sudoku puzzle grid without detecting inner digit noise.

### Unsuccessful / Failure Cases:
1. **Low-Resolution Eyes in Multi-Face Image:** Eye cascade failed to detect eyes in `multiplefaces.jpg` because the individual faces were smaller than the minimum eye classifier window.
2. **False Positives with Loose Haar Parameters:** When `minNeighbors` was reduced to 2, textured background patterns and clothing folds were mistakenly classified as faces.
3. **Missed Detections with Strict Haar Parameters:** Increasing `scaleFactor` to 1.3 and `minNeighbors` to 8 caused the detector to miss over 75% of valid faces that were slightly turned or partially shaded.
4. **Infinite Line Clutter in Standard Hough:** Standard Hough transform lines extend infinitely across the entire image canvas, failing to represent finite physical object boundaries.
