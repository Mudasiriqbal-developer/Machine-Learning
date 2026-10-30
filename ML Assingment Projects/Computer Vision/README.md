# Project 3: Computer Vision with OpenCV

## 📌 Project Overview
This project explores classical, geometry-based, and feature-based computer vision algorithms implemented using **OpenCV (`cv2`)**. The study covers four fundamental vision operations:
1. **Face and Eye Detection:** Multi-scale object detection using pre-trained **Haar Feature-based Cascade Classifiers** with localized ROI processing.
2. **Feature Detection & Matching:** A comparative investigation of **ORB** (Oriented FAST and Rotated BRIEF) versus **SIFT** (Scale-Invariant Feature Transform) evaluating descriptor robustness, invariance, and matching speed.
3. **Edge Detection:** Multi-stage **Canny Edge Detection** examining Gaussian smoothing, gradient operators, non-maximum suppression, and hysteresis thresholding.
4. **Line Detection:** Geometric shape detection contrasting the **Standard Hough Line Transform** (`HoughLines`) with the **Probabilistic Hough Line Transform** (`HoughLinesP`).

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
│   ├── face_eye_detection_single.png  # [Placeholder: Single face detection result]
│   ├── face_eye_detection_multi.png   # [Placeholder: Multi-face detection result]
│   ├── sift_matches.png               # [Placeholder: SIFT feature matching visualization]
│   ├── orb_matches.png                # [Placeholder: ORB feature matching visualization]
│   ├── canny_edge_comparisons.png     # [Placeholder: Canny threshold comparisons]
│   └── hough_line_comparison.png      # [Placeholder: HoughLines vs HoughLinesP result]
├── computer_vision.ipynb           # Main Jupyter notebook (experiments & analysis)
├── README.md                       # Project documentation
└── REPORT.md                       # Formal analytical report
```

---

## 📊 Dataset & Asset Link / Location

- **Local Images Path:** `data/images/`
- **Haar Cascades Path:** `data/haarcascades/`
- **Asset Descriptions:**
  - `Trump.jpg`, `multiplefaces.jpg`: Face & facial feature detection benchmarks.
  - `Effiel1.png`, `Effiel2.png`: Benchmark image pair undergoing scale and perspective variation for feature matching.
  - `filtered_road.jpg`, `soduku.jpg`, `scan.jpg`: Structured scenes for edge filtering and line transforms.

---

## ⚙️ Installation Steps

1. Navigate to the parent assignment directory:
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

Open and execute all cells within the Jupyter Notebook:
```bash
cd "Computer Vision"
jupyter notebook computer_vision.ipynb
```
All generated visualizations and annotated images will be automatically exported to `outputs/`.

---

## 📈 Evaluation Results & Comparison (Placeholder)

> *Results will be populated during experimental execution.*

### 1. Face & Eye Detection (Haar Cascades)
| Test Image | Faces Detected | Eyes Detected | False Positives | Processing Time (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Single Face (`Trump.jpg`) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| Multi-Face (`multiplefaces.jpg`) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |

### 2. Feature Matching: ORB vs. SIFT
| Algorithm | Keypoints (Img 1) | Keypoints (Img 2) | Good Matches | Computation Time (ms) | Invariance Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SIFT (L2 Norm)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| **ORB (Hamming)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |

### 3. Line Detection: Standard vs. Probabilistic Hough Transform
| Algorithm | Execution Speed | Endpoint Accuracy | Parameter Sensitivity | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Hough (`HoughLines`)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| **Probabilistic Hough (`HoughLinesP`)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |

---

## 👥 Reproducibility & Seed
All splits and stochastic algorithms use `random_state=42`.
