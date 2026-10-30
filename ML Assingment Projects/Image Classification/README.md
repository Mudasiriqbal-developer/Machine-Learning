# Project 2: Image Classification (Cars, Cricket Balls, Ice-Cream Cones)

## 📌 Project Overview
This project investigates multiclass image classification across three visually distinct categories:
- **Cars**
- **Cricket Balls**
- **Ice-Cream Cones**

The core objective is to analyze the trade-offs between **Raw Flattened Pixels** (unstructured intensity baseline) and **Histogram of Oriented Gradients (HOG)** (structural edge and gradient descriptors). Each feature representation is paired with two classical classifiers:
1. **Multinomial Logistic Regression**
2. **Support Vector Machine (SVM)**

This yields a comprehensive 2x2 experimental matrix (4 configurations). Performance is evaluated using training and testing accuracy, precision, recall, confusion matrices, and qualitative sample prediction analysis. The best model is serialized with `joblib` and deployed via an interactive **Streamlit** image classification app.

---

## 📁 Folder Structure

```text
Image Classification/
├── data/
│   └── images/
│       ├── Cars/                   # Raw car images (.jpg, .png, .jfif)
│       ├── Cricket ball/           # Raw cricket ball images
│       └── Ice cream cone/         # Raw ice-cream cone images
├── models/
│   ├── .gitkeep
│   ├── best_image_classifier.joblib  # [Placeholder: serialized best classifier]
│   └── class_mapping.joblib          # [Placeholder: serialized class labels]
├── outputs/
│   ├── .gitkeep
│   ├── class_distribution.png        # [Placeholder: Dataset distribution plot]
│   ├── hog_visualization.png         # [Placeholder: Sample HOG gradient maps]
│   ├── confusion_matrices.png        # [Placeholder: 2x2 confusion matrix grid]
│   └── sample_predictions.png        # [Placeholder: Test predictions grid]
├── app.py                          # Streamlit interactive image classification app
├── image_classification.ipynb      # Main Jupyter notebook (training & evaluation)
├── README.md                       # Project documentation
└── REPORT.md                       # Formal analytical report
```

---

## 📊 Dataset Link & Location

- **Local Path:** `data/images/`
  - `data/images/Cars/`
  - `data/images/Cricket ball/`
  - `data/images/Ice cream cone/`
- **Class Labels:**
  - `0`: Car
  - `1`: Cricket Ball
  - `2`: Ice Cream Cone
- **Image Variations:** Diverse resolutions, lighting conditions, background contexts, and viewpoints.

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
Open and execute all cells:
```bash
cd "Image Classification"
jupyter notebook image_classification.ipynb
```

### 2. Running the Streamlit Image Prediction App
Launch the interactive web user interface:
```bash
cd "Image Classification"
streamlit run app.py
```
Open your browser at `http://localhost:8501`, upload an image (`.jpg`, `.jpeg`, `.png`), and inspect the model prediction and confidence score.

---

## 📈 Evaluation Results & Comparison (Placeholder)

> *Results will be populated during experimental training and evaluation.*

| Experiment # | Feature Representation | Classifier | Training Accuracy | Testing Accuracy | Generalization Gap |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Raw Pixels | Logistic Regression | *[TBD]* | *[TBD]* | *[TBD]* |
| **2** | HOG Descriptors | Logistic Regression | *[TBD]* | *[TBD]* | *[TBD]* |
| **3** | Raw Pixels | Support Vector Machine (SVM) | *[TBD]* | *[TBD]* | *[TBD]* |
| **4** | HOG Descriptors | Support Vector Machine (SVM) | *[TBD]* | *[TBD]* | *[TBD]* |

**Best Model Selected:** *[To be determined based on test accuracy and robustness]*

---

## 👥 Reproducibility & Seed
All splits and stochastic algorithms use `random_state=42`.
