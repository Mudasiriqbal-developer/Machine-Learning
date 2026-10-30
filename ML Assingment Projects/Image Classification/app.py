"""
Image Classification - Interactive Web Application Stub
Course Assignment | Due Date: 3 November 2026

Framework: Streamlit
Task: Classify uploaded images into Cars, Cricket Balls, or Ice-Cream Cones using the best trained model.
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image

# ==============================================================================
# Page Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="Image Classifier (Cars, Cricket Balls, Ice Cream)",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ 3-Class Image Classification App")
st.markdown("""
Upload an image of a **Car**, **Cricket Ball**, or **Ice-Cream Cone** to predict its class using
our trained classical machine learning pipeline.
""")

# ==============================================================================
# Model & Mapping Loader Stub
# ==============================================================================
MODEL_PATH = os.path.join("models", "best_image_classifier.joblib")
CLASS_MAPPING_PATH = os.path.join("models", "class_mapping.joblib")

# Default class label dictionary
CLASSES = {0: "Car", 1: "Cricket Ball", 2: "Ice Cream Cone"}

@st.cache_resource
def load_classifier():
    """Load serialized classifier and label mapping."""
    # STUB: To be implemented after model training in image_classification.ipynb
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    return model

model = load_classifier()

# ==============================================================================
# Image Preprocessing & Feature Extraction Function Stub
# ==============================================================================
def preprocess_and_extract_features(image_array):
    """
    STUB: Preprocess uploaded image and extract feature representation (HOG / Raw Pixels).
    Must match the exact preprocessing applied during training in image_classification.ipynb.
    """
    # Placeholder: convert to grayscale, resize, and extract HOG
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (100, 100))
    # Features will be extracted using skimage.feature.hog or raw flattening
    return resized.flatten().reshape(1, -1)

# ==============================================================================
# User Interface & File Upload
# ==============================================================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a car, cricket ball, or ice-cream cone."
)

if uploaded_file is not None:
    # Read image using PIL and convert to numpy array
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if st.button("🔍 Predict Class", use_container_width=True):
            if model is None:
                st.warning("⚠️ Model artifact not found in `models/`. Please train and save the model via the notebook first.")
            else:
                features = preprocess_and_extract_features(image_np)
                prediction = model.predict(features)[0]
                pred_label = CLASSES.get(prediction, f"Class {prediction}")
                st.success(f"🎯 Predicted Category: **{pred_label}**")
