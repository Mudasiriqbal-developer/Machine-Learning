"""
House Price Prediction - Interactive Web Application Stub
Course Assignment | Due Date: 3 November 2026

Framework: Streamlit
Task: Serve the trained regression model for real-time house price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==============================================================================
# Application Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Ames House Price Prediction App")
st.markdown("""
Predict the estimated sale price of a residential property based on key architectural,
spatial, and quality features using our trained regression model.
""")

# ==============================================================================
# Model & Asset Loading Stub
# ==============================================================================
MODEL_PATH = os.path.join("models", "best_house_price_model.joblib")
ENCODER_PATH = os.path.join("models", "neighborhood_encoder.joblib")

@st.cache_resource
def load_model_artifacts():
    """Load serialized model and preprocessors from the models/ directory."""
    # STUB: To be implemented after model training in house_price_prediction.ipynb
    model = None
    encoder = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, encoder = load_model_artifacts()

# ==============================================================================
# User Input Form (Placeholder / Scaffold)
# ==============================================================================
st.subheader("Property Characteristics")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=6)
    gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=300, max_value=6000, value=1500, step=50)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=850, step=50)
    year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2000, step=1)

with col2:
    garage_cars = st.slider("Garage Car Capacity", min_value=0, max_value=5, value=2)
    full_bath = st.slider("Full Bathrooms Above Grade", min_value=0, max_value=4, value=2)
    lot_area = st.number_input("Lot Size (sq ft)", min_value=1000, max_value=100000, value=8000, step=100)
    neighborhood = st.selectbox(
        "Neighborhood",
        options=["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "Edwards"]
    )

# ==============================================================================
# Prediction Logic (Stub)
# ==============================================================================
if st.button("🔮 Estimate Sale Price", use_container_width=True):
    if model is None:
        st.warning("⚠️ Model artifact not found in `models/`. Please train and save the model via the notebook first.")
    else:
        # Preprocessing & inference pipeline will be finalized when model is trained
        st.info("Prediction pipeline will run here once model training is completed.")
