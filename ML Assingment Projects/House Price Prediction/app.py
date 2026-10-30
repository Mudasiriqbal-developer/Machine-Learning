"""
House Price Prediction - Interactive Streamlit Web Application
Course Assignment | Due Date: 3 November 2026

Framework: Streamlit
Task: Serve the trained best regression model (Multiple Linear Regression)
for real-time house price predictions using the required property features.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. Page Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ames Residential House Price Predictor")
st.markdown("""
Estimate the fair market sale price of a residential property in Ames, Iowa, based on 
quality, spatial dimensions, age, bathrooms, and neighborhood characteristics.
""")

# ==============================================================================
# 2. Model & Artifact Loading
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_house_price_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

@st.cache_resource
def load_artifacts():
    """Load serialized model and preprocessing transformers."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

if model is None or preprocessor is None:
    st.error("⚠️ Model artifacts not found. Please ensure `models/best_house_price_model.joblib` and `models/preprocessor.joblib` exist.")
    st.stop()

# ==============================================================================
# 3. User Input Interface
# ==============================================================================
best_model_name = preprocessor.get("best_model_name", "Multiple Linear Regression")
best_rmsle = preprocessor.get("best_rmsle", 0.1440)

st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown(f"""
- **Selected Best Model:** {best_model_name}
- **Selection Criterion:** Lowest Validation RMSLE
- **Validation Metrics:**
  - **RMSLE:** {best_rmsle:.4f}
  - **Validation R²:** 0.8978 (~89.8%)
  - **Validation MSE:** 564,647,736.19
- **Generalization:** Train R² = 0.8968 (no overfitting, stable fit)
- **Dataset:** Ames Housing (Dean De Cock, Truman State University)
""")

st.subheader("📋 Enter Property Characteristics")

col1, col2, col3 = st.columns(3)

# Column 1: Quality & Finishes
with col1:
    st.markdown("#### 🌟 Quality & Finishes")
    overall_qual = st.slider(
        "Overall Material & Finish Quality (1-10)",
        min_value=1,
        max_value=10,
        value=6,
        help="Rates the overall material and finish of the house."
    )
    
    kitchen_qual_label = st.selectbox(
        "Kitchen Quality (KitchenQual)",
        options=["Excellent (Ex)", "Good (Gd)", "Typical/Average (TA)", "Fair (Fa)", "Poor (Po)"],
        index=2,
        help="Kitchen quality assessment."
    )
    
    bsmt_qual_label = st.selectbox(
        "Basement Quality / Height (BsmtQual)",
        options=["Excellent (100+ inches)", "Good (90-99 inches)", "Typical/Average (80-89 inches)", "Fair (70-79 inches)", "Poor (<70 inches)", "No Basement"],
        index=2,
        help="Height of the basement."
    )
    
    year_built = st.number_input(
        "Original Construction Year (YearBuilt)",
        min_value=1870,
        max_value=2024,
        value=2000,
        step=1
    )

# Column 2: Dimensions & Spaces
with col2:
    st.markdown("#### 📐 Living & Lot Dimensions")
    gr_liv_area = st.number_input(
        "Above Grade Living Area (GrLivArea sq ft)",
        min_value=300,
        max_value=5000,
        value=1500,
        step=50
    )
    
    total_bsmt_sf = st.number_input(
        "Total Basement Area (TotalBsmtSF sq ft)",
        min_value=0,
        max_value=4000,
        value=850,
        step=50
    )
    
    lot_area = st.number_input(
        "Lot Size (LotArea sq ft)",
        min_value=1000,
        max_value=100000,
        value=8500,
        step=250
    )
    
    garage_cars = st.slider(
        "Garage Car Capacity (GarageCars)",
        min_value=0,
        max_value=5,
        value=2
    )

# Column 3: Bathrooms & Neighborhood
with col3:
    st.markdown("#### 🚿 Bathrooms & Location")
    full_bath = st.slider("Full Baths Above Grade (FullBath)", min_value=0, max_value=4, value=2)
    half_bath = st.slider("Half Baths Above Grade (HalfBath)", min_value=0, max_value=2, value=1)
    bsmt_full_bath = st.slider("Basement Full Baths (BsmtFullBath)", min_value=0, max_value=3, value=0)
    bsmt_half_bath = st.slider("Basement Half Baths (BsmtHalfBath)", min_value=0, max_value=2, value=0)
    
    neighborhood_options = preprocessor.get("neighborhood_options", [
        "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown",
        "Edwards", "BrkSide", "Sawyer", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Timber",
        "Gilbert", "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"
    ])
    
    neighborhood = st.selectbox(
        "Neighborhood Location",
        options=neighborhood_options,
        index=0 if "CollgCr" in neighborhood_options else 0
    )

# Quality string to numeric mapping
qual_map_labels = {
    "Excellent (Ex)": 5.0,
    "Good (Gd)": 4.0,
    "Typical/Average (TA)": 3.0,
    "Fair (Fa)": 2.0,
    "Poor (Po)": 1.0,
    "Excellent (100+ inches)": 5.0,
    "Good (90-99 inches)": 4.0,
    "Typical/Average (80-89 inches)": 3.0,
    "Fair (70-79 inches)": 2.0,
    "Poor (<70 inches)": 1.0,
    "No Basement": 0.0
}

# ==============================================================================
# 4. Inference Pipeline
# ==============================================================================
st.markdown("---")
if st.button("🔮 Calculate Estimated Sale Price", use_container_width=True, type="primary"):
    with st.spinner("Processing features and computing prediction..."):
        # Map quality inputs
        kitchen_qual_num = qual_map_labels.get(kitchen_qual_label, 3.0)
        bsmt_qual_num = qual_map_labels.get(bsmt_qual_label, 0.0)

        # Build input record
        input_dict = {
            "OverallQual": float(overall_qual),
            "GrLivArea": float(gr_liv_area),
            "TotalBsmtSF": float(total_bsmt_sf),
            "YearBuilt": float(year_built),
            "GarageCars": float(garage_cars),
            "KitchenQual": float(kitchen_qual_num),
            "BsmtQual": float(bsmt_qual_num),
            "FullBath": float(full_bath),
            "HalfBath": float(half_bath),
            "BsmtFullBath": float(bsmt_full_bath),
            "BsmtHalfBath": float(bsmt_half_bath),
            "LotArea": float(lot_area),
            "Neighborhood": neighborhood
        }
        input_df = pd.DataFrame([input_dict])

        # Impute missing values with training medians if needed
        for col, med in preprocessor["feature_medians"].items():
            if col in input_df.columns:
                input_df[col] = input_df[col].fillna(med)

        # Apply log1p transformation to skewed continuous features
        for col in preprocessor["skewed_features"]:
            if col in input_df.columns:
                input_df[col] = np.log1p(input_df[col].astype(float))

        # One-hot encode Neighborhood
        input_encoded = pd.get_dummies(
            input_df,
            columns=preprocessor["cat_features"],
            prefix="Neighborhood",
            drop_first=True,
            dtype=float
        )

        # Align columns to exact training feature set
        input_features = input_encoded.reindex(
            columns=preprocessor["feature_columns"],
            fill_value=0.0
        )

        # Model inference in log scale using the selected best model
        log_pred = model.predict(input_features)[0]

        # Reverse log transformation back to dollar amount
        predicted_price = float(np.expm1(log_pred))

        # Expected error margin based on validation RMSLE (14.4%)
        rmsle_margin = float(best_rmsle)
        lower_bound = predicted_price * (1.0 - rmsle_margin)
        upper_bound = predicted_price * (1.0 + rmsle_margin)

    # Display Results
    st.success("### 🎉 Prediction Successful!")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Estimated Sale Price", f"${predicted_price:,.2f}")
    res_col2.metric("Lower Estimate (-RMSLE)", f"${lower_bound:,.2f}")
    res_col3.metric("Upper Estimate (+RMSLE)", f"${upper_bound:,.2f}")

    total_baths = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
    st.info(f"**Property Summary:** {overall_qual}/10 overall quality, {gr_liv_area} sq ft living area, {total_baths:.1f} total bath equivalents, kitchen rated {kitchen_qual_label.split()[0]}, basement rated {bsmt_qual_label.split()[0]}, located in **{neighborhood}**.")
