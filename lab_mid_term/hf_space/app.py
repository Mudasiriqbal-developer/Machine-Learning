import threading
import time
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import uvicorn
from api import app as fastapi_app

API_URL = "http://127.0.0.1:8000"


@st.cache_resource
def start_fastapi():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=1)
        if r.status_code == 200:
            return
    except Exception:
        pass

    config = uvicorn.Config(app=fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    start_time = time.time()
    while time.time() - start_time < 15:
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)


start_fastapi()

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.title("Adult Income Prediction")
st.write("Predict whether an individual earns more than $50,000 per year based on census profile features.")


@st.cache_data
def get_categorical_options():
    csv_path = Path(__file__).parent / "data" / "adult.csv"
    df = pd.read_csv(csv_path, skipinitialspace=True, na_values="?")
    df.columns = df.columns.str.strip()
    return {
        "workclass": sorted([str(x) for x in df["workclass"].dropna().unique()]),
        "education": sorted([str(x) for x in df["education"].dropna().unique()]),
        "marital_status": sorted([str(x) for x in df["marital-status"].dropna().unique()]),
        "occupation": sorted([str(x) for x in df["occupation"].dropna().unique()]),
        "relationship": sorted([str(x) for x in df["relationship"].dropna().unique()]),
        "race": sorted([str(x) for x in df["race"].dropna().unique()]),
        "sex": sorted([str(x) for x in df["sex"].dropna().unique()]),
        "native_country": sorted([str(x) for x in df["native-country"].dropna().unique()]),
    }


cat_options = get_categorical_options()

# Fetch metrics from API
api_online = False
metrics_data = None
try:
    resp = requests.get(f"{API_URL}/metrics", timeout=2)
    if resp.status_code == 200:
        metrics_data = resp.json()
        api_online = True
    else:
        st.warning(f"FastAPI responded with status code {resp.status_code}.")
except Exception:
    st.error("FastAPI backend is currently offline. Please start it using: python -m uvicorn api:app --port 8000")

# Model test metrics display
st.subheader("Model Performance & Information")
if api_online and metrics_data:
    st.markdown(f"**Best Model:** {metrics_data.get('model_name', 'N/A')}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
    m2.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
    m3.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
    m4.metric("F1 Score", f"{metrics_data.get('f1', 0):.4f}")
    m5.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0):.4f}")

    top5 = metrics_data.get("top5_features", [])
    if top5:
        st.write("#### Top 5 Important Features")
        top5_df = pd.DataFrame(top5)
        st.bar_chart(data=top5_df.set_index("feature")["importance"])
else:
    st.info("Start the FastAPI backend to view live model metrics and feature importance.")

st.markdown("---")
st.subheader("Input Person Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=39, step=1)
    workclass = st.selectbox(
        "Workclass",
        options=cat_options["workclass"],
        index=cat_options["workclass"].index("Private") if "Private" in cat_options["workclass"] else 0
    )
    education = st.selectbox(
        "Education",
        options=cat_options["education"],
        index=cat_options["education"].index("Bachelors") if "Bachelors" in cat_options["education"] else 0
    )
    education_num = st.number_input("Education Num (Years)", min_value=1, max_value=16, value=13, step=1)
    marital_status = st.selectbox(
        "Marital Status",
        options=cat_options["marital_status"],
        index=cat_options["marital_status"].index("Never-married") if "Never-married" in cat_options["marital_status"] else 0
    )

with col2:
    occupation = st.selectbox(
        "Occupation",
        options=cat_options["occupation"],
        index=cat_options["occupation"].index("Adm-clerical") if "Adm-clerical" in cat_options["occupation"] else 0
    )
    relationship = st.selectbox(
        "Relationship",
        options=cat_options["relationship"],
        index=cat_options["relationship"].index("Not-in-family") if "Not-in-family" in cat_options["relationship"] else 0
    )
    race = st.selectbox(
        "Race",
        options=cat_options["race"],
        index=cat_options["race"].index("White") if "White" in cat_options["race"] else 0
    )
    sex = st.selectbox(
        "Sex",
        options=cat_options["sex"],
        index=cat_options["sex"].index("Male") if "Male" in cat_options["sex"] else 0
    )

with col3:
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=10000, value=0, step=100)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, step=1)
    native_country = st.selectbox(
        "Native Country",
        options=cat_options["native_country"],
        index=cat_options["native_country"].index("United-States") if "United-States" in cat_options["native_country"] else 0
    )

if st.button("Predict Income"):
    if not api_online:
        st.error("Cannot predict: FastAPI backend is offline. Start the API first.")
    else:
        payload = {
            "age": int(age),
            "workclass": str(workclass),
            "education": str(education),
            "education_num": int(education_num),
            "marital_status": str(marital_status),
            "occupation": str(occupation),
            "relationship": str(relationship),
            "race": str(race),
            "sex": str(sex),
            "capital_gain": int(capital_gain),
            "capital_loss": int(capital_loss),
            "hours_per_week": int(hours_per_week),
            "native_country": str(native_country)
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                label = result["label"]
                prob = result["probability"]
                pred = result["prediction"]

                if pred == 1:
                    st.success(f"### Prediction: {label} (High Income)")
                else:
                    st.info(f"### Prediction: {label} (Low/Moderate Income)")
                st.write(f"**Probability of earning >$50K:** {prob * 100:.2f}% ({prob:.4f})")
            else:
                st.error(f"Prediction error ({resp.status_code}): {resp.text}")
        except Exception:
            st.error("Prediction request failed. Please verify that the API is running at http://127.0.0.1:8000.")
