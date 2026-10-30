import streamlit as st, cv2, numpy as np, joblib
from skimage.feature import hog
from PIL import Image

st.title("Image Classification")
model = joblib.load("models/best_model.pkl")
categories = ["Cars", "Cricket ball", "Ice cream cone"]
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if file:
    img = np.array(Image.open(file).convert("L"))
    img = cv2.resize(img, (64, 64)) / 255.0
    feat = hog(img).reshape(1, -1)
    pred = model.predict(feat)[0]
    st.image(file, width=200)
    st.write(f"Predicted Class: **{categories[pred]}**")
