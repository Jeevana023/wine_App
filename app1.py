import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# =========================
# MUST BE FIRST STREAMLIT COMMAND
# =========================
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")


# =========================
# Background Image Function
# =========================
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Improve readability */
        h1, h2, h3, p, label {{
            color: white !important;
            font-weight: bold;
        }}

        /* Input box styling */
        input {{
            background-color: rgba(255,255,255,0.85) !important;
            border-radius: 8px;
        }}

        /* Button styling */
        .stButton > button {{
            background-color: #8B0000;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Apply Background
# =========================
set_background("wine.jpg")


# =========================
# Load Model & Scaler
# =========================
@st.cache_resource
def load_artifacts():
    with open("finalRF_model.sav", "rb") as f:
        model = pickle.load(f)

    with open("scaler_model.sav", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()

# =========================
# App UI
# =========================
st.title("🍷 Wine Quality Prediction App")
st.title("EDUNET FOUNDATION BY SAP")
st.write("Enter wine chemical properties to predict quality")

st.divider()

# =========================
# User Inputs
# =========================
fixed_acidity = st.number_input("Fixed Acidity", 0.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0)
citric_acid = st.number_input("Citric Acid", 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0)
chlorides = st.number_input("Chlorides", 0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
density = st.number_input("Density", 0.0)
pH = st.number_input("pH", 0.0)
sulphates = st.number_input("Sulphates", 0.0)
alcohol = st.number_input("Alcohol", 0.0)

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Wine Quality"):

    input_df = pd.DataFrame(
        [
            {
                "fixed acidity": fixed_acidity,
                "volatile acidity": volatile_acidity,
                "citric acid": citric_acid,
                "residual sugar": np.log(residual_sugar + 1),
                "chlorides": chlorides,
                "free sulfur dioxide": np.log(free_sulfur_dioxide + 1),
                "total sulfur dioxide": np.log(total_sulfur_dioxide + 1),
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol,
            }
        ]
    )

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🍷 Predicted Wine Quality: **{prediction}**")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled).max()
        st.info(f"Prediction Confidence: **{prob:.2f}**")
