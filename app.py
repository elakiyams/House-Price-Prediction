import streamlit as st
import joblib
import pandas as pd


# Load model
model = joblib.load(
    "model/house_price_model.pkl"
)


# Page settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)


# Title
st.title("🏠 House Price Prediction")

st.write(
    "Enter house details to predict the price."
)


# User inputs
area = st.number_input(
    "Area (sq.ft)",
    min_value=500,
    value=1200
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

age = st.number_input(
    "House Age",
    min_value=0,
    max_value=100,
    value=5
)


# Prediction button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Age": [age]
    })

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"Predicted House Price: ₹{prediction:,.0f}"
    )

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "house_price_model.pkl")

model = joblib.load(model_path)
