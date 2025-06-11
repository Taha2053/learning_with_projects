import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model = joblib.load('house_price_model.pkl')
except FileNotFoundError:
    st.error("Error: 'house_price_model.pkl' not found. Make sure the model artifact is in the same directory.")
    st.stop()


try:
    scaler = joblib.load('house_scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'house_scaler.pkl' not found. Make sure the scaler artifact is in the same directory.")
    st.stop()


try:
    model_features = joblib.load('house_model_features.pkl')
except FileNotFoundError:
    st.error("Error: 'house_model_features.pkl' not found. Make sure the feature list artifact is in the same directory.")
    st.stop()

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")

st.write("""
This app predicts house prices based on various property features using a trained CatBoost Regression model.
Please enter the details of the house below:
""")


st.sidebar.header("House Features")

area = st.sidebar.slider("Area (in sq ft)", min_value=500, max_value=15000, value=2500, step=50)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=6, value=3, step=1)
bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=4, value=2, step=1)
stories = st.sidebar.slider("Number of Stories", min_value=1, max_value=4, value=2, step=1)
parking = st.sidebar.slider("Number of Parking Spaces", min_value=0, max_value=3, value=1, step=1)

mainroad = st.sidebar.radio("Proximity to Main Road", ['Yes', 'No'])
guestroom = st.sidebar.radio("Guest Room Available", ['Yes', 'No'])
basement = st.sidebar.radio("Basement Available", ['Yes', 'No'])
hotwaterheating = st.sidebar.radio("Hot Water Heating System", ['Yes', 'No'])
airconditioning = st.sidebar.radio("Air Conditioning", ['Yes', 'No'])
prefarea = st.sidebar.radio("Preferred Area", ['Yes', 'No'])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['Furnished', 'Semi-furnished', 'Unfurnished'])

mainroad_yes = 1 if mainroad == 'Yes' else 0
guestroom_yes = 1 if guestroom == 'Yes' else 0
basement_yes = 1 if basement == 'Yes' else 0
hotwaterheating_yes = 1 if hotwaterheating == 'Yes' else 0
airconditioning_yes = 1 if airconditioning == 'Yes' else 0
prefarea_yes = 1 if prefarea == 'Yes' else 0

furnishingstatus_semi_furnished = 0
furnishingstatus_unfurnished = 0
if furnishingstatus == 'Semi-furnished':
    furnishingstatus_semi_furnished = 1
elif furnishingstatus == 'Unfurnished':
    furnishingstatus_unfurnished = 1

input_data = pd.DataFrame([[
    area,
    bedrooms,
    bathrooms,
    stories,
    parking,
    mainroad_yes,
    guestroom_yes,
    basement_yes,
    hotwaterheating_yes,
    airconditioning_yes,
    prefarea_yes,
    furnishingstatus_semi_furnished,
    furnishingstatus_unfurnished
]], columns=[
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_yes', 'guestroom_yes', 'basement_yes', 'hotwaterheating_yes',
    'airconditioning_yes', 'prefarea_yes', 'furnishingstatus_semi-furnished',
    'furnishingstatus_unfurnished'
])

input_data = input_data.reindex(columns=model_features, fill_value=0)

numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad_yes', 'guestroom_yes', 'hotwaterheating_yes', 'basement_yes', 'airconditioning_yes', 'prefarea_yes', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']

input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

if st.sidebar.button("Predict Price"):
    st.subheader("Prediction Results:")
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"The predicted house price is: **${predicted_price:,.2f}**")
        st.write("---")
        st.info("Please note: This prediction is based on the trained model and the provided inputs. Actual prices may vary.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

