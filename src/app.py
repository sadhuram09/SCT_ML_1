import streamlit as st
import numpy as np
import joblib

# Load your trained regression model
model = joblib.load('./models/final_linear_model.pkl')

st.title("House Price Prediction App")
st.write("Use the form below to enter house features for sale price prediction:")

# User inputs for features
gr_liv_area = st.number_input('Above Ground Living Area (sq ft)', min_value=300, max_value=6000, value=1500)
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=3)
full_bath = st.number_input('Full Bathrooms', min_value=0, max_value=5, value=2)
half_bath = st.number_input('Half Bathrooms', min_value=0, max_value=3, value=1)
lot_area = st.number_input('Lot Area (sq ft)', min_value=1000, max_value=200000, value=12000)
overall_qual = st.number_input('Overall Quality (1-10)', min_value=1, max_value=10, value=5)
year_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=1990)

# Calculate total bathrooms as in model
total_bath = full_bath + 0.5 * half_bath

# Prepare features array for prediction
features = np.array([[gr_liv_area, bedrooms, total_bath, lot_area, overall_qual, year_built]])

# Predict button and output
if st.button('Predict Sale Price'):
    prediction = model.predict(features)[0]
    st.success(f'Estimated Sale Price: ${float(prediction):,.2f}')
