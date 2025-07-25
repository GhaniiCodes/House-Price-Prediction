import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components

# Custom CSS with Tailwind CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Load the preprocessor and model
try:
    with open('Preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    model = load_model('model.h5')
except FileNotFoundError:
    st.error("Preprocessor.pkl or model.h5 not found. Please ensure both files are in the same directory as this app.")
    st.stop()

# Streamlit app layout
st.markdown("""
    <div class="bg-gray-100 min-h-screen p-6">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">House Price Prediction</h1>
        <div class="max-w-lg mx-auto bg-white p-6 rounded-lg shadow-lg">
            <p class="text-gray-600 mb-4">Enter the house details to predict its price.</p>
""", unsafe_allow_html=True)

# Input form
with st.form("house_price_form"):
    st.markdown('<div class="space-y-4">', unsafe_allow_html=True)

    # Numerical inputs
    area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000, step=100,
                           help="Enter the area in square feet")
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1,
                               help="Number of bedrooms")
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1,
                                help="Number of bathrooms")
    floors = st.number_input("Floors", min_value=1, max_value=5, value=2, step=1,
                             help="Number of floors")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, step=1,
                                 help="Year the house was built")

    # Categorical inputs
    location = st.selectbox("Location", options=["Downtown", "Suburban", "Rural", "Urban"],
                            help="Select the house location")
    condition = st.selectbox("Condition", options=["Excellent", "Good", "Fair", "Poor"],
                             help="Select the house condition")
    garage = st.selectbox("Garage", options=["Yes", "No"], help="Does the house have a garage?")

    # Submit button
    submitted = st.form_submit_button("Predict Price")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if submitted:
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'YearBuilt': [year_built],
            'Location': [location],
            'Condition': [condition],
            'Garage': [garage]
        })

        # Preprocess input
        preprocessed_input = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(preprocessed_input)[0][0]

        # Display result
        st.markdown(f"""
            <div class="mt-6 p-4 bg-green-100 rounded-lg">
                <h2 class="text-xl font-semibold text-gray-800">Predicted House Price:</h2>
                <p class="text-2xl text-green-600">${prediction:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

st.markdown('</div></div>', unsafe_allow_html=True)