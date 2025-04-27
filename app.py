import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Cancer logistic model.pkl')

# Set up the Streamlit app layout
st.title("Cancer Prediction App")

# Input fields for user data
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
area_error = st.number_input("Area Error", min_value=0.0)
worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0)
mean_area = st.number_input("Mean Area", min_value=0.0)
worst_area = st.number_input("Worst Area", min_value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    try:
        # Prepare input for prediction
        features = np.array([[mean_perimeter, area_error, worst_perimeter, mean_area, worst_area]])
        
        # Make prediction (0 for Benign, 1 for Malignant)
        prediction = model.predict(features)[0]
        
        # Convert numerical result to label
        if prediction == 1:
            result = "Malignant"
        else:
            result = "Benign"
        
        # Display the result
        st.success(f"The result is: {result}")
        
    except ValueError:
        # Display an error message if invalid data is entered
        st.error("Please enter valid numerical values for all fields.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #999999; font-size: 12px; padding-top: 10px;'>
        © Ireri Mugambi 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
