import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and label encoder
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    lecolor = joblib.load('lecolor.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'random_forest_model.pkl', 'scaler.pkl', and 'lecolor.pkl' are in the same directory.")
    st.stop()

# Define the star type labels
startypelabels = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# Use the loaded LabelEncoder's classes to create a mapping for the web app
starcolor_mapping = {label: i for i, label in enumerate(lecolor.classes_)}

# Streamlit App UI
st.title("Star Type Classifier")
st.write("Enter the physical properties of a star to classify its type.")

# Sidebar for user input
with st.sidebar:
    st.header("Star Properties Input")
    
    temperature = st.number_input("Temperature (K)", min_value=1.0, value=5000.0, format="%.2f")
    luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0, value=0.5, format="%.2f")
    radius = st.number_input("Radius (R/Ro)", min_value=0.0, value=0.5, format="%.2f")
    absmag = st.number_input("Absolute Magnitude (Mv)", value=5.0, format="%.2f")
    
    # Use the keys from the mapping for the dropdown menu
    color_options = sorted(list(starcolor_mapping.keys()))
    starcolor_label = st.selectbox("Star Color", options=color_options)
    
# Prediction button
if st.button("Classify Star"):
    # Create a DataFrame from user inputs
    input_df = pd.DataFrame([[temperature, luminosity, radius, absmag, starcolor_mapping[starcolor_label]]],
                            columns=['Temperature', 'Luminosity', 'Radius', 'AbsoluteMagnitude', 'StarColorEncoded'])
    
    # Scale the input data using the pre-trained scaler
    scaled_input = scaler.transform(input_df)
    
    # Make a prediction
    prediction_index = model.predict(scaled_input)[0]
    predicted_star_type = startypelabels.get(prediction_index, "Unknown Star Type")

    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"Based on the provided properties, the star is most likely a: **{predicted_star_type}**")
    
    st.markdown("---")
    st.subheader("Input Values")
    st.table(input_df)
