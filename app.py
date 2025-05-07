import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stNumberInput>div>input, .stSlider>div>input {
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #ced4da;
    }
    .stSelectbox>div>select {
        border-radius: 8px;
        padding: 10px;
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #343a40;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model, scaler, and selector
try:
    model = joblib.load('Diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
except FileNotFoundError:
    st.error("🚨 Model or preprocessing files not found. Ensure 'Diabetes_model.pkl', 'scaler.pkl', and 'selector.pkl' are in the correct directory.")
    st.stop()

# App title and header image
st.title("🩺 Diabetes Risk Prediction Platform")
st.image("https://www.cdc.gov/diabetes/images/library/spotlights/diabetes-testing.jpg", use_column_width=True)
st.markdown("**A cutting-edge tool to predict diabetes risk with high accuracy. Input patient data below to get instant results.**")

# Sidebar for inputs
st.sidebar.header("📋 Patient Data Input")
st.sidebar.markdown("Provide the following health metrics to predict diabetes risk.")

# Use columns for organized input layout
col1, col2 = st.sidebar.columns(2)

# Input fields with sliders and number inputs
with col1:
    year = st.number_input('Year', min_value=1900, max_value=2025, value=2023, step=1, help="Enter the year of data collection.")
    age = st.slider('Age', min_value=0, max_value=120, value=30, step=1, help="Patient's age in years.")
    bmi = st.slider('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1, help="Body Mass Index (kg/m²).")
    hba1c_level = st.slider('HbA1c Level (%)', min_value=3.0, max_value=15.0, value=5.5, step=0.1, help="Hemoglobin A1c level.")

with col2:
    blood_glucose_level = st.slider('Blood Glucose Level (mg/dL)', min_value=50, max_value=300, value=100, step=1, help="Fasting blood glucose level.")
    hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Does the patient have hypertension?")
    heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Does the patient have heart disease?")
    loc_freq = st.slider('Location Frequency', min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Frequency of patient's location (0 to 1).")

# Single race selection
st.sidebar.subheader("Race")
race = st.sidebar.selectbox('Race', ['African American', 'Asian', 'Caucasian', 'Hispanic', 'Other'], help="Select the patient's race.")
race_african_american = 1 if race == 'African American' else 0
race_asian = 1 if race == 'Asian' else 0
race_caucasian = 1 if race == 'Caucasian' else 0
race_hispanic = 1 if race == 'Hispanic' else 0
race_other = 1 if race == 'Other' else 0

# Gender selection
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'], help="Select the patient's gender.")
gender_female = 1 if gender == 'Female' else 0
gender_male = 1 if gender == 'Male' else 0
gender_other = 1 if gender == 'Other' else 0

# Calculate derived features
age_bmi = age * bmi
comorbidity_count = hypertension + heart_disease

# Create DataFrame for prediction
df = pd.DataFrame({
    'year': [year],
    'age': [age],
    'race:AfricanAmerican': [race_african_american],
    'race:Asian': [race_asian],
    'race:Caucasian': [race_caucasian],
    'race:Hispanic': [race_hispanic],
    'race:Other': [race_other],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'bmi': [bmi],
    'hbA1c_level': [hba1c_level],
    'blood_glucose_level': [blood_glucose_level],
    'gender_Female': [gender_female],
    'gender_Male': [gender_male],
    'gender_Other': [gender_other],
    'loc_freq': [loc_freq],
    'age_bmi': [age_bmi],
    'comorbidity_count': [comorbidity_count]
})

# Display input summary
with st.expander("📊 View Input Summary"):
    st.write("**Patient Data Summary**")
    try:
        st.dataframe(df.style.format("{:.2f}", subset=['year', 'age', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'loc_freq', 'age_bmi', 'comorbidity_count']))
    except KeyError as e:
        st.error(f"🚨 Column mismatch in DataFrame: {e}. Please check the column names.")
        st.stop()

# Preprocess the input data
try:
    num_cols = ['year', 'age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
    df[num_cols] = scaler.transform(df[num_cols])
    X_selected = selector.transform(df)
except Exception as e:
    st.error(f"🚨 Preprocessing error: {e}")
    st.stop()

# Prediction button with confirmation
if st.sidebar.button('Analyze & Predict', key='predict_button'):
    with st.spinner("🔄 Analyzing data and generating prediction..."):
        try:
            # Make prediction
            prediction = model.predict(X_selected)
            prediction_proba = model.predict_proba(X_selected)[0]

            # Display result
            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.markdown('<div class="success-box">✅ The patient is predicted to be Healthy</div>', unsafe_allow_html=True)
                st.write(f"**Confidence (Healthy):** {prediction_proba[0]:.2%}")
            else:
                st.markdown('<div class="error-box">⚠️ The patient is predicted to have Diabetes</div>', unsafe_allow_html=True)
                st.write(f"**Confidence (Diabetes):** {prediction_proba[1]:.2%}")

            # Radar chart for input features
            st.subheader("Input Feature Visualization")
            radar_cols = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'loc_freq']
            radar_values = df[radar_cols].iloc[0].values
            fig = go.Figure(data=go.Scatterpolar(
                r=radar_values,
                theta=radar_cols,
                fill='toself',
                name='Patient Data'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Patient Health Metrics"
            )
            st.plotly_chart(fig)

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                selected_features = df.columns[selector.get_support()]
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title="Feature Importance in Prediction",
                             color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig)

            # Downloadable prediction report
            report = f"""
Diabetes Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Prediction: {'Healthy' if prediction[0] == 0 else 'Diabetes'}
Confidence: {prediction_proba[1 if prediction[0] == 1 else 0]:.2%}
Input Data:
{df.to_string(index=False)}
"""
            st.download_button(
                label="📥 Download Prediction Report",
                data=report,
                file_name="diabetes_prediction_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"🚨 Prediction error: {e}")

# Instructions and about section
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    1. Enter the patient’s health metrics in the sidebar using sliders and dropdowns.
    2. Select the patient’s gender and race (only one race can be selected).
    3. Click the **Analyze & Predict** button to generate a diabetes risk prediction.
    4. View the results, including confidence scores, a radar chart of input features, and feature importance.
    5. Download a detailed prediction report for your records.
    """)

with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app uses a Random Forest model trained on preprocessed health data to predict diabetes risk.
    The model leverages features like age, BMI, HbA1c, and more, with advanced preprocessing for accuracy.
    Developed for healthcare professionals and researchers to assist in early diabetes detection.
    """)