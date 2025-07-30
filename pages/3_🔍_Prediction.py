import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Anxiety Level Prediction",
    page_icon="🧠",
    layout="wide"
)

# CSS custom untuk tampilan yang lebih baik
st.markdown("""
<style>
    /* Style dasar */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Membuat radio button lebih besar dan mudah diklik */
    .stRadio > div {
        gap: 20px;
    }
    .stRadio > div > label {
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        transition: all 0.3s;
    }
    .stRadio > div > label:hover {
        background-color: #e9f7ef;
        border-color: #2ecc71;
    }
    
    /* Style untuk header */
    .header-style {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Style untuk progress bar custom */
    .progress-container {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 3px;
        margin: 20px 0;
    }
    .progress-bar {
        height: 20px;
        border-radius: 8px;
        background-color: #3498db;
        width: 0%;
        transition: width 0.5s;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("🧠 Anxiety Level Prediction")
st.markdown("""
<div style='background-color:#e8f4f8; padding:15px; border-radius:10px; margin-bottom:20px;'>
    <h3 style='color:#2c3e50;'>Please fill out the form below to get your prediction:</h3>
    <p style='color:#7f8c8d;'>This tool helps assess potential anxiety levels based on lifestyle and health factors.</p>
</div>
""", unsafe_allow_html=True)

# 1. Load Model and Preprocessing
@st.cache_resource
def load_resources():
    model_path = os.path.join('model', 'best_xgb.pkl')
    preprocess_path = os.path.join('model', 'preprocess.pkl')
    return (
        joblib.load(model_path),
        joblib.load(preprocess_path)
    )

model, preprocess = load_resources()

# 2. Input Form - ALL MODEL FEATURES
with st.form("input_form"):
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("<h2 class='header-style'>Personal & Lifestyle Data</h2>", unsafe_allow_html=True)
        
        age = st.slider("**Age**", 18, 80, 30, help="Select your current age")
        sleep = st.slider("**Sleep Hours per Day**", 3.0, 12.0, 7.0, 0.5, help="Average hours of sleep per night")
        activity = st.slider("**Physical Activity (hours/week)**", 0.0, 20.0, 5.0, 0.5, help="Total hours of exercise per week")
        caffeine = st.slider("**Caffeine Intake (mg/day)**", 0, 500, 100, 10, help="Typical daily caffeine consumption")
        alcohol = st.slider("**Alcohol Consumption (drinks/week)**", 0, 20, 0, 1, help="Standard alcoholic drinks per week")
        smoking = st.radio("**Do you smoke?**", ["No", "Yes"], horizontal=True, key="smoking")
        diet = st.slider("**Diet Quality (1-10)**", 1, 10, 6, 1, help="Self-rated diet quality where 10 is healthiest")
        
    with cols[1]:
        st.markdown("<h2 class='header-style'>Mental & Physical Health</h2>", unsafe_allow_html=True)
        
        stress = st.slider("**Stress Level (1-10)**", 1, 10, 5, 1, help="Perceived stress level where 10 is most stressed")
        heart_rate = st.slider("**Heart Rate (bpm)**", 50, 120, 72, 1, help="Resting heart rate in beats per minute")
        breathing = st.slider("**Breathing Rate (breaths/min)**", 10, 30, 16, 1, help="Resting breathing rate")
        sweating = st.slider("**Sweating Level (1-5)**", 1, 5, 2, 1, help="Frequency of excessive sweating")
        
        # Radio button dengan layout yang lebih baik
        st.markdown("<p style='margin-bottom:5px;'><strong>Experience dizziness?</strong></p>", unsafe_allow_html=True)
        dizziness = st.radio("", ["No", "Yes"], horizontal=True, key="dizziness", label_visibility="collapsed")
        
        therapy = st.slider("**Therapy Sessions (per month)**", 0, 10, 0, 1, help="Number of mental health therapy sessions")
        
        st.markdown("<p style='margin-bottom:5px;'><strong>Family history of anxiety?</strong></p>", unsafe_allow_html=True)
        family_history = st.radio("", ["No", "Yes"], horizontal=True, key="family_history", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom:5px;'><strong>Currently on medication?</strong></p>", unsafe_allow_html=True)
        medication = st.radio("", ["No", "Yes"], horizontal=True, key="medication", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom:5px;'><strong>Recent traumatic event?</strong></p>", unsafe_allow_html=True)
        life_event = st.radio("", ["No", "Yes"], horizontal=True, key="life_event", label_visibility="collapsed")
    
    # Submit button dengan styling
    submitted = st.form_submit_button("**Get Prediction**", use_container_width=True, type="primary")

# 3. Prediction Process
if submitted:
    try:
        # Define the correct feature order expected by the model
        correct_feature_order = [
            'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
            'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
            'Smoking', 'Family History of Anxiety', 'Stress Level (1-10)',
            'Heart Rate (bpm)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Dizziness', 'Medication',
            'Therapy Sessions (per month)', 'Recent Major Life Event',
            'Diet Quality (1-10)'
        ]
        
        # Create DataFrame with manual encoding and correct column order
        input_dict = {
            'Age': age,
            'Sleep Hours': sleep,
            'Physical Activity (hrs/week)': activity,
            'Caffeine Intake (mg/day)': caffeine,
            'Alcohol Consumption (drinks/week)': alcohol,
            'Smoking': 1 if smoking == "Yes" else 0,
            'Family History of Anxiety': 1 if family_history == "Yes" else 0,
            'Stress Level (1-10)': stress,
            'Heart Rate (bpm)': heart_rate,
            'Breathing Rate (breaths/min)': breathing,
            'Sweating Level (1-5)': sweating,
            'Dizziness': 1 if dizziness == "Yes" else 0,
            'Medication': 1 if medication == "Yes" else 0,
            'Therapy Sessions (per month)': therapy,
            'Recent Major Life Event': 1 if life_event == "Yes" else 0,
            'Diet Quality (1-10)': diet
        }
        
        # Create DataFrame with correct column order
        input_data = pd.DataFrame([input_dict])[correct_feature_order]
        
        # NUMERICAL SCALING
        numerical_cols = [
            'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
            'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
            'Stress Level (1-10)', 'Heart Rate (bpm)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Therapy Sessions (per month)', 'Diet Quality (1-10)'
        ]
        
        for col in numerical_cols:
            if col in input_data.columns and col in preprocess['scalers']:
                input_data[col] = preprocess['scalers'][col].transform(input_data[[col]])
        
        # Ensure all columns are numeric
        input_data = input_data.astype(float)
        
        # 5. Prediction - Convert to Python float explicitly
        proba = float(model.predict_proba(input_data)[0][1])  # Konversi ke float standard
        
        # 6. Show Results with better visualization
        st.divider()
        st.markdown("<h2 style='text-align:center; color:#2c3e50;'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # Custom progress bar untuk menghindari error float32
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width:{proba*100}%"></div>
        </div>
        <p style="text-align:center; margin-top:5px;"><b>{proba*100:.1f}% Risk Score</b></p>
        """, unsafe_allow_html=True)
        
        if proba >= 0.5:
            st.markdown(f"""
            <div style='background-color:#fde8e8; padding:20px; border-radius:10px; margin-top:20px;'>
                <h3 style='color:#e74c3c;'>🔴 High Risk: {proba*100:.1f}%</h3>
                <h4 style='color:#2c3e50;'>Recommendations:</h4>
                <ul style='color:#2c3e50;'>
                    <li>Consult with a mental health professional</li>
                    <li>Practice relaxation techniques (deep breathing, meditation)</li>
                    <li>Evaluate your sleep patterns and aim for 7-9 hours</li>
                    <li>Consider regular therapy sessions</li>
                    <li>Reduce caffeine and alcohol intake</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#e8f8f0; padding:20px; border-radius:10px; margin-top:20px;'>
                <h3 style='color:#2ecc71;'>🟢 Low Risk: {(1-proba)*100:.1f}%</h3>
                <h4 style='color:#2c3e50;'>Prevention Tips:</h4>
                <ul style='color:#2c3e50;'>
                    <li>Maintain 7-9 hours of quality sleep</li>
                    <li>Engage in regular physical activity (3-5 hours/week)</li>
                    <li>Limit stimulants like caffeine</li>
                    <li>Practice mindfulness and stress management</li>
                    <li>Maintain a balanced diet (fruits, vegetables, whole grains)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Disclaimer with icon
        st.markdown("""
        <div style='background-color:#f2f4f4; padding:15px; border-radius:8px; margin-top:30px;'>
            <p style='color:#7f8c8d; font-size:0.9em;'>
            <span style='font-weight:bold;'>⚠️ Disclaimer:</span> 
            This is a statistical prediction (Dummy Data), not a medical diagnosis. 
            Consult a healthcare professional for proper assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional: Show raw data in expander
        with st.expander("Show input data"):
            st.dataframe(input_data.style.highlight_max(axis=0, color='#f39c12'))
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.warning("Please check your inputs and try again")