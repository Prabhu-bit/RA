import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import io
import sys

# Add src directory to Python path for importing preprocess module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, validate_input_data, convert_to_numeric, convert_categorical, standardize_gender

# File paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')

# Load the preprocessing pipeline and the trained model
@st.cache_resource
def load_resources():
    preprocessor = joblib.load(PREPROCESSING_PIPELINE_PATH)
    model = joblib.load(XGBOOST_MODEL_PATH)
    return preprocessor, model

preprocessor, model = load_resources()

# Map numerical labels back to class names
CLASS_LABELS = {0: 'Healthy', 1: 'Seropositive RA', 2: 'Seronegative RA'}

def format_errors(errors):
    """Format error messages for display."""
    if not errors:
        return None
    
    error_message = "Please correct the following issues:\n"
    for error in errors:
        error_message += f"- {error}\n"
    return error_message

def get_user_input():
    st.sidebar.header('Patient Data Input')
    input_method = st.sidebar.radio("Choose input method:", ('Form Input', 'CSV String Input', 'Upload Clinical Report'))

    df = pd.DataFrame(columns=ALL_FEATURES)

    if input_method == 'Form Input':
        input_data = {}
        
        # Numeric inputs with proper validation
        input_data['Age'] = st.sidebar.number_input('Age', min_value=0, max_value=120, value=45)
        input_data['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        input_data['ESR'] = st.sidebar.number_input('ESR (mm/hr)', min_value=0.0, value=20.0, format="%.1f")
        input_data['CRP'] = st.sidebar.number_input('CRP (mg/L)', min_value=0.0, value=5.0, format="%.1f")
        input_data['RF'] = st.sidebar.number_input('RF (IU/mL)', min_value=0.0, value=15.0, format="%.1f")
        input_data['Anti-CCP'] = st.sidebar.number_input('Anti-CCP (U/mL)', min_value=0.0, value=10.0, format="%.1f")
        
        # Categorical inputs with consistent options
        categorical_options = ['Negative', 'Positive']
        input_data['HLA-B27'] = st.sidebar.selectbox('HLA-B27', categorical_options)
        input_data['ANA'] = st.sidebar.selectbox('ANA', categorical_options)
        input_data['Anti-Ro'] = st.sidebar.selectbox('Anti-Ro', categorical_options)
        input_data['Anti-La'] = st.sidebar.selectbox('Anti-La', categorical_options)
        input_data['Anti-dsDNA'] = st.sidebar.selectbox('Anti-dsDNA', categorical_options)
        input_data['Anti-Sm'] = st.sidebar.selectbox('Anti-Sm', categorical_options)
        
        # Additional numeric inputs with proper validation
        input_data['C3'] = st.sidebar.number_input('C3 (g/L)', min_value=0.0, value=1.0, format="%.2f")
        input_data['C4'] = st.sidebar.number_input('C4 (g/L)', min_value=0.0, value=0.2, format="%.2f")
        
        df = pd.DataFrame([input_data])

    elif input_method == 'CSV String Input':
        st.sidebar.markdown("Enter comma-separated values for the following features:")
        feature_list = ", ".join(ALL_FEATURES)
        st.sidebar.markdown(f"**Required columns:** {feature_list}")
        st.sidebar.markdown("""
        **Format guidelines:**
        - Numeric values: plain numbers (e.g., 45, 20.5)
        - Gender: 'Male' or 'Female'
        - Test results: 'Positive'/'Negative' or 1/0
        """)
        
        default_csv = "45,Male,20.0,5.0,15.0,10.0,Negative,Negative,Negative,Negative,Negative,Negative,1.0,0.2"
        csv_string = st.sidebar.text_area("CSV Data", default_csv, height=100)
        
        if csv_string:
            try:
                csv_file = io.StringIO(csv_string)
                df = pd.read_csv(csv_file, header=None, names=ALL_FEATURES)
                
                # Validate input data
                errors = validate_input_data(df)
                if errors:
                    error_message = format_errors(errors)
                    st.sidebar.error(error_message)
                    return pd.DataFrame()
                
            except Exception as e:
                st.sidebar.error(f"Error parsing CSV data: {str(e)}")
                st.sidebar.markdown("""
                **Common issues:**
                - Incorrect number of values
                - Invalid numeric values
                - Missing commas
                - Extra whitespace
                """)
                return pd.DataFrame()

    elif input_method == 'Upload Clinical Report':
        st.sidebar.warning("⚠️ Clinical report upload feature is in development.")
        uploaded_file = st.sidebar.file_uploader("Upload Clinical Report", type=["pdf", "docx"], disabled=False)
        
        if uploaded_file is not None:
            st.sidebar.info("This feature will be available in a future update.")
            return pd.DataFrame()

    # Validate all input data before returning
    if not df.empty:
        errors = validate_input_data(df)
        if errors:
            error_message = format_errors(errors)
            st.sidebar.error(error_message)
            return pd.DataFrame()
            
    return df

def main():
    st.set_page_config(layout="wide", page_title="RA Subtype Classifier")
    st.title('Rheumatoid Arthritis Subtype Classifier')
    st.markdown("A multimodal AI system for classifying RA subtypes and healthy controls.")
    
    # Add information about the model and features
    with st.expander("ℹ️ About this classifier"):
        st.markdown("""
        This classifier helps identify Rheumatoid Arthritis (RA) subtypes using clinical and laboratory data.
        
        **Available features:**
        - **Demographics:** Age, Gender
        - **Lab Tests:** ESR, CRP, RF, Anti-CCP
        - **Autoantibodies:** HLA-B27, ANA, Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm
        - **Complement Levels:** C3, C4
        
        **Prediction classes:**
        - Healthy
        - Seropositive RA
        - Seronegative RA
        """)

    user_input_df = get_user_input()

    if not user_input_df.empty:
        # Show input validation button
        if st.sidebar.button('Validate Input'):
            errors = validate_input_data(user_input_df)
            if errors:
                st.error(format_errors(errors))
            else:
                st.success("✅ All input data is valid!")
        
        # Show classify button
        if st.sidebar.button('Classify'):
            try:
                # Show the preprocessed input data
                st.subheader("Input Data")
                st.dataframe(user_input_df)
                
                # Process and predict
                processed_input = preprocessor.transform(user_input_df)
                prediction_proba = model.predict_proba(processed_input)
                predicted_class_idx = np.argmax(prediction_proba)
                predicted_class_label = CLASS_LABELS[predicted_class_idx]
                confidence = prediction_proba[0][predicted_class_idx]

                # Display results in a clean format
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("🔍 Prediction Result")
                    st.markdown(f"**Predicted Class:** {predicted_class_label}")
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
                
                with col2:
                    # Create a bar chart for probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    classes = list(CLASS_LABELS.values())
                    probabilities = prediction_proba[0]
                    ax.bar(classes, probabilities)
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Add probability details in an expander
                with st.expander("📊 Detailed Probabilities"):
                    probs_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    st.dataframe(probs_df.style.format({'Probability': '{:.2%}'}))
                
            except Exception as e:
                st.error("❌ Error during prediction")
                st.error(f"Details: {str(e)}")
                st.info("Please ensure all input values are in the correct format and try again.")
    
    else:
        st.info("👈 Please enter patient data using the sidebar to start.")

    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    **Note:** This tool is for research purposes only and should not be used as the sole basis for medical decisions.
    Always consult with qualified healthcare professionals for diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()