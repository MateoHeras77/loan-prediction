import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set page configuration
st.set_page_config(
    page_title="Loan Risk Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #FFEBEE;
        border: 2px solid #EF5350;
    }
    .low-risk {
        background-color: #E8F5E9;
        border: 2px solid #66BB6A;
    }
    .feature-section {
        padding: 1.5rem;
        background-color: #F5F5F5;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #9E9E9E;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Loan Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Enter customer details to predict loan risk level</p>", unsafe_allow_html=True)

# Custom unpickler for compatibility
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect module if it is the old numpy internal module
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)

# Load dataset to get possible values for categorical variables
@st.cache_data
def load_dataset():
    try:
        with open('data/processed/df_loan.pkl', 'rb') as f:
            df = CustomUnpickler(f).load()
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_risk_classifier.pkl')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load dataset and model
df = load_dataset()
model = load_model()

# Get unique values for categorical variables
if df is not None:
    profession_options = df['Profession'].unique().tolist()
    city_options = df['CITY'].unique().tolist()[:20]  # Limit to 20 cities to avoid overwhelming UI
    state_options = df['STATE'].unique().tolist()
    house_ownership_options = df['House_Ownership'].unique().tolist()
else:
    # Default options if dataset couldn't be loaded
    profession_options = ["Accountant", "Architect", "Doctor", "Engineer", "Lawyer", "Manager", "Mechanic", "Teacher"]
    city_options = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
    state_options = ["Karnataka", "Tamil Nadu", "Delhi", "Telangana", "West Bengal", "Maharashtra"]
    house_ownership_options = ["Rented", "Owned", "Norent_noown"]

# Sidebar - Explanation and Documentation
with st.sidebar:
    st.title("About the App")
    # add logo
    st.image("https://wpvip.guscancolleges.ca/unfc/wp-content/uploads/sites/7/2025/03/UNF-primary-logo.svg", use_container_width=True)
    st.info("""
    This application predicts the risk level of a loan applicant based on their personal and financial information.
    
    **Risk Levels:**
    - **Low Risk**: Applicant is likely to repay the loan on time
    - **High Risk**: Applicant has higher probability of defaulting on the loan
    
    The prediction is made using a machine learning model trained on historical loan data.
    """)
    
    # Display dataset info if loaded
    if df is not None:
        st.subheader("Dataset Information")
        st.write(f"Original dataset size: {df.shape}")
        st.write(f"Risk distribution: {df['Risk_Flag'].value_counts().to_dict()}")

# Main form
with st.form("prediction_form"):
    st.write("### Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        profession = st.selectbox("Profession", options=profession_options)
    
    with col2:
        income = st.number_input("Income", min_value=0, value=50000)
        experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=5)
        house_ownership = st.selectbox("House Ownership", options=house_ownership_options)
    
    st.write("### Additional Details")
    col3, col4 = st.columns(2)
    
    with col3:
        car_ownership = st.selectbox("Car Ownership", ["Yes", "No"])
        current_house_yrs = st.number_input("Current House Years", min_value=0, max_value=50, value=2)
        current_job_yrs = st.number_input("Current Job Years", min_value=0, max_value=50, value=3)
    
    with col4:
        city = st.selectbox("City", options=city_options)
        state = st.selectbox("State", options=state_options)
    
    # Submit button
    submit_button = st.form_submit_button("Predict Risk")

# Predict function
def predict(data, model):
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame(data, index=[0])
        
        # Do NOT apply one-hot encoding as model expects original categorical columns
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # More detailed error information
        st.error(f"Input data: {data}")
        if hasattr(model, 'feature_names_in_'):
            st.error(f"Model expects features: {model.feature_names_in_}")
            
            # Debug: Show what's missing
            missing_cols = set(model.feature_names_in_) - set(data.keys())
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                
            # Debug: Show what's extra
            extra_cols = set(data.keys()) - set(model.feature_names_in_)
            if extra_cols:
                st.error(f"Extra columns: {extra_cols}")
        return None, None

# Process prediction when form is submitted
if submit_button:
    if model:
        # Map form inputs to the format expected by the model
        input_data = {
            'Income': income,
            'Age': age,
            'Experience': experience,
            'Married/Single': marital_status,
            'House_Ownership': house_ownership,
            'Car_Ownership': car_ownership,
            'Profession': profession,
            'CITY': city,
            'STATE': state,
            'CURRENT_JOB_YRS': current_job_yrs,
            'CURRENT_HOUSE_YRS': current_house_yrs
        }
        
        # Get prediction
        prediction, prediction_proba = predict(input_data, model)
        
        if prediction is not None:
            # Display prediction with nice formatting
            st.write("## Prediction Result")
            
            if prediction == "High Risk":
                risk_class = "high-risk"
                risk_probability = prediction_proba[1] * 100
            else:
                risk_class = "low-risk"
                risk_probability = prediction_proba[0] * 100
            
            st.markdown(f"""
                <div class='prediction-box {risk_class}'>
                    <h3>Predicted Risk Level: {prediction}</h3>
                    <p>Confidence: {risk_probability:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display input data in a table for reference
            st.write("### Input Summary")
            input_df = pd.DataFrame(input_data, index=[0])
            st.dataframe(input_df)
            
            # Visualization of risk probability
            st.write("### Risk Probability")
            fig, ax = plt.subplots(figsize=(10, 2))
            colors = ["#66BB6A", "#EF5350"]
            sns.barplot(x=[prediction_proba[0]*100, prediction_proba[1]*100], 
                      y=["Low Risk", "High Risk"],
                      palette=colors,
                      ax=ax)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_ylabel("")
            st.pyplot(fig)
    else:
        st.error("Model could not be loaded. Please check the model file path.")

# Footer
st.markdown("<footer>Loan Risk Prediction System - © 2023</footer>", unsafe_allow_html=True)
