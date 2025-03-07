import pickle
import pandas as pd
import joblib

class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler to handle potential namespace issues with pickled files.
    """
    def find_class(self, module, name):
        # Redirect module if it is the old numpy internal module
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)

def load_data(file_path='data/processed/df_loan.pkl'):
    """
    Load the loan dataset with custom unpickler for compatibility
    """
    try:
        with open(file_path, 'rb') as f:
            df = CustomUnpickler(f).load()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_model(model_path='models/best_risk_classifier.pkl'):
    """
    Load the pre-trained loan risk prediction model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_input(input_data, model_columns=None):
    """
    Preprocess input data to match the format expected by the model
    
    Args:
        input_data: Dictionary of user inputs
        model_columns: List of columns expected by the model
        
    Returns:
        Processed DataFrame ready for prediction
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Process categorical features with one-hot encoding
    categorical_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 
                         'Profession', 'CITY', 'STATE']
    
    # One-hot encode categorical variables
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Align with model columns if provided
    if model_columns is not None:
        for col in model_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
        
        # Keep only the columns expected by the model
        input_df_encoded = input_df_encoded[model_columns]
    
    return input_df_encoded

def get_model_features(model):
    """
    Extract feature names from a trained model
    """
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    
    # If the model doesn't have feature_names_in_ attribute, try to extract from steps
    if hasattr(model, 'steps'):
        for name, step in reversed(model.steps):  # Check from last step
            if hasattr(step, 'feature_names_in_'):
                return step.feature_names_in_
    
    return None

def check_input_compatibility(input_data, model):
    """
    Check if input data is compatible with model's expected features
    
    Args:
        input_data: Dictionary of user inputs
        model: Trained model
        
    Returns:
        bool: True if compatible, False otherwise
        dict: Dictionary with missing and extra columns
    """
    expected_features = get_model_features(model)
    
    if expected_features is None:
        return False, {"error": "Could not extract feature names from model"}
    
    input_features = set(input_data.keys())
    expected_features_set = set(expected_features)
    
    missing_cols = expected_features_set - input_features
    extra_cols = input_features - expected_features_set
    
    is_compatible = len(missing_cols) == 0
    
    return is_compatible, {
        "missing": missing_cols,
        "extra": extra_cols,
        "expected": expected_features
    }

def format_input_for_model(input_data, model):
    """
    Format input data to match exactly what the model expects
    
    Args:
        input_data: Dictionary of user inputs
        model: Trained model
        
    Returns:
        pd.DataFrame: DataFrame with columns in the order expected by model
    """
    input_df = pd.DataFrame(input_data, index=[0])
    
    expected_features = get_model_features(model)
    if expected_features is not None:
        # Add missing columns with default values
        for col in expected_features:
            if col not in input_df.columns:
                if pd.api.types.is_numeric_dtype(input_df.dtypes.iloc[0]):
                    input_df[col] = 0
                else:
                    input_df[col] = "Unknown"
        
        # Reorder columns to match model's expectation
        input_df = input_df[expected_features]
    
    return input_df
