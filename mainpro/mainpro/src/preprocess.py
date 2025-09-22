import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_NUMERIC_PATH = os.path.join(DATA_DIR, 'train_numeric.csv')
PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')

# Feature definitions
NUMERIC_FEATURES = ['Age', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'C3', 'C4']
CATEGORICAL_FEATURES = ['Gender', 'HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']
ALL_FEATURES = ['Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']

def convert_to_numeric(X):
    """Convert string values to numeric, handling various formats."""
    def _convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # Remove any units or special characters
            x = x.strip().lower()
            try:
                return float(''.join(c for c in x if c.isdigit() or c in '.-'))
            except ValueError:
                return np.nan
        return np.nan
    
    return pd.DataFrame(X).applymap(_convert)

def convert_categorical(X):
    """Convert categorical values to standardized format."""
    def _standardize_categorical(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return 1 if x > 0 else 0
        if isinstance(x, str):
            x = x.strip().lower()
            if x in ['positive', 'pos', '1', 'yes', 'true']:
                return 1
            elif x in ['negative', 'neg', '0', 'no', 'false']:
                return 0
            else:
                return np.nan
        return np.nan
    
    return pd.DataFrame(X).applymap(_standardize_categorical)

def standardize_gender(X):
    """Standardize gender values."""
    def _standardize_gender(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in ['m', 'male', '1']:
            return 'male'
        elif x in ['f', 'female', '0']:
            return 'female'
        else:
            return np.nan
    
    return pd.DataFrame(X).applymap(_standardize_gender)

def build_preprocessing_pipeline():
    """
    Builds and saves the preprocessing pipeline with robust data type handling.
    """
    # Create numeric preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('convert', FunctionTransformer(convert_to_numeric)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create gender preprocessing pipeline
    gender_transformer = Pipeline(steps=[
        ('standardize', FunctionTransformer(standardize_gender)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create binary categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('convert', FunctionTransformer(convert_categorical)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create the full preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('gender', gender_transformer, ['Gender']),
            ('cat', categorical_transformer, [col for col in CATEGORICAL_FEATURES if col != 'Gender'])
        ],
        remainder='drop'
    )

    return preprocessor

def validate_input_data(df):
    """
    Validate input data and provide helpful error messages.
    """
    errors = []
    
    # Check if all required columns are present
    missing_cols = set(ALL_FEATURES) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for obviously invalid values in numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            invalid_mask = df[col].apply(lambda x: 
                not (pd.isna(x) or 
                     (isinstance(x, (int, float)) and not np.isinf(x)) or
                     (isinstance(x, str) and any(c.isdigit() for c in x))))
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, col].unique()
                errors.append(f"Invalid values in {col}: {invalid_values}")
    
    # Check categorical values
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if col == 'Gender':
                valid_values = {'m', 'male', 'f', 'female', '0', '1'}
                invalid_mask = df[col].apply(lambda x: 
                    not pd.isna(x) and str(x).strip().lower() not in valid_values)
            else:
                valid_values = {'positive', 'pos', 'negative', 'neg', 'yes', 'no', 'true', 'false', '0', '1', 0, 1}
                invalid_mask = df[col].apply(lambda x: 
                    not pd.isna(x) and str(x).strip().lower() not in valid_values and x not in {0, 1})
            
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, col].unique()
                errors.append(f"Invalid values in {col}: {invalid_values}")
    
    return errors

def main():
    """
    Main function to run the preprocessing.
    """
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Build the pipeline
    pipeline = build_preprocessing_pipeline()
    
    # Save the pipeline
    joblib.dump(pipeline, PREPROCESSING_PIPELINE_PATH)
    print(f"Preprocessing pipeline saved to {PREPROCESSING_PIPELINE_PATH}")

if __name__ == '__main__':
    main()
