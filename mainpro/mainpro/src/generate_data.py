import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
SEED = 42
np.random.seed(SEED)

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKUP_DIR = os.path.join(DATA_DIR, 'backups')
SEROPOSITIVE_PATH = os.path.join(DATA_DIR, 'seropositive.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
TRAIN_NUMERIC_PATH = os.path.join(DATA_DIR, 'train_numeric.csv')

# Ensure backup directory exists
os.makedirs(BACKUP_DIR, exist_ok=True)

def generate_synthetic_data(seropositive_df, num_healthy, num_seronegative):
    """
    Generates synthetic data for healthy and seronegative RA patients.
    """
    # --- Generate Healthy Data ---
    healthy_data = {
        'Age': np.random.choice(seropositive_df['Age'], num_healthy),
        'Gender': np.random.choice(seropositive_df['Gender'], num_healthy),
        'ESR': np.random.normal(10, 7, num_healthy).clip(0),
        'CRP': np.random.normal(2, 2, num_healthy).clip(0),
        'RF': np.random.normal(8, 6, num_healthy).clip(0),
        'Anti-CCP': np.random.normal(4, 3, num_healthy).clip(0),
        'HLA-B27': np.random.choice([0, 1], num_healthy, p=[0.9, 0.1]),
        'ANA': np.random.choice([0, 1], num_healthy, p=[0.95, 0.05]),
        'Anti-Ro': np.random.choice([0, 1], num_healthy, p=[0.95, 0.05]),
        'Anti-La': np.random.choice([0, 1], num_healthy, p=[0.95, 0.05]),
        'Anti-dsDNA': np.random.choice([0, 1], num_healthy, p=[0.95, 0.05]),
        'Anti-Sm': np.random.choice([0, 1], num_healthy, p=[0.95, 0.05]),
        'C3': np.random.normal(1.1, 0.2, num_healthy).clip(0),
        'C4': np.random.normal(0.25, 0.05, num_healthy).clip(0),
    }
    healthy_df = pd.DataFrame(healthy_data)

    # --- Generate Seronegative RA Data ---
    seronegative_data = {
        'Age': np.random.choice(seropositive_df['Age'], num_seronegative),
        'Gender': np.random.choice(seropositive_df['Gender'], num_seronegative),
        'ESR': np.random.normal(30, 20, num_seronegative).clip(0),
        'CRP': np.random.normal(10, 6, num_seronegative).clip(0),
        'RF': np.random.normal(8, 6, num_seronegative).clip(0, 19), # Force negative
        'Anti-CCP': np.random.normal(4, 3, num_seronegative).clip(0, 19), # Force negative
        'HLA-B27': np.random.choice([0, 1], num_seronegative, p=[0.85, 0.15]),
        'ANA': np.random.choice([0, 1], num_seronegative, p=[0.8, 0.2]),
        'Anti-Ro': np.random.choice([0, 1], num_seronegative, p=[0.9, 0.1]),
        'Anti-La': np.random.choice([0, 1], num_seronegative, p=[0.9, 0.1]),
        'Anti-dsDNA': np.random.choice([0, 1], num_seronegative, p=[0.9, 0.1]),
        'Anti-Sm': np.random.choice([0, 1], num_seronegative, p=[0.9, 0.1]),
        'C3': np.random.normal(1.0, 0.2, num_seronegative).clip(0),
        'C4': np.random.normal(0.2, 0.05, num_seronegative).clip(0),
    }
    seronegative_df = pd.DataFrame(seronegative_data)

    return healthy_df, seronegative_df

def map_categorical_to_strings(df):
    """
    Maps 0/1 numerical categorical features and np.nan to 'Negative'/'Positive'/'Missing' strings.
    """
    df_copy = df.copy()
    categorical_cols = ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']
    for col in categorical_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map({0: 'Negative', 1: 'Positive'}).fillna('Missing')
    return df_copy

def main():
    """
    Main function to run the data generation process.
    """
    # Load seropositive data
    seropositive_df = pd.read_csv(SEROPOSITIVE_PATH)
    # Convert 'Positive'/'Negative' to 1/0 in seropositive_df
    for col in ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']:
        seropositive_df[col] = seropositive_df[col].map({'Positive': 1, 'Negative': 0}).fillna(np.nan)

    # Generate synthetic data
    num_healthy_to_generate = len(seropositive_df)
    num_seronegative_to_generate = len(seropositive_df)
    healthy_df, seronegative_df = generate_synthetic_data(seropositive_df, num_healthy_to_generate, num_seronegative_to_generate)

    # Add labels
    seropositive_df['label'] = 1
    healthy_df['label'] = 0
    seronegative_df['label'] = 2

    # Ensure correct data types for training and test data
    for col in ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']:
        healthy_df[col] = healthy_df[col].astype(int)
        seronegative_df[col] = seronegative_df[col].astype(int)
        seropositive_df[col] = seropositive_df[col].astype(int)
    
    # Gender should remain object/string type for one-hot encoding
    healthy_df['Gender'] = healthy_df['Gender'].astype(str)
    seronegative_df['Gender'] = seronegative_df['Gender'].astype(str)
    seropositive_df['Gender'] = seropositive_df['Gender'].astype(str)

    # Backup generated data (with string representation for categorical features)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    map_categorical_to_strings(healthy_df).to_csv(os.path.join(BACKUP_DIR, f'healthy_{timestamp}.csv'), index=False)
    map_categorical_to_strings(seronegative_df).to_csv(os.path.join(BACKUP_DIR, f'seronegative_{timestamp}.csv'), index=False)

    # Prepare test data (numerical for model consumption)
    test_healthy = healthy_df.sample(100)
    test_seropositive = seropositive_df.sample(100)
    test_seronegative = seronegative_df.sample(100)
    test_data = pd.concat([test_healthy, test_seropositive, test_seronegative])
    
    # Save test data (with string representation for categorical features)
    try:
        existing_test_df = pd.read_csv(TEST_DATA_PATH)
        # Convert existing test_df categorical columns to strings for consistent concatenation
        existing_test_df = map_categorical_to_strings(existing_test_df)
        final_test_df = pd.concat([existing_test_df, map_categorical_to_strings(test_data)])
    except FileNotFoundError:
        final_test_df = map_categorical_to_strings(test_data)
    final_test_df.to_csv(TEST_DATA_PATH, index=False)


    # Prepare training data (numerical for model consumption)
    train_healthy = healthy_df.drop(test_healthy.index)
    train_seropositive = seropositive_df.drop(test_seropositive.index)
    train_seronegative = seronegative_df.drop(test_seronegative.index)
    train_data = pd.concat([train_healthy, train_seropositive, train_seronegative])

    # Save training data
    train_data.to_csv(TRAIN_NUMERIC_PATH, index=False)

    # Generate 100 new samples, roughly equally distributed among the three subtypes
    num_samples_per_subtype = 33 # For a total of 99 samples, close to 100

    # Generate synthetic data for healthy and seronegative
    healthy_df_new, seronegative_df_new = generate_synthetic_data(seropositive_df, num_samples_per_subtype, num_samples_per_subtype)

    # Sample from existing seropositive data
    seropositive_df_new = seropositive_df.sample(n=num_samples_per_subtype, random_state=SEED)

    # Add labels to the new data
    healthy_df_new['label'] = 0
    seropositive_df_new['label'] = 1
    seronegative_df_new['label'] = 2

    # Combine all new samples
    new_samples_df = pd.concat([healthy_df_new, seropositive_df_new, seronegative_df_new], ignore_index=True)

    # Map categorical features back to strings for the new samples for better readability
    new_samples_df = map_categorical_to_strings(new_samples_df)

    # Save the new samples to a CSV file
    NEW_SAMPLES_PATH = os.path.join(DATA_DIR, 'new_samples.csv')
    new_samples_df.to_csv(NEW_SAMPLES_PATH, index=False)
    print(f"Generated 100 new samples (33 per subtype) and saved to {NEW_SAMPLES_PATH}")

    print("Data generation and preparation complete.")
    print(f"Training data saved to {TRAIN_NUMERIC_PATH}")
    print(f"Test data updated in {TEST_DATA_PATH}")

if __name__ == '__main__':
    main()
