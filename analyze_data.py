import pandas as pd
import os

# File paths
BASE_DIR = 'f:/vs-code/RA/mainpro/mainpro'
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEROPOSITIVE_PATH = os.path.join(BASE_DIR, 'seropositive_data.csv')
SERONEGATIVE_PATH = os.path.join(BASE_DIR, 'seronegative_data.csv')

print("--- Analyzing seropositive_data.csv ---")
try:
    seropositive_df = pd.read_csv(SEROPOSITIVE_PATH)
    print("Columns:", seropositive_df.columns.tolist())
    print("Shape:", seropositive_df.shape)
    print("RA-Subtype values:", seropositive_df['RA-Subtype'].unique())
    print("Descriptive statistics for RF and Anti-CCP:")
    print(seropositive_df[['RF', 'Anti-CCP']].describe())
    print("\n")
except FileNotFoundError:
    print(f"File not found: {SEROPOSITIVE_PATH}")
    # Let's try the data directory
    SEROPOSITIVE_PATH = os.path.join(DATA_DIR, 'seropositive.csv')
    try:
        seropositive_df = pd.read_csv(SEROPOSITIVE_PATH)
        print("Columns:", seropositive_df.columns.tolist())
        print("Shape:", seropositive_df.shape)
        print("RA-Subtype values:", seropositive_df['RA-Subtype'].unique())
        print("Descriptive statistics for RF and Anti-CCP:")
        print(seropositive_df[['RF', 'Anti-CCP']].describe())
        print("\n")
    except FileNotFoundError:
        print(f"File not found: {SEROPOSITIVE_PATH}")


print("--- Analyzing seronegative_data.csv ---")
try:
    seronegative_df = pd.read_csv(SERONEGATIVE_PATH)
    print("Columns:", seronegative_df.columns.tolist())
    print("Shape:", seronegative_df.shape)
    print("RA-Subtype values:", seronegative_df['RA-Subtype'].unique())
    print("Descriptive statistics for RF and Anti-CCP:")
    print(seronegative_df[['RF', 'Anti-CCP']].describe())
except FileNotFoundError:
    print(f"File not found: {SERONEGATIVE_PATH}")
    # Let's try the data directory
    SERONEGATIVE_PATH = os.path.join(DATA_DIR, 'seronegative.csv')
    try:
        seronegative_df = pd.read_csv(SERONEGATIVE_PATH)
        print("Columns:", seronegative_df.columns.tolist())
        print("Shape:", seronegative_df.shape)
        print("RA-Subtype values:", seronegative_df['RA-Subtype'].unique())
        print("Descriptive statistics for RF and Anti-CCP:")
        print(seronegative_df[['RF', 'Anti-CCP']].describe())
    except FileNotFoundError:
        print(f"File not found: {SERONEGATIVE_PATH}")
