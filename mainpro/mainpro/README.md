# Rheumatoid Arthritis Subtype Classifier

This project implements a multimodal AI system for classifying Rheumatoid Arthritis (RA) subtypes and healthy controls based on clinical and laboratory data. The system provides an interactive Streamlit application for user input and real-time predictions, along with explainability features using SHAP.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Explainability (SHAP)](#explainability-shap)
- [Usage](#usage)
  - [Running the Streamlit Application](#running-the-streamlit-application)
  - [Input Methods](#input-methods)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Project Overview

Rheumatoid Arthritis (RA) is a chronic autoimmune disease that primarily affects joints. Accurate classification of RA subtypes (Seropositive RA, Seronegative RA) and differentiation from healthy individuals is crucial for personalized treatment and better patient outcomes. This project leverages machine learning models to achieve this classification, offering a user-friendly interface for clinicians and researchers.

## Features

- **Multiclass Classification:** Distinguishes between Healthy, Seropositive RA, and Seronegative RA.
- **Interactive Web Application:** Built with Streamlit for easy data input and prediction.
- **Multiple Input Methods:** Supports form-based input, CSV string input, and a placeholder for clinical report uploads.
- **Robust Preprocessing Pipeline:** Handles missing values, scales numerical features, and one-hot encodes categorical features.
- **Trained Machine Learning Models:** Utilizes Logistic Regression, XGBoost, and MLP classifiers.
- **Model Explainability:** Integrates SHAP (SHapley Additive exPlanations) to provide insights into model predictions.
- **Data Generation:** Includes a script to generate synthetic data for training and testing purposes.

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd mainpro
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate initial data (if not already present):**
    ```bash
    python3 src/generate_data.py
    ```
    This will create `seropositive.csv`, `test_data.csv`, `train_numeric.csv`, and `new_samples.csv` in the `data/` directory.

5.  **Build and save the preprocessing pipeline:**
    ```bash
    python3 src/preprocess.py
    ```
    This will save the `preprocessing_pipeline.joblib` in the `models/` directory.

6.  **Train the models:**
    ```bash
    python3 src/train.py
    ```
    This will train Logistic Regression, XGBoost, and MLP models, save the best performing model (`xgboost_model.joblib`), and generate SHAP explanation plots in the `models/` directory.

## Data

The project uses synthetic clinical and laboratory data. The main dataset (`seropositive.csv`) is used as a base for generating additional healthy and seronegative RA samples. The features include:

**Numeric Features:**
- `Age`
- `ESR` (Erythrocyte Sedimentation Rate)
- `CRP` (C-reactive protein)
- `RF` (Rheumatoid Factor)
- `Anti-CCP` (Anti-cyclic citrullinated peptide)
- `C3` (Complement 3)
- `C4` (Complement 4)

**Categorical Features:**
- `Gender` (Male/Female)
- `HLA-B27` (Human Leukocyte Antigen B27)
- `ANA` (Antinuclear Antibodies)
- `Anti-Ro` (Anti-Ro/SSA antibodies)
- `Anti-La` (Anti-La/SSB antibodies)
- `Anti-dsDNA` (Anti-double-stranded DNA antibodies)
- `Anti-Sm` (Anti-Smith antibodies)

## Preprocessing

The `src/preprocess.py` script defines and builds a robust preprocessing pipeline using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`. The pipeline includes:

-   **Numeric Features:** Imputation (median) and Scaling (StandardScaler).
-   **Categorical Features:** Standardization (custom `FunctionTransformer` for consistent values), Imputation (most frequent), and One-Hot Encoding.

The fitted pipeline is saved as `preprocessing_pipeline.joblib` and loaded by the Streamlit application for consistent data transformation.

## Models

The `src/train.py` script trains and evaluates three different machine learning models:

-   **Logistic Regression:** A linear model for classification.
-   **XGBoost (Extreme Gradient Boosting):** A powerful gradient boosting framework.
-   **MLP (Multilayer Perceptron):** A basic neural network classifier.

The models are evaluated based on accuracy, F1-score, and ROC AUC. The best performing model (typically XGBoost) is saved as `xgboost_model.joblib`.

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain the output of the best-trained model. The `src/train.py` script generates a SHAP summary plot, which is saved as `shap_summary_plot.png` in the `models/` directory. This plot helps understand the importance and impact of each feature on the model's predictions.

## Usage

### Running the Streamlit Application

To start the interactive web application, run the following command from the project root directory:

```bash
streamlit run app/app.py
```

This will open the application in your web browser, typically at `http://localhost:8501` (or another available port).

### Input Methods

The Streamlit application provides three ways to input patient data:

1.  **Form Input:** Fill out individual fields for each feature.
2.  **CSV String Input:** Enter comma-separated values directly into a text area. Ensure the order of values matches the expected feature order:
    `Age,Gender,ESR,CRP,RF,Anti-CCP,HLA-B27,ANA,Anti-Ro,Anti-La,Anti-dsDNA,Anti-Sm,C3,C4`
3.  **Upload Clinical Report (In Development):** A placeholder for future functionality to extract data from PDF/DOCX reports.

After entering data, click the "Classify" button to get predictions and confidence scores.

## Project Structure

```
mainpro/
├── app/
│   └── app.py             # Streamlit web application
├── data/
│   ├── seropositive.csv   # Base data for seropositive RA
│   ├── test_data.csv      # Test dataset
│   ├── train_numeric.csv  # Training dataset
│   ├── new_samples.csv    # Newly generated samples
│   └── backups/           # Directory for data backups
├── models/
│   ├── preprocessing_pipeline.joblib # Saved preprocessing pipeline
│   ├── xgboost_model.joblib          # Saved best trained model
│   ├── logistic_regression_model.joblib # Saved Logistic Regression model
│   ├── mlp_model.joblib              # Saved MLP model
│   └── shap_summary_plot.png         # SHAP explanation plot
├── src/
│   ├── generate_data.py   # Script to generate synthetic data
│   ├── preprocess.py      # Script to build and save the preprocessing pipeline
│   └── train.py           # Script to train and evaluate models
├── requirements.txt       # Python dependencies
└── run_app.sh             # Shell script to run the Streamlit app
```

## Future Enhancements

-   Implement robust text extraction and parsing from PDF/DOCX clinical reports.
-   Integrate more advanced explainability techniques.
-   Add user authentication and data storage.
-   Improve model monitoring and retraining mechanisms.
-   Expand to include more diverse datasets and RA subtypes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
