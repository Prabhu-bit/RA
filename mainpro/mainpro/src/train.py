import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import shap
import matplotlib.pyplot as plt
import sys
from sklearn.pipeline import Pipeline # Import Pipeline

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_NUMERIC_PATH = os.path.join(DATA_DIR, 'train_numeric.csv')
PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
MLP_MODEL_PATH = os.path.join(MODELS_DIR, 'mlp_model.joblib')
SHAP_SUMMARY_PLOT_PATH = os.path.join(MODELS_DIR, 'shap_summary_plot.png')

# Add src directory to Python path for importing preprocess module
sys.path.append(BASE_DIR)
from src.preprocess import build_preprocessing_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES, convert_to_numeric, convert_categorical, standardize_gender


def train_and_evaluate_models(X, y):
    """
    Trains and evaluates multiple models.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load preprocessor
    preprocessor = joblib.load(PREPROCESSING_PIPELINE_PATH)

    # Preprocess data and fit the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save the fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSING_PIPELINE_PATH)
    print(f"Fitted preprocessing pipeline saved to {PREPROCESSING_PIPELINE_PATH}")

    # Get feature names after preprocessing
    transformed_feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        
        # Handle pipelines within the ColumnTransformer
        if isinstance(transformer, Pipeline):
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                transformed_feature_names.extend(last_step.get_feature_names_out(features))
            else:
                # Fallback for steps that don't have get_feature_names_out
                transformed_feature_names.extend(features)
        elif hasattr(transformer, 'get_feature_names_out'):
            transformed_feature_names.extend(transformer.get_feature_names_out(features))
        else:
            # For FunctionTransformer or other simple transformers, use original feature names
            transformed_feature_names.extend(features)

    feature_names = transformed_feature_names
    X_processed_df = pd.DataFrame(X_test_processed, columns=feature_names) # Use X_test_processed here

    # --- Models ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', 
                                     n_estimators=200, learning_rate=0.1, max_depth=3),
        'MLP': MLPClassifier(random_state=42, max_iter=1000, alpha=0.01, hidden_layer_sizes=(100,), learning_rate_init=0.001)
    }

    best_model = None
    best_f1 = 0

    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        # Train
        model.fit(X_train_processed, y_train)

        # Predict
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC needs to be handled for multiclass
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        print(f"Accuracy (Train): {accuracy_score(y_train, model.predict(X_train_processed)):.4f}")
        print(f"F1 Score (Train): {f1_score(y_train, model.predict(X_train_processed), average='weighted'):.4f}")
        print(f"ROC AUC (Train): {roc_auc_score(y_train, model.predict_proba(X_train_processed), multi_class='ovr'):.4f}")
        print(f"Accuracy (Test): {accuracy:.4f}")
        print(f"F1 Score (Test): {f1:.4f}")
        print(f"ROC AUC (Test): {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            if name == 'Logistic Regression':
                joblib.dump(best_model, LOGISTIC_REGRESSION_MODEL_PATH)
            elif name == 'XGBoost':
                joblib.dump(best_model, XGBOOST_MODEL_PATH)
            else:
                joblib.dump(best_model, MLP_MODEL_PATH)

    print(f"\nBest model saved with F1 score: {best_f1:.4f}")

    # --- SHAP Explainability for the best model ---
    if best_model:
        print("\n--- Generating SHAP explanations for the best model ---")
        explainer = shap.TreeExplainer(best_model) # Use TreeExplainer for tree-based models like XGBoost
        shap_values = explainer.shap_values(X_test_processed)

        # If shap_values is a list (for multi-output models), take the absolute mean over classes
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            # For multi-class, shap_values is a list of arrays, one for each class.
            # We can plot the mean absolute SHAP value across all classes.
            # Or, for a specific class, e.g., class 1 (Seropositive RA):
            # shap.summary_plot(shap_values[1], X_test_processed, feature_names=feature_names, show=False)
            # For a general summary, we can average the absolute SHAP values across classes
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
            shap.summary_plot(mean_abs_shap_values, X_test_processed, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
        
        plt.savefig(SHAP_SUMMARY_PLOT_PATH, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {SHAP_SUMMARY_PLOT_PATH}")


def main():
    """
    Main function to run the training process.
    """
    # Load data
    df = pd.read_csv(TRAIN_NUMERIC_PATH)
    # Drop the 'Disease' column if it exists, as it's a label from the original seropositive data
    if 'Disease' in df.columns:
        df = df.drop('Disease', axis=1)
    X = df.drop('label', axis=1)
    y = df['label']

    # Train and evaluate
    train_and_evaluate_models(X, y)


if __name__ == '__main__':
    main()
