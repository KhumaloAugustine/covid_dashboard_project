"""
    This script trains two Random Forest Regressor models:
    one to predict 'New_deaths' and another to predict 'daily_vaccinations'.
    It saves the trained models and their respective feature lists using joblib.
    
    Uses centralized config and data_preprocessing modules to follow DRY principles.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Import from centralized modules
from config import DATA_FILE, FEATURES_DEATHS, FEATURES_VACCINATIONS
from data_preprocessing import preprocess_covid_data, prepare_model_data

print("Starting model training script...")

# --- 1. Load and Preprocess Data ---
try:
    data = pd.read_csv(DATA_FILE, index_col=0)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Please ensure it's in the same directory.")
    exit()

# Use shared preprocessing function
data = preprocess_covid_data(data)
print("Data preprocessing completed.")

# --- Define Target Variables ---
TARGET_DEATHS = 'New_deaths'
TARGET_VACCINATIONS = 'daily_vaccinations'


# --- Training Function ---
def train_and_save_model(data_df, target_col, features_list, model_name_prefix):
    """
    Trains a Random Forest Regressor model for a given target and features,
    and saves the model and feature list.
    
    Uses prepare_model_data from data_preprocessing for data preparation.
    """
    print(f"\n--- Training model for {target_col} ---")

    # Use shared data preparation function
    X, y_filtered_transformed, valid_indices = prepare_model_data(data_df, target_col, features_list)
    
    if X is None:
        print(f"No valid data to train model for {target_col}. Skipping.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_filtered_transformed, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions_transformed = model.predict(X_test)
    predictions = np.expm1(predictions_transformed)  # Inverse transform
    predictions[predictions < 0] = 0  # Ensure non-negative predictions

    # Get original scale target values for evaluation
    y_test_original_scale = data_df.loc[y_test.index, target_col]

    mse = mean_squared_error(y_test_original_scale, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original_scale, predictions)

    print(f"Model Performance for {target_col}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")

    # Save the trained model and its features
    model_filepath = f'trained_{model_name_prefix}_model.pkl'
    features_filepath = f'model_features_{model_name_prefix}.pkl'
    joblib.dump(model, model_filepath)
    joblib.dump(features_list, features_filepath)
    print(f"Model saved to {model_filepath}")
    print(f"Features saved to {features_filepath}")
    print(f"--- Finished training for {target_col} ---")


# --- Execute Training for Both Models ---
train_and_save_model(data, TARGET_DEATHS, FEATURES_DEATHS, 'deaths')
train_and_save_model(data, TARGET_VACCINATIONS, FEATURES_VACCINATIONS, 'vaccinations')

print("\nAll models trained and saved.")
