"""
    This script trains two Random Forest Regressor models:
    one to predict 'New_deaths' and another to predict 'daily_vaccinations'.
    It saves the trained models and their respective feature lists using joblib. 
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

print("Starting model training script...")

# --- 1. Load Data ---
try:
    data = pd.read_csv('covid_vaccination_mortality.csv', index_col=0)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'covid_vaccination_mortality.csv' not found. Please ensure it's in the same directory.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
data['date'] = pd.to_datetime(data['date'])

# Fill NaNs with 0 for relevant numerical columns
numerical_cols_to_fill_zero = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'New_deaths', 'ratio'
]
for col in numerical_cols_to_fill_zero:
    data[col] = data[col].fillna(0)

# Ensure population is numeric and handle potential zeros/Na
data['population'] = pd.to_numeric(data['population'], errors='coerce')
data.dropna(subset=['population'], inplace=True)
data = data[data['population'] > 0] # Filter out zero population entries

# Feature Engineering
data['vaccination_coverage'] = data['people_fully_vaccinated'] / data['population']
data['vaccination_coverage'].fillna(0, inplace=True) # Handle division by zero or NaNs

min_date = data['date'].min()
data['days_since_start'] = (data['date'] - min_date).dt.days

# Calculate daily vaccinations (needs to be grouped by country first)
# Sort data to ensure correct difference calculation for daily_vaccinations
data = data.sort_values(by=['country', 'date'])
data['daily_vaccinations'] = data.groupby('country')['total_vaccinations'].diff().fillna(0)
# Ensure daily_vaccinations are non-negative (can happen with data corrections/resets)
data['daily_vaccinations'] = data['daily_vaccinations'].apply(lambda x: max(0, x))

# --- Define Target Variables and Features for Each Model ---

# Model 1: New Deaths Prediction
target_deaths = 'New_deaths'
features_deaths = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'population', 'ratio', 'vaccination_coverage', 'days_since_start'
]

# Model 2: Daily Vaccinations Prediction
target_vaccinations = 'daily_vaccinations'
features_vaccinations = [
    'people_vaccinated', 'people_fully_vaccinated', 'population',
    'vaccination_coverage', 'days_since_start', 'New_deaths' # New_deaths included as a feature here
]


# --- Training Function ---
def train_and_save_model(data_df, target_col, features_list, model_name_prefix):
    """
    Trains a Random Forest Regressor model for a given target and features,
    and saves the model and feature list.
    """
    print(f"\n--- Training model for {target_col} ---")

    # Prepare features (X) and target (y)
    y = data_df[target_col]
    
    # Apply log1p transformation to the target variable to handle skewed data,
    # ensuring targets are non-negative for log transformation.
    y_clean = y[y >= 0] 
    y_transformed = np.log1p(y_clean)

    # Align X and y after filtering and transformation
    valid_indices = y_transformed.dropna().index
    X = data_df.loc[valid_indices, features_list]
    y_filtered_transformed = y_transformed.loc[valid_indices]

    # Handle potential inf values after log1p for robustness (shouldn't happen with >=0 filter)
    if np.isinf(y_filtered_transformed).any():
        inf_indices = np.isinf(y_filtered_transformed)
        X = X.loc[~inf_indices]
        y_filtered_transformed = y_filtered_transformed.loc[~inf_indices]

    # Drop rows with any NaN in features after selection
    initial_rows = len(X)
    X.dropna(inplace=True)
    y_filtered_transformed = y_filtered_transformed.loc[X.index]
    if len(X) < initial_rows:
        print(f"Dropped {initial_rows - len(X)} rows due to NaN values in features for {target_col}.")

    if X.empty:
        print(f"No valid data to train model for {target_col}. Skipping.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_filtered_transformed, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions_transformed = model.predict(X_test)
    predictions = np.expm1(predictions_transformed) # Inverse transform
    predictions[predictions < 0] = 0 # Ensure non-negative predictions

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
train_and_save_model(data, target_deaths, features_deaths, 'deaths')
train_and_save_model(data, target_vaccinations, features_vaccinations, 'vaccinations')

print("\nAll models trained and saved.")
