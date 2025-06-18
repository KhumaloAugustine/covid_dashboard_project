# train_covid_model.py
# This script loads, preprocesses, and trains a RandomForestRegressor model
# to predict 'New_deaths' from COVID-19 data. It then saves the model.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor # Using RandomForest for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score # Metrics for regression
import joblib # For saving/loading models
import numpy as np # For numerical operations like log transformation

print("Starting COVID-19 mortality prediction model training script...")

# --- 1. Load Data ---
try:
    # Load the dataset. The first column seems to be an unnamed index, so we drop it.
    covid_data = pd.read_csv('covid_vaccination_mortality.csv', index_col=0)
    print("Successfully loaded 'covid_vaccination_mortality.csv'.")
except FileNotFoundError:
    print("Error: 'covid_vaccination_mortality.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
print("Starting data preprocessing...")

# Convert 'date' column to datetime objects
covid_data['date'] = pd.to_datetime(covid_data['date'])

# Handle missing values: Fill NaNs in vaccination columns with 0, as absence of data
# likely means 0 vaccinations for that entry.
# 'ratio' might also have NaNs if population is 0 or vaccinations are 0.
numerical_cols_to_fill_zero = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'New_deaths', 'ratio'
]
for col in numerical_cols_to_fill_zero:
    # Use .loc to avoid SettingWithCopyWarning by operating on the original DataFrame
    covid_data.loc[:, col] = covid_data[col].fillna(0)

# Drop rows where 'population' is missing or zero as it's crucial for ratios/per capita
covid_data.dropna(subset=['population'], inplace=True)
covid_data = covid_data[covid_data['population'] > 0].copy() # Use .copy() after filtering

# Feature Engineering (creating potentially useful features)
# Ensure columns exist before operations.
covid_data.loc[:, 'vaccination_coverage'] = covid_data['people_fully_vaccinated'] / covid_data['population']
covid_data.loc[:, 'vaccination_coverage'] = covid_data['vaccination_coverage'].fillna(0) # Handle division by zero or NaNs

# Create a numerical representation for 'date' (e.g., days since first date)
# This helps the model understand temporal progression.
min_date = covid_data['date'].min()
covid_data.loc[:, 'days_since_start'] = (covid_data['date'] - min_date).dt.days

print("Data preprocessing complete. Features engineered.")

# --- 3. Feature Selection for Model ---
# Define features (X) and target (y) for the regression model.
# We are predicting 'New_deaths'.
features = [
    'total_vaccinations',
    'people_vaccinated',
    'people_fully_vaccinated',
    'population',
    'ratio',
    'vaccination_coverage',
    'days_since_start'
]

# Ensure all selected features exist and convert them to numeric type, handling errors
# and filling any remaining NaNs in features with 0.
for col in features:
    if col in covid_data.columns:
        covid_data.loc[:, col] = pd.to_numeric(covid_data[col], errors='coerce').fillna(0)
    else:
        print(f"Warning: Feature '{col}' not found in data. Skipping.")
        features.remove(col) # Remove missing feature from list

X = covid_data[features]
y = covid_data['New_deaths']

# Important: Handle potential target variable outliers or skewness
# New_deaths can be zero or highly skewed. Log transformation is common for skewed targets.
# Add a small constant (e.g., 1) to avoid log(0), so log1p(x) = log(1+x).
# Ensure 'New_deaths' are non-negative before transformation.
y_clean = y[y >= 0] # Filter out any unexpected negative values in New_deaths
y_transformed = np.log1p(y_clean)

# --- CRITICAL FIX: Drop rows where y_transformed is NaN or infinity ---
# This ensures that X and y_transformed have no missing values that would
# cause issues during model training.
valid_indices = y_transformed.dropna().index # Get indices where y_transformed is not NaN
X_filtered = X.loc[valid_indices]
y_filtered_transformed = y_transformed.loc[valid_indices]

# Also ensure no infinite values after transformation
if np.isinf(y_filtered_transformed).any():
    inf_indices = np.isinf(y_filtered_transformed)
    X_filtered = X_filtered.loc[~inf_indices]
    y_filtered_transformed = y_filtered_transformed.loc[~inf_indices]
    print(f"Removed {inf_indices.sum()} infinite values from target.")

if X_filtered.empty or y_filtered_transformed.empty:
    print("Error: No valid data remaining after filtering NaNs/Infs in target. Cannot train model.")
    exit()

print(f"Target variable cleaned. Remaining samples: {len(y_filtered_transformed)}")

# --- 4. Data Splitting ---
# Split the data into training and testing sets using the filtered data.
X_train, X_test, y_train_transformed, y_test_transformed = train_test_split(
    X_filtered, y_filtered_transformed, test_size=0.2, random_state=42
)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# --- 5. Train the Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("RandomForestRegressor model initialized.")

model.fit(X_train, y_train_transformed)
print("Model training complete.")

# --- 6. Evaluate the Model (Optional but good practice) ---
predictions_transformed = model.predict(X_test)
# Inverse transform the predictions to get them back to the original scale
predictions = np.expm1(predictions_transformed) # expm1(x) = exp(x) - 1

# Ensure predictions are non-negative
predictions[predictions < 0] = 0

# For evaluation metrics, use the original scale of y_test, which corresponds
# to the indices of y_filtered_transformed before its transformation.
# Need to get the original 'New_deaths' values for these test indices.
y_test_original_scale = covid_data.loc[y_test_transformed.index, 'New_deaths']

mse = mean_squared_error(y_test_original_scale, predictions)
r2 = r2_score(y_test_original_scale, predictions)

print(f"Model Mean Squared Error (MSE): {mse:.2f}")
print(f"Model R-squared (R2): {r2:.2f}")

# --- 7. Save the Trained Model and Feature Names ---
joblib.dump(model, 'trained_covid_model.pkl')
print("Trained model saved as 'trained_covid_model.pkl'.")

joblib.dump(features, 'model_features.pkl')
print("Model features saved as 'model_features.pkl'.")

print("COVID-19 model training and saving process finished.")
