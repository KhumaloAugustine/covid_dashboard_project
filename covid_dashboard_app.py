# covid_dashboard_app.py
# This script creates an interactive Streamlit dashboard
# for analyzing COVID-19 vaccination and mortality data,
# and predicting new deaths using a trained regression model.

import streamlit as st
import pandas as pd
import joblib # Used for loading the pre-trained machine learning model
import matplotlib.pyplot as plt # For creating static plots
import seaborn as sns # For enhanced data visualizations
import numpy as np # For numerical operations (e.g., log transform inverse)
import datetime # Import the datetime module

# --- 1. Dashboard Configuration and Title ---
st.set_page_config(
    page_title="COVID-19 Vaccination & Mortality Dashboard",
    layout="wide", # Using wide layout for more space
    initial_sidebar_state="auto"
)

st.title("🦠 COVID-19 Vaccination & Mortality Dashboard")
st.markdown("---")

# --- 2. Load Data and Pre-trained Model ---
@st.cache_data
def load_data():
    """Loads the COVID-19 dataset and performs initial preprocessing."""
    try:
        data = pd.read_csv('covid_vaccination_mortality.csv', index_col=0)
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

        # Feature Engineering (same as in training script)
        data['vaccination_coverage'] = data['people_fully_vaccinated'] / data['population']
        data['vaccination_coverage'].fillna(0, inplace=True) # Handle division by zero or NaNs

        min_date = data['date'].min()
        data['days_since_start'] = (data['date'] - min_date).dt.days

        return data
    except FileNotFoundError:
        st.error("Error: 'covid_vaccination_mortality.csv' not found. Please ensure it's in the same directory.")
        st.stop()

covid_data = load_data()

# Load the trained model and features used for training
try:
    model = joblib.load('trained_covid_model.pkl')
    model_features = joblib.load('model_features.pkl')
    st.sidebar.success("Model and data loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'trained_covid_model.pkl' or 'model_features.pkl' not found. Please run 'train_covid_model.py' first.")
    st.stop()

# --- 3. Dashboard Introduction ---
st.write(
    """
    This dashboard provides insights into COVID-19 vaccination efforts and their correlation with mortality
    across different countries and over time. It also features a predictive model to estimate new deaths.
    """
)
st.markdown("---")

# --- 4. Interactive Data Filtering (Sidebar) ---
st.sidebar.header("📊 Data Filters")

# Country selection
all_countries = covid_data['country'].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select Country(ies):",
    options=all_countries,
    default=['Afghanistan'] # Default to a few countries for demonstration
)

# Date Range Slider
min_date_data = covid_data['date'].min().to_pydatetime()
max_date_data = covid_data['date'].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date_data,
    max_value=max_date_data,
    value=(min_date_data, max_date_data),
    format="YYYY-MM-DD"
)

# Apply filters
filtered_data = covid_data[
    covid_data['country'].isin(selected_countries) &
    (covid_data['date'] >= date_range[0]) &
    (covid_data['date'] <= date_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
    st.stop() # Stop execution if no data to display

# --- 5. Descriptive Analysis (EDA) - Main Section ---
st.header("📈 COVID-19 Data Trends & Insights")
st.write(f"Displaying data for **{', '.join(selected_countries)}** from **{date_range[0].strftime('%Y-%m-%d')}** to **{date_range[1].strftime('%Y-%m-%d')}**.")

# 5.1 Time Series Plots
st.subheader("5.1 Time Series Trends")
col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    st.write("#### New Deaths Over Time")
    fig_deaths_ts, ax_deaths_ts = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=filtered_data, x='date', y='New_deaths', hue='country', marker='o', ax=ax_deaths_ts)
    ax_deaths_ts.set_title('New Deaths Over Time')
    ax_deaths_ts.set_xlabel('Date')
    ax_deaths_ts.set_ylabel('New Deaths')
    ax_deaths_ts.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_deaths_ts)

with col_ts2:
    st.write("#### Total Vaccinations Over Time")
    fig_vacc_ts, ax_vacc_ts = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=filtered_data, x='date', y='total_vaccinations', hue='country', marker='o', ax=ax_vacc_ts)
    ax_vacc_ts.set_title('Total Vaccinations Over Time')
    ax_vacc_ts.set_xlabel('Date')
    ax_vacc_ts.set_ylabel('Total Vaccinations')
    ax_vacc_ts.tick_params(axis='x', rotation=45)
    ax_vacc_ts.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
    plt.tight_layout()
    st.pyplot(fig_vacc_ts)

st.markdown("---")

# 5.2 Distributions & Relationships
st.subheader("5.2 Key Distributions and Relationships")

col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    st.write("#### Distribution of New Deaths")
    fig_hist_deaths, ax_hist_deaths = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_data['New_deaths'], kde=True, bins=min(30, len(filtered_data)), color='salmon', ax=ax_hist_deaths)
    ax_hist_deaths.set_title('Distribution of New Deaths')
    ax_hist_deaths.set_xlabel('New Deaths')
    ax_hist_deaths.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig_hist_deaths)

with col_dist2:
    st.write("#### Distribution of Vaccination Coverage (People Fully Vaccinated / Population)")
    fig_hist_coverage, ax_hist_coverage = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_data['vaccination_coverage'], kde=True, bins=min(30, len(filtered_data)), color='lightgreen', ax=ax_hist_coverage)
    ax_hist_coverage.set_title('Distribution of Vaccination Coverage')
    ax_hist_coverage.set_xlabel('Vaccination Coverage')
    ax_hist_coverage.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig_hist_coverage)

st.write("#### Correlation Heatmap of Numerical Features")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
# Select only relevant numerical columns for correlation
numerical_data_for_corr = filtered_data[
    ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
     'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start']
].corr()
sns.heatmap(numerical_data_for_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title('Correlation Matrix of Key Metrics')
plt.tight_layout()
st.pyplot(fig_corr)

st.markdown("---")

# --- 6. Predictive Analysis Section ---
st.header("🔮 Predict New COVID-19 Deaths")
st.write("Enter values for key metrics to predict the number of New Deaths.")

# Create input fields for the model features
input_col1, input_col2, input_col3 = st.columns(3)

# Initialize input values with means from the filtered data, or sensible defaults
# Using .mean() from filtered_data makes inputs more relevant to current view
# Added checks for empty filtered_data for default values.
default_total_vacc = filtered_data['total_vaccinations'].mean() if not filtered_data.empty else 0
default_people_vacc = filtered_data['people_vaccinated'].mean() if not filtered_data.empty else 0
default_fully_vacc = filtered_data['people_fully_vaccinated'].mean() if not filtered_data.empty else 0
default_population = filtered_data['population'].mean() if not filtered_data.empty else 1000000 # Avoid 0 pop
default_ratio = filtered_data['ratio'].mean() if not filtered_data.empty else 0
default_vacc_coverage = filtered_data['vaccination_coverage'].mean() if not filtered_data.empty else 0.1
# For days_since_start, use max_date_data + 7 days as a default future prediction point
default_days_since_start_date = max_date_data + pd.Timedelta(days=7) # This is a datetime.datetime object

with input_col1:
    total_vaccinations_pred = st.number_input("Total Vaccinations:",
                                              min_value=0.0,
                                              value=float(default_total_vacc), # Ensure float
                                              step=10000.0,
                                              format="%.0f")
    people_vaccinated_pred = st.number_input("People Vaccinated (at least one dose):",
                                             min_value=0.0,
                                             value=float(default_people_vacc), # Ensure float
                                             step=10000.0,
                                             format="%.0f")
    people_fully_vaccinated_pred = st.number_input("People Fully Vaccinated:",
                                                    min_value=0.0,
                                                    value=float(default_fully_vacc), # Ensure float
                                                    step=10000.0,
                                                    format="%.0f")
with input_col2:
    population_pred = st.number_input("Population:",
                                      min_value=1.0, # Population must be at least 1
                                      value=float(default_population), # Ensure float
                                      step=100000.0,
                                      format="%.0f")
    ratio_pred = st.number_input("Ratio:",
                                 min_value=0.0,
                                 value=float(default_ratio), # Ensure float
                                 step=0.01,
                                 format="%.4f")
    vaccination_coverage_pred = st.number_input("Vaccination Coverage (Fully Vaccinated / Population):",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=float(default_vacc_coverage), # Ensure float
                                                step=0.01,
                                                format="%.2f")
with input_col3:
    # A date picker for 'days_since_start'
    prediction_date_input = st.date_input("Select Prediction Date:",
                                    value=default_days_since_start_date.date(), # st.date_input expects datetime.date
                                    min_value=covid_data['date'].min().date()) # Cannot predict before historical start
    
    # Convert prediction_date_input (datetime.date) to datetime.datetime for subtraction
    prediction_datetime = datetime.datetime.combine(prediction_date_input, datetime.time.min)

    # Calculate days_since_start from the *earliest* date in the *entire* dataset
    # to ensure consistency with how it was calculated during model training.
    days_since_start_pred = (prediction_datetime - covid_data['date'].min().to_pydatetime()).days
    st.info(f"Calculated days since start: {days_since_start_pred}")


# Prepare input for prediction, ensuring column order matches training
input_for_prediction = pd.DataFrame([[
    total_vaccinations_pred,
    people_vaccinated_pred,
    people_fully_vaccinated_pred,
    population_pred,
    ratio_pred,
    vaccination_coverage_pred,
    days_since_start_pred
]], columns=model_features)

if st.button("Predict New Deaths"):
    try:
        # Predict on the transformed scale (log1p)
        predicted_deaths_transformed = model.predict(input_for_prediction)[0]
        # Inverse transform to get actual death count
        predicted_deaths = np.expm1(predicted_deaths_transformed)
        # Ensure prediction is not negative
        predicted_deaths = max(0, predicted_deaths)

        st.success(f"**Predicted New Deaths:** {int(round(predicted_deaths))} ")
        st.caption("*(Prediction is an estimate based on the model's training data. Results may vary.)*")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid numbers and model_features.pkl is correctly loaded.")

st.markdown("---")

# --- 7. Model Insights and Performance Section ---
st.header("⚙️ Model Insights and Performance Overview")
st.write("Understand how the prediction model works and its overall accuracy.")

# 7.1 Feature Importance
st.subheader("7.1 Feature Importance")
st.write("Random Forest models indicate the relative importance of each feature in making predictions.")

# Get feature importances from the trained model
feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': model_features, # Use model_features for consistent order
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

st.dataframe(importance_df.set_index('Feature'))
st.caption("A higher 'Importance' value means the feature had more influence on the model's predicted number of deaths.")

# 7.2 Model Performance (Accuracy Metrics)
st.subheader("7.2 Model Performance (R-squared & RMSE)")
st.write("These metrics indicate how well the model explains the variability in New Deaths and its typical prediction error.")

# Recalculate metrics on the original data split (or load from train_covid_model.py)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure X_full and y_full are derived consistently with train_covid_model.py
# First, re-apply preprocessing that train_covid_model.py does for X and y
temp_covid_data = load_data() # Reload original data to get consistent preprocessing
temp_covid_data.dropna(subset=['population'], inplace=True)
temp_covid_data = temp_covid_data[temp_covid_data['population'] > 0].copy()

temp_covid_data['vaccination_coverage'] = temp_covid_data['people_fully_vaccinated'] / temp_covid_data['population']
temp_covid_data['vaccination_coverage'].fillna(0, inplace=True)

min_date_temp = temp_covid_data['date'].min()
temp_covid_data['days_since_start'] = (temp_covid_data['date'] - min_date_temp).dt.days

X_full = temp_covid_data[model_features] # Use the features the model was trained on
y_full = temp_covid_data['New_deaths']

# Handle NaN/Inf in target variable just like in training script
y_clean_eval = y_full[y_full >= 0]
y_transformed_eval = np.log1p(y_clean_eval)

valid_indices_eval = y_transformed_eval.dropna().index
X_filtered_eval = X_full.loc[valid_indices_eval]
y_filtered_transformed_eval = y_transformed_eval.loc[valid_indices_eval]

if np.isinf(y_filtered_transformed_eval).any():
    inf_indices_eval = np.isinf(y_filtered_transformed_eval)
    X_filtered_eval = X_filtered_eval.loc[~inf_indices_eval]
    y_filtered_transformed_eval = y_filtered_transformed_eval.loc[~inf_indices_eval]


_, X_test_eval, _, y_test_transformed_eval = train_test_split(
    X_filtered_eval, y_filtered_transformed_eval, test_size=0.2, random_state=42
)

test_predictions_transformed = model.predict(X_test_eval)
test_predictions = np.expm1(test_predictions_transformed)
test_predictions[test_predictions < 0] = 0 # Ensure non-negative

y_test_original_scale = temp_covid_data.loc[y_test_transformed_eval.index, 'New_deaths']

test_mse = mean_squared_error(y_test_original_scale, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_original_scale, test_predictions)

col_metrics1, col_metrics2 = st.columns(2)
with col_metrics1:
    st.metric(label="R-squared (R²)", value=f"{test_r2:.2f}")
    st.caption("R-squared measures how well future samples are likely to be predicted by the model. Higher is better (max 1.0).")
with col_metrics2:
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{test_rmse:.2f}")
    st.caption("RMSE represents the typical magnitude of the prediction error in the original units (number of deaths). Lower is better.")

st.markdown("---")

# --- 8. Project Details and Footer ---
st.header("About This Project")
st.write(
    """
    This project demonstrates a comprehensive data science workflow, including:
    -   **Data Loading & Preprocessing:** Handling time-series data, missing values, and feature engineering with `pandas`.
    -   **Exploratory Data Analysis (EDA):** Visualizing complex trends and distributions using `matplotlib` and `seaborn`.
    -   **Supervised Machine Learning (Regression):** Building and evaluating a `RandomForestRegressor` from `scikit-learn` to predict continuous outcomes.
    -   **Model Persistence:** Efficiently saving and loading trained models with `joblib`.
    -   **Interactive Dashboard Development:** Creating a dynamic and user-friendly web application with `Streamlit`.

    This dashboard serves as a hands-on example of applying data science to real-world health data.
    """
)
st.markdown("---")
st.write("Developed by Augustine Khumalo | Connect with me on LinkedIn!(https://www.linkedin.com/in/augustine-khumalo)")
