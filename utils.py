# This file contains utility functions for data loading, model loading,
# and general helper functions used across the COVID-19 dashboard application.

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# --- Configuration for file paths (can be moved to a separate config.py if more complex) ---
DATA_FILE = 'covid_vaccination_mortality.csv'
DEATHS_MODEL_FILE = 'trained_deaths_model.pkl'
DEATHS_FEATURES_FILE = 'model_features_deaths.pkl'
VACC_MODEL_FILE = 'trained_vaccinations_model.pkl'
VACC_FEATURES_FILE = 'model_features_vaccinations.pkl'

@st.cache_data
def load_data(file_path=DATA_FILE):
    """
    Loads the COVID-19 dataset from a CSV file and performs initial preprocessing.
    
    Preprocessing steps include:
    - Converting 'date' column to datetime objects.
    - Filling NaN values with 0 for specified numerical columns (total_vaccinations, etc.).
    - Ensuring 'population' is numeric and dropping rows with zero or missing population.
    - Engineering new features: 'vaccination_coverage', 'new_deaths_per_million',
      'total_vaccinations_per_hundred', 'daily_vaccinations',
      'daily_vaccinated_per_million', 'daily_deaths_growth_rate',
      'daily_vaccinations_growth_rate', and 'days_since_start'.
    
    Args:
        file_path (str): The path to the CSV data file.
        
    Returns:
        pandas.DataFrame: The preprocessed COVID-19 data.
    
    Raises:
        FileNotFoundError: If the specified data file is not found.
        Exception: For any other unexpected errors during data loading or processing.
    """
    with st.spinner(f"Loading and preparing data from {file_path}... This may take a moment."):
        try:
            data = pd.read_csv(file_path, index_col=0)
            data['date'] = pd.to_datetime(data['date'])

            # Fill NaNs with 0 for relevant numerical columns
            numerical_cols_to_fill_zero = [
                'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'New_deaths', 'ratio'
            ]
            for col in numerical_cols_to_fill_zero:
                data[col] = data[col].fillna(0)

            # Ensure population is numeric and handle potential zeros/NaNs
            data['population'] = pd.to_numeric(data['population'], errors='coerce')
            data.dropna(subset=['population'], inplace=True)
            data = data[data['population'] > 0] # Filter out entries with zero population

            # Feature Engineering: Create new, more insightful metrics
            data['vaccination_coverage'] = data['people_fully_vaccinated'] / data['population']
            data['vaccination_coverage'].fillna(0, inplace=True) # Handle division by zero or NaNs

            data['new_deaths_per_million'] = (data['New_deaths'] / data['population']) * 1_000_000
            data['total_vaccinations_per_hundred'] = (data['total_vaccinations'] / data['population']) * 100
            
            # Calculate daily vaccinations (difference from previous day's total_vaccinations)
            # Sorting by country and date is essential for correct difference calculation across groups.
            data = data.sort_values(by=['country', 'date']) 
            data['daily_vaccinations'] = data.groupby('country')['total_vaccinations'].diff().fillna(0)
            data['daily_vaccinations'] = data['daily_vaccinations'].apply(lambda x: max(0, x)) # Ensure non-negative daily counts
            
            data['daily_vaccinated_per_million'] = (data['daily_vaccinations'] / data['population']) * 1_000_000
            data['daily_vaccinated_per_million'].fillna(0, inplace=True) # Fill NaNs for consistency

            # Growth Rates (percentage change) for daily metrics
            # A small epsilon is added to the denominator for robustness against division by zero in pct_change.
            data['daily_deaths_growth_rate'] = data.groupby('country')['New_deaths'].pct_change().replace([np.inf, -np.inf], np.nan)
            data['daily_deaths_growth_rate'].fillna(0, inplace=True) 

            data['daily_vaccinations_growth_rate'] = data.groupby('country')['daily_vaccinations'].pct_change().replace([np.inf, -np.inf], np.nan)
            data['daily_vaccinations_growth_rate'].fillna(0, inplace=True)

            min_date = data['date'].min()
            data['days_since_start'] = (data['date'] - min_date).dt.days

            return data
        except FileNotFoundError:
            st.error(f"Error: '{file_path}' not found. Please ensure it's in the same directory as the script.")
            st.stop() # Stop the app if data is not found
        except Exception as e:
            st.error(f"An unexpected error occurred during data loading: {e}")
            st.stop()

@st.cache_resource
def load_models(deaths_model_path=DEATHS_MODEL_FILE, deaths_features_path=DEATHS_FEATURES_FILE, 
                vacc_model_path=VACC_MODEL_FILE, vacc_features_path=VACC_FEATURES_FILE):
    """
    Loads pre-trained machine learning models and their associated features using joblib.
    
    Args:
        deaths_model_path (str): Path to the trained deaths model (.pkl file).
        deaths_features_path (str): Path to the features list for the deaths model (.pkl file).
        vacc_model_path (str): Path to the trained vaccinations model (.pkl file).
        vacc_features_path (str): Path to the features list for the vaccinations model (.pkl file).
        
    Returns:
        tuple: A tuple containing two dictionaries (models, features).
               `models`: Dictionary with keys 'deaths' and 'vaccinations' holding the loaded models.
                         Value is None if a model file is not found.
               `features`: Dictionary with keys 'deaths' and 'vaccinations' holding the feature lists.
                           Value is None if a feature file is not found.
    """
    models_dict = {}
    features_dict = {}
    with st.spinner("Loading machine learning models..."):
        # Load Deaths Model
        try:
            models_dict['deaths'] = joblib.load(deaths_model_path)
            features_dict['deaths'] = joblib.load(deaths_features_path)
            st.sidebar.success("Deaths prediction model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error(f"Error: '{deaths_model_path}' or '{deaths_features_path}' not found. "
                             "Please ensure 'train_covid_model.py' has been run successfully to generate these files.")
            models_dict['deaths'] = None 
            features_dict['deaths'] = None
            st.stop() # Stop if the critical deaths model is missing
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred loading the deaths model: {e}")
            models_dict['deaths'] = None
            features_dict['deaths'] = None

        # Load Vaccinations Model
        try:
            models_dict['vaccinations'] = joblib.load(vacc_model_path)
            features_dict['vaccinations'] = joblib.load(vacc_features_path)
            st.sidebar.success("Daily vaccinations prediction model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.warning(f"Warning: '{vacc_model_path}' or '{vacc_features_path}' not found. "
                               "Daily vaccinations prediction will not be available. Please run 'train_covid_model.py'.")
            models_dict['vaccinations'] = None
            features_dict['vaccinations'] = None
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred loading the vaccinations model: {e}")
            models_dict['vaccinations'] = None
            features_dict['vaccinations'] = None
            
    return models_dict, features_dict

def get_default_input_value(data_frame, col_name, fallback_value):
    """
    Helper function to get a sensible default value for number input fields based on existing data.
    If the column exists in the DataFrame and has non-zero sum, its mean is used.
    Otherwise, a predefined fallback value is returned.
    
    Args:
        data_frame (pandas.DataFrame): The DataFrame to check for column data.
        col_name (str): The name of the column to check.
        fallback_value (float): A default value to use if the column is not suitable.
        
    Returns:
        float: The calculated default value or the fallback value.
    """
    if not data_frame.empty and col_name in data_frame.columns and data_frame[col_name].sum() > 0:
        return float(data_frame[col_name].mean())
    return float(fallback_value)

def setup_sidebar_filters(data):
    """
    Sets up the interactive filters in the Streamlit sidebar for country and date range.
    
    Args:
        data (pandas.DataFrame): The full COVID-19 dataset.
        
    Returns:
        tuple: A tuple containing (selected_countries, date_range).
               `selected_countries` (list): List of countries chosen by the user.
               `date_range` (tuple): A tuple (start_date, end_date) from the slider.
    """
    st.sidebar.header("📊 Global Data Filters")
    st.sidebar.write("Use these filters to customize the data displayed in the main sections of the dashboard.")

    all_countries = data['country'].unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "Select Country(ies):",
        options=all_countries,
        # Set a sensible default: US, India, Brazil if available, otherwise first 3 countries
        default=['United States', 'India', 'Brazil'] if 'United States' in all_countries else all_countries[:min(3, len(all_countries))]
    )

    min_date_data = data['date'].min().to_pydatetime()
    max_date_data = data['date'].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "Select Date Range:",
        min_value=min_date_data,
        max_value=max_date_data,
        value=(min_date_data, max_date_data), # Default to full range
        format="YYYY-MM-DD",
        help="Drag the ends of the slider to select a specific period, or click to adjust."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"Data last updated: **{data['date'].max().strftime('%Y-%m-%d')}**")

    # Button to clear all filters
    if st.sidebar.button("Reset Filters"):
        st.session_state.clear() # Clear all Streamlit session state
        st.rerun() # Rerun the app to apply default filters

    return selected_countries, date_range