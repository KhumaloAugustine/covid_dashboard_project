# This file contains utility functions for data loading, model loading,
# and general helper functions used across the COVID-19 dashboard application.
# Follows Single Responsibility Principle with focused, cohesive functions.

import streamlit as st
import pandas as pd
import joblib

# Import centralized configuration
from config import (
    DATA_FILE, DEATHS_MODEL_FILE, DEATHS_FEATURES_FILE,
    VACC_MODEL_FILE, VACC_FEATURES_FILE, DEFAULT_COUNTRIES
)
from data_preprocessing import preprocess_covid_data

@st.cache_data
def load_data(file_path=DATA_FILE):
    """
    Loads the COVID-19 dataset from a CSV file and performs initial preprocessing.
    
    Uses the centralized data_preprocessing module for all preprocessing steps.
    
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
            data = preprocess_covid_data(data)
            return data
        except FileNotFoundError:
            st.error(f"Error: '{file_path}' not found. Please ensure it's in the same directory as the script.")
            st.stop()
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
    
    # Use default countries from config if available, otherwise use first 3 countries
    default_selection = [c for c in DEFAULT_COUNTRIES if c in all_countries]
    if not default_selection:
        default_selection = all_countries[:min(3, len(all_countries))]
    
    selected_countries = st.sidebar.multiselect(
        "Select Country(ies):",
        options=all_countries,
        default=default_selection
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