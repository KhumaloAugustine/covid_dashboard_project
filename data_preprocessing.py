# This module contains data preprocessing and feature engineering functions
# shared between the dashboard application and model training script.
# This follows the DRY principle by having a single implementation for data preprocessing.

import pandas as pd
import numpy as np
from config import NUMERICAL_COLS_TO_FILL_ZERO


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values with 0 for specified numerical columns.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with missing values filled
    """
    for col in NUMERICAL_COLS_TO_FILL_ZERO:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    return data


def clean_population_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure population is numeric and remove rows with zero or missing population.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with cleaned population data
    """
    data['population'] = pd.to_numeric(data['population'], errors='coerce')
    data.dropna(subset=['population'], inplace=True)
    data = data[data['population'] > 0]
    return data


def calculate_vaccination_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate vaccination coverage as a proportion of the population.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with vaccination_coverage column added
    """
    data['vaccination_coverage'] = data['people_fully_vaccinated'] / data['population']
    data['vaccination_coverage'] = data['vaccination_coverage'].fillna(0)
    return data


def calculate_days_since_start(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of days since the earliest date in the dataset.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with days_since_start column added
    """
    min_date = data['date'].min()
    data['days_since_start'] = (data['date'] - min_date).dt.days
    return data


def calculate_daily_vaccinations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily vaccinations as the difference from previous day's total.
    
    Args:
        data: DataFrame to process (must be sorted by country and date)
        
    Returns:
        DataFrame with daily_vaccinations column added
    """
    data = data.sort_values(by=['country', 'date'])
    data['daily_vaccinations'] = data.groupby('country')['total_vaccinations'].diff().fillna(0)
    data['daily_vaccinations'] = data['daily_vaccinations'].apply(lambda x: max(0, x))
    return data


def calculate_per_capita_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-capita metrics for deaths and vaccinations.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with per-capita metrics added
    """
    data['new_deaths_per_million'] = (data['New_deaths'] / data['population']) * 1_000_000
    data['total_vaccinations_per_hundred'] = (data['total_vaccinations'] / data['population']) * 100
    data['daily_vaccinated_per_million'] = (data['daily_vaccinations'] / data['population']) * 1_000_000
    data['daily_vaccinated_per_million'] = data['daily_vaccinated_per_million'].fillna(0)
    return data


def calculate_growth_rates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily growth rates for deaths and vaccinations.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with growth rate columns added
    """
    data['daily_deaths_growth_rate'] = data.groupby('country')['New_deaths'].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    data['daily_deaths_growth_rate'] = data['daily_deaths_growth_rate'].fillna(0)
    
    data['daily_vaccinations_growth_rate'] = data.groupby('country')['daily_vaccinations'].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    data['daily_vaccinations_growth_rate'] = data['daily_vaccinations_growth_rate'].fillna(0)
    return data


def preprocess_covid_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing and feature engineering steps to the COVID-19 data.
    This is the main entry point for data preprocessing.
    
    Args:
        data: Raw DataFrame loaded from CSV
        
    Returns:
        Fully preprocessed DataFrame ready for analysis
    """
    # Convert date column
    data['date'] = pd.to_datetime(data['date'])
    
    # Apply preprocessing steps in order
    data = fill_missing_values(data)
    data = clean_population_data(data)
    data = calculate_vaccination_coverage(data)
    data = calculate_daily_vaccinations(data)
    data = calculate_per_capita_metrics(data)
    data = calculate_growth_rates(data)
    data = calculate_days_since_start(data)
    
    return data


def prepare_model_data(data: pd.DataFrame, target_col: str, features_list: list) -> tuple:
    """
    Prepare data for model training by handling transformations and cleaning.
    
    Args:
        data: Preprocessed DataFrame
        target_col: Name of the target column
        features_list: List of feature column names
        
    Returns:
        Tuple of (X, y_transformed, valid_indices) or (None, None, None) if no valid data
    """
    y = data[target_col]
    
    # Apply log1p transformation to handle skewed data
    y_clean = y[y >= 0]
    y_transformed = np.log1p(y_clean)
    
    # Align X and y after filtering
    valid_indices = y_transformed.dropna().index
    X = data.loc[valid_indices, features_list]
    y_filtered_transformed = y_transformed.loc[valid_indices]
    
    # Handle potential inf values
    if np.isinf(y_filtered_transformed).any():
        inf_indices = np.isinf(y_filtered_transformed)
        X = X.loc[~inf_indices]
        y_filtered_transformed = y_filtered_transformed.loc[~inf_indices]
    
    # Drop rows with NaN in features
    initial_rows = len(X)
    X = X.dropna()
    y_filtered_transformed = y_filtered_transformed.loc[X.index]
    
    rows_dropped = initial_rows - len(X)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to NaN values in features for {target_col}.")
    
    if X.empty:
        return None, None, None
    
    return X, y_filtered_transformed, X.index
