# This module contains all centralized configuration constants and settings
# for the COVID-19 dashboard application.
# This follows the DRY principle by having a single source of truth for all configuration.

# --- File Paths ---
DATA_FILE = 'covid_vaccination_mortality.csv'
DEATHS_MODEL_FILE = 'trained_deaths_model.pkl'
DEATHS_FEATURES_FILE = 'model_features_deaths.pkl'
VACC_MODEL_FILE = 'trained_vaccinations_model.pkl'
VACC_FEATURES_FILE = 'model_features_vaccinations.pkl'

# --- Dashboard UI Configuration ---
PAGE_TITLE = "COVID-19 Vaccination & Mortality Dashboard"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# --- Data Preprocessing Constants ---
# Numerical columns that should be filled with 0 when NaN
NUMERICAL_COLS_TO_FILL_ZERO = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'New_deaths', 'ratio'
]

# Features for the Deaths prediction model
FEATURES_DEATHS = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'population', 'ratio', 'vaccination_coverage', 'days_since_start'
]

# Features for the Daily Vaccinations prediction model
FEATURES_VACCINATIONS = [
    'people_vaccinated', 'people_fully_vaccinated', 'population',
    'vaccination_coverage', 'days_since_start', 'New_deaths'
]

# Default feature values for predictions
DEFAULT_FEATURE_VALUES = {
    'total_vaccinations': 100000.0,
    'people_vaccinated': 50000.0,
    'people_fully_vaccinated': 25000.0,
    'population': 10000000.0,
    'ratio': 0.05,
    'vaccination_coverage': 0.025,
    'New_deaths': 10.0,
}

# Columns for correlation analysis
CORRELATION_COLUMNS = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start',
    'new_deaths_per_million', 'total_vaccinations_per_hundred',
    'daily_vaccinations', 'daily_vaccinated_per_million',
    'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'
]

# Dashboard navigation pages
PAGES = {
    "📊 Data Overview": "data_overview",
    "📈 Trends & Insights": "trends_insights",
    "🔍 Data Diagnostics": "data_diagnostics",
    "🧪 Advanced Analysis": "advanced_analysis",
    "📈 Forecasting": "forecasting",
    "📊 Scenario Analysis": "scenario_analysis",
    "⚙️ Model Info & About": "model_info_about"
}

# Default countries for initial selection
DEFAULT_COUNTRIES = ['United States', 'India', 'Brazil']
