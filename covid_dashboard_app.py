# covid_dashboard_app.py
# This script creates an interactive Streamlit dashboard
# for analyzing COVID-19 vaccination and mortality data,
# and predicting new deaths or daily vaccinations using trained regression models.
# It now features a simplified sidebar-based navigation for improved usability,
# especially on mobile, ensuring tabs are always accessible.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split # For model evaluation
from sklearn.metrics import mean_squared_error, r2_score # For model evaluation metrics

# Import utility functions from utils.py and plotting_utils.py
from utils import load_data, load_models, get_default_input_value, setup_sidebar_filters
from plotting_utils import (
    display_summary_metrics, display_raw_data_sample, display_summary_statistics,
    display_country_comparison_table, display_latest_country_stats,
    generate_bar_plots_for_period_summary, display_top_bottom_countries,
    display_data_dictionary, plot_time_series, plot_vaccination_progress,
    plot_cumulative_and_rolling_averages, plot_distributions_and_correlations,
    plot_interactive_scatter, plot_comparative_distributions, plot_daily_change,
    plot_latest_vaccination_status_distribution, plot_monthly_yearly_trends,
    plot_growth_rate_trends, plot_lag_plot, plot_daily_metrics_comparison_for_date,
    plot_geographic_distribution_map, plot_pair_plot, plot_time_series_decomposition,
    plot_outliers_boxplot, display_data_types_and_uniques, plot_missing_values_heatmap,
    display_simple_forecast, display_advanced_forecast, display_scenario_analysis
)

# --- Configuration ---
# Centralize key configurations for easier management and scalability.
PAGE_TITLE = "COVID-19 Vaccination & Mortality Dashboard"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded" # Set sidebar to expanded by default for better visibility
DATA_FILE = 'covid_vaccination_mortality.csv'
DEATHS_MODEL_FILE = 'trained_deaths_model.pkl'
DEATHS_FEATURES_FILE = 'model_features_deaths.pkl'
VACC_MODEL_FILE = 'trained_vaccinations_model.pkl'
VACC_FEATURES_FILE = 'model_features_vaccinations.pkl'

# --- 1. Dashboard Configuration and Title ---
st.set_page_config(
    page_title=PAGE_TITLE,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

st.title("🦠 COVID-19 Vaccination & Mortality Dashboard")
st.markdown("---")

# --- Initial User Guide / Getting Started ---
def display_initial_guide():
    """Displays the initial welcome message and guide for the user."""
    st.info(
        """
        **Welcome!** This interactive dashboard allows you to explore COVID-19 vaccination and mortality data from various countries.
        
        **To get started:**
        1.  Use the **filters on the left sidebar** to select specific countries and a date range relevant to your analysis.
        2.  Navigate through the **sections using the sidebar menu** to view detailed data overviews, interactive trends and insights, prediction models, and information about the models used.
        
        This dashboard is designed to be user-friendly and responsive across different devices, including mobile phones, tablets, and laptops.
        """
    )
    st.markdown("---")

display_initial_guide()


# --- 2. Load Data and Pre-trained Models ---
# Data loading now handled by utils.py
covid_data = load_data(DATA_FILE)

# Model loading now handled by utils.py
models, model_features_dict = load_models(DEATHS_MODEL_FILE, DEATHS_FEATURES_FILE, VACC_MODEL_FILE, VACC_FEATURES_FILE)

# --- 3. Interactive Data Filtering (Sidebar) ---
selected_countries, date_range = setup_sidebar_filters(covid_data)

filtered_data = covid_data[
    covid_data['country'].isin(selected_countries) &
    (covid_data['date'] >= date_range[0]) &
    (covid_data['date'] <= date_range[1])
].copy()

if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your country and date range selections in the sidebar.")
    st.stop()

# Define default_days_since_start_date for prediction tab, accessible globally
default_days_since_start_date_for_input = (covid_data['date'].max() + pd.Timedelta(days=7)).date()


# --- Section Functions (formerly Tab functions) ---
def display_prediction_tool_section(filtered_data, covid_data, models, model_features_dict, default_days_since_start_date_for_input):
    """
    Manages the content for the Prediction Section.
    
    Args:
        filtered_data (pandas.DataFrame): The currently filtered data (for default input values).
        covid_data (pandas.DataFrame): The full data (for calculating days_since_start base date).
        models (dict): Dictionary of loaded ML models.
        model_features_dict (dict): Dictionary of features required by each model.
        default_days_since_start_date_for_input (datetime.date): Default date for prediction input.
    """
    st.header("🔮 COVID-19 Prediction Tool")
    st.write("Use our trained machine learning models to predict future **New Deaths** or **Daily Vaccinations** based on input features. Adjust the sliders and input fields below to see how different scenarios affect the predictions.")

    prediction_type = st.radio(
        "Choose Prediction Type:",
        ('New Deaths', 'Daily Vaccinations'),
        key='prediction_type_selector',
        help="Select whether you want to predict daily new deaths or daily vaccination counts."
    )

    current_model = None
    current_features = []
    prediction_label = ""
    model_key = 'deaths' if prediction_type == 'New Deaths' else 'vaccinations'

    if models.get(model_key):
        current_model = models[model_key]
        current_features = model_features_dict[model_key]
        prediction_label = f"Predicted {prediction_type}"
    
    if current_model is None:
        st.warning(f"The model for '{prediction_type}' is not loaded. Please ensure 'train_covid_model.py' has been run successfully to generate the necessary model files.")
    else:
        st.write(f"Enter values for the features below to predict **{prediction_type}**: *(Default values are based on the average of your currently filtered data.)*")

        input_values = {}
        input_cols = st.columns(3)

        feature_defaults = {
            'total_vaccinations': 100000.0, 'people_vaccinated': 50000.0, 'people_fully_vaccinated': 25000.0,
            'population': 10000000.0, 'ratio': 0.05, 'vaccination_coverage': 0.025, 'New_deaths': 10.0, 
        }

        with input_cols[0]:
            for feature in ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']:
                if feature in current_features:
                    input_values[feature] = st.number_input(
                        feature.replace('_', ' ').title() + ":",
                        min_value=0.0, value=get_default_input_value(filtered_data, feature, feature_defaults.get(feature, 0.0)),
                        step=10000.0, format="%.0f", key=f'pred_input_{feature}'
                    )
        with input_cols[1]:
            for feature in ['population', 'ratio', 'vaccination_coverage']:
                if feature in current_features:
                    min_val = 1.0 if feature == 'population' else 0.0
                    max_val = 1.0 if feature == 'vaccination_coverage' else None
                    step_val = 0.01 if feature in ['ratio', 'vaccination_coverage'] else 100000.0
                    format_str = "%.4f" if feature in ['ratio', 'vaccination_coverage'] else "%.0f"

                    input_values[feature] = st.number_input(
                        feature.replace('_', ' ').title() + ":",
                        min_value=min_val, max_value=max_val,
                        value=get_default_input_value(filtered_data, feature, feature_defaults.get(feature, min_val)),
                        step=step_val, format=format_str, key=f'pred_input_{feature}'
                    )
        with input_cols[2]:
            prediction_date_input = st.date_input("Select Prediction Date:",
                                            value=default_days_since_start_date_for_input,
                                            min_value=covid_data['date'].min().date(),
                                            key='prediction_date_input_section3' # Updated key
                                            )
            
            prediction_datetime = datetime.datetime.combine(prediction_date_input, datetime.time.min)
            input_values['days_since_start'] = (prediction_datetime - covid_data['date'].min().to_pydatetime()).days
            st.info(f"Calculated days since start: **{input_values['days_since_start']}** days.")
            
            if 'New_deaths' in current_features: 
                input_values['New_deaths'] = st.number_input("New Deaths (for Daily Vaccinations Model):",
                                                             min_value=0.0,
                                                             value=get_default_input_value(filtered_data, 'New_deaths', feature_defaults.get('New_deaths', 0.0)),
                                                             step=100.0, format="%.0f", key='pred_input_new_deaths'
                                                            )

        final_input_for_prediction = pd.DataFrame([input_values])[current_features]

        if st.button(f"Predict {prediction_type}", key='predict_button'):
            with st.spinner(f"Generating {prediction_type.lower()} prediction..."):
                try:
                    predicted_value_transformed = current_model.predict(final_input_for_prediction)[0]
                    predicted_value = np.expm1(predicted_value_transformed)
                    predicted_value = max(0, round(predicted_value)) 

                    st.success(f"**{prediction_label}:** {int(predicted_value):,} ")
                    st.caption("*(Prediction is an estimate based on the model's training data. Results may vary and are best interpreted in the context of the model's performance metrics.)*")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}. Please check your input values.")
                    st.warning("Ensure all input values are valid numbers. If the issue persists, verify that the trained model files (`.pkl` files) are correctly generated and loaded.")
    st.markdown("---")

def display_model_insights_section(models, model_features_dict, full_data):
    """
    Manages the content for the Model Insights and Performance section.
    
    Args:
        models (dict): Dictionary of loaded ML models.
        model_features_dict (dict): Dictionary of features required by each model.
        full_data (pandas.DataFrame): The complete, unfiltered dataset for evaluation.
    """
    st.header("⚙️ Model Insights and Performance Overview")
    st.write("Understand how our machine learning models work, which factors they consider most important, and how accurate their predictions are.")

    model_insight_selector = st.selectbox(
        "Select Model to View Insights:",
        options=['New Deaths Model', 'Daily Vaccinations Model'],
        key='model_insight_selector_section' # Updated key
    )

    selected_model_name_key = 'deaths' if model_insight_selector == 'New Deaths Model' else 'vaccinations'
    selected_model = models.get(selected_model_name_key)
    selected_features = model_features_dict.get(selected_model_name_key)

    if selected_model is None:
        st.warning(f"Model for '{model_insight_selector}' is not loaded. Please run 'train_covid_model.py' to generate model files before viewing its insights.")
    else:
        st.subheader(f"7.1 Feature Importance for {model_insight_selector}")
        st.write("This table shows which features (input variables) the Random Forest model considered most important when making its predictions. A higher 'Importance' value means the feature had more influence on the model's predicted outcome.")

        with st.spinner("Calculating feature importances..."):
            feature_importances = selected_model.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            st.dataframe(importance_df.set_index('Feature'))
            st.caption("Feature importance is calculated based on how much each feature contributes to reducing impurity in the decision trees of the Random Forest.")

        st.subheader(f"7.2 Model Performance for {model_insight_selector} (R-squared & RMSE)")
        st.write("These metrics indicate how well the model performs on unseen data. They help you understand the reliability and accuracy of the predictions.")

        with st.spinner(f"Calculating {model_insight_selector} performance metrics..."):
            target_col_eval = 'New_deaths' if selected_model_name_key == 'deaths' else 'daily_vaccinations'
            features_list_eval = model_features_dict[selected_model_name_key]

            y_full_temp_eval = full_data[target_col_eval]
            y_clean_eval_temp = y_full_temp_eval[y_full_temp_eval >= 0]
            y_transformed_eval_temp = np.log1p(y_clean_eval_temp)

            valid_indices_eval_temp = y_transformed_eval_temp.dropna().index 
            X_full_temp_eval = full_data.loc[valid_indices_eval_temp, features_list_eval]
            y_filtered_transformed_eval_temp = y_transformed_eval_temp.loc[valid_indices_eval_temp]

            if np.isinf(y_filtered_transformed_eval_temp).any():
                inf_indices_eval_temp = np.isinf(y_filtered_transformed_eval_temp)
                X_full_temp_eval = X_full_temp_eval.loc[~inf_indices_eval_temp]
                y_filtered_transformed_eval_temp = y_filtered_transformed_eval_temp.loc[~inf_indices_eval_temp]

            initial_rows_eval = len(X_full_temp_eval)
            X_full_temp_eval.dropna(inplace=True)
            y_filtered_transformed_eval_temp = y_filtered_transformed_eval_temp.loc[X_full_temp_eval.index]
            if len(X_full_temp_eval) < initial_rows_eval:
                st.warning(f"For model evaluation of {model_insight_selector}, dropped {initial_rows_eval - len(X_full_temp_eval)} rows due to missing feature values.")
            
            if X_full_temp_eval.empty:
                st.info(f"Not enough clean data to perform robust evaluation for {model_insight_selector}. Try a wider date range or more countries.")
            else:
                X_train_eval, X_test_eval, y_train_transformed_eval, y_test_transformed_eval = train_test_split(
                    X_full_temp_eval, y_filtered_transformed_eval_temp, test_size=0.2, random_state=42
                )

                test_predictions_transformed = selected_model.predict(X_test_eval)
                test_predictions = np.expm1(test_predictions_transformed)
                test_predictions[test_predictions < 0] = 0

                y_test_original_scale = full_data.loc[y_test_transformed_eval.index, target_col_eval]

                test_mse = mean_squared_error(y_test_original_scale, test_predictions)
                test_rmse = np.sqrt(test_mse)
                test_r2 = r2_score(y_test_original_scale, test_predictions)

                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric(label="R-squared (R²)", value=f"{test_r2:.2f}")
                with col_metrics2:
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{test_rmse:,.2f}")
    st.markdown("---")

def display_about_section():
    """Displays information about the project and data source."""
    st.header("About This Project")
    st.write(
        """
        This project demonstrates a comprehensive data science workflow, from data preparation to interactive visualization and machine learning prediction. Key components include:
        -   **Data Loading & Preprocessing:** Efficiently handling time-series data, imputing missing values, and engineering new features using `pandas`.
        -   **Exploratory Data Analysis (EDA):** Creating insightful visualizations with `matplotlib` and `seaborn` to uncover trends, distributions, and relationships within the data.
        -   **Supervised Machine Learning (Regression):** Building and evaluating robust `RandomForestRegressor` models from `scikit-learn` to predict continuous outcomes (new deaths and daily vaccinations).
        -   **Model Persistence:** Saving and loading trained models using `joblib` for efficient deployment and reuse.
        -   **Interactive Web Application Development:** Building a user-friendly dashboard with `Streamlit`.
        -   **Data Visualization:** Creating informative plots with `matplotlib` and `seaborn`.

        **Data Source:**
        The dataset used for this project is **"COVID vaccination vs. mortality"** from Kaggle:
        [https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality](https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality)

        **Dataset Context:**
        The COVID-19 pandemic significantly impacted global health. This dataset was compiled to help investigate the potential relationship between coronavirus vaccination efforts and mortality rates, providing valuable time-series data on vaccination progress and daily death counts across various countries.

        This dashboard serves as a practical example of applying data science methodologies to real-world public health data.
        """
    )
    st.markdown("---")
    st.write("Developed by Augustine Khumalo")
    st.markdown("""
    **Connect with me:**
    * **LinkedIn:** [Augustine Khumalo](https://www.linkedin.com/in/augustine-khumalo)
    * **Mobile:** +27 65 857 3653 
    * **Email:** mr.a.s.khumalo@gmail.com 
    """)

def display_data_preprocessing_section(full_data):
    """
    Displays insights into the data preprocessing steps.
    
    Args:
        full_data (pandas.DataFrame): The complete, unfiltered data.
    """
    st.header("🗄️ Data Preprocessing & Feature Engineering Overview")
    st.write("This section provides insights into how the raw data is cleaned, transformed, and augmented with new features before being used for analysis and modeling.")

    st.subheader("Initial Data Snapshot and Missing Values")
    st.write("Before preprocessing, raw data often contains missing values and may not be in the optimal format for analysis. Here's a look at the missing values in the raw dataset.")
    
    # Reload raw data for demonstration purposes, without caching
    try:
        raw_data = pd.read_csv(DATA_FILE, index_col=0)
        st.write("#### Raw Data Info (First 5 Rows)")
        st.dataframe(raw_data.head())
        
        st.write("#### Missing Values (NaN Counts)")
        missing_data = raw_data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if not missing_data.empty:
            st.dataframe(missing_data.to_frame(name='Missing Count'))
            st.caption("These missing values are handled by filling with zeros or dropping rows, as described below.")
        else:
            st.info("No missing values found in the raw dataset (or they are all zeros).")

    except FileNotFoundError:
        st.error(f"Raw data file '{DATA_FILE}' not found. Cannot display raw data insights.")
    st.markdown("---")

    st.subheader("Data Cleaning Steps")
    st.markdown(
        """
        The following steps are applied during data loading:
        - **Date Conversion:** The `date` column is converted to datetime objects to enable time-series analysis.
        - **Handling Missing Numerical Values:** `NaN` values in `total_vaccinations`, `people_vaccinated`, `people_fully_vaccinated`, `New_deaths`, and `ratio` are filled with `0`. This assumes that missing vaccination/death counts imply no new cases/vaccinations on that day.
        - **Population Data Cleaning:** The `population` column is converted to a numeric type, and any rows where `population` is `NaN` or `0` are removed. This is crucial for per-capita calculations.
        - **Ensuring Non-Negative Daily Vaccinations:** `daily_vaccinations` are ensured to be non-negative by taking `max(0, x)`.
        - **Handling Infinite Growth Rates:** Infinite values in `daily_deaths_growth_rate` and `daily_vaccinations_growth_rate` (which can occur from division by zero) are replaced with `NaN`, and then `NaN`s are filled with `0`.
        """
    )
    st.markdown("---")

    st.subheader("Feature Engineering")
    st.markdown(
        """
        New, more insightful features are derived from the raw data to provide a richer context for analysis and better inputs for the machine learning models:
        - **`vaccination_coverage`:** Calculated as `people_fully_vaccinated / population`. This represents the proportion of the population that is fully vaccinated (ranges from 0 to 1).
        - **`new_deaths_per_million`:** Calculated as `(New_deaths / population) * 1,000,000`. This normalizes daily death counts by population size, making comparisons across countries more meaningful.
        - **`total_vaccinations_per_hundred`:** Calculated as `(total_vaccinations / population) * 100`. This normalizes total vaccinations by population, providing a per-capita vaccination rate.
        - **`daily_vaccinations`:** Calculated as the daily difference in `total_vaccinations` for each country. This represents the number of vaccine doses administered on a given day.
        - **`daily_vaccinated_per_million`:** Calculated as `(daily_vaccinations / population) * 1,000,000`. Normalizes daily vaccination rates by population.
        - **`daily_deaths_growth_rate`:** The percentage change in `New_deaths` from the previous day. Helps identify acceleration or deceleration of deaths.
        - **`daily_vaccinations_growth_rate`:** The percentage change in `daily_vaccinations` from the previous day. Indicates the pace of vaccination efforts.
        - **`days_since_start`:** The number of days that have passed since the earliest date in the entire dataset. This time-based feature can capture overall trends in time-series models.
        """
    )
    st.markdown("---")

    st.subheader("Final Data Snapshot (After Preprocessing)")
    st.write("Here's a look at the data after all cleaning and feature engineering steps have been applied. This is the dataset used for all analyses and predictions in the dashboard.")
    st.dataframe(full_data.head())
    st.write(f"Total entries after preprocessing: {len(full_data):,} from {full_data['country'].nunique()} countries.")
    st.markdown("---")


# --- Main Application Flow (using sidebar navigation) ---
# Define the navigation options for the sidebar
PAGES = {
    "📊 Data Overview": "data_overview",
    "📈 Trends & Insights": "trends_insights",
    "🔍 Data Diagnostics": "data_diagnostics",
    "🧪 Advanced Analysis": "advanced_analysis",
    "📈 Forecasting": "forecasting",
    "📊 Scenario Analysis": "scenario_analysis",
    "⚙️ Model Info & About": "model_info_about"
}

st.sidebar.title("Dashboard Navigation")
selected_page = st.sidebar.radio("Go to:", list(PAGES.keys()))

# Use if/elif/else to display content based on selection
if selected_page == "📊 Data Overview":
    st.header("📊 Data Overview & Key Statistics")
    st.write(f"Explore the raw data and key aggregated statistics for **{', '.join(selected_countries)}** from **{date_range[0].strftime('%Y-%m-%d')}** to **{date_range[1].strftime('%Y-%m-%d')}**.")
    display_summary_metrics(filtered_data)
    display_raw_data_sample(filtered_data)
    display_summary_statistics(filtered_data)
    display_country_comparison_table(filtered_data)
    display_latest_country_stats(filtered_data)
    generate_bar_plots_for_period_summary(filtered_data)
    
    all_countries_count_for_top_bottom = len(filtered_data['country'].unique())
    if all_countries_count_for_top_bottom > 1:
        display_top_bottom_countries(filtered_data, 'New_deaths', 'Daily New Deaths', "Adjust 'N' to see more or fewer top/bottom countries by average daily deaths.")
        display_top_bottom_countries(filtered_data, 'total_vaccinations', 'Total Vaccinations', "Adjust 'N' to see more or fewer top/bottom vaccinated countries.")
        display_top_bottom_countries(filtered_data, 'daily_deaths_growth_rate', 'Daily Deaths Growth Rate', "Adjust 'N' to see more or fewer top/bottom countries by death growth rate.")
        display_top_bottom_countries(filtered_data, 'daily_vaccinations_growth_rate', 'Daily Vaccinations Growth Rate', "Adjust 'N' to see more or fewer top/bottom countries by vaccination growth rate.")
    else:
        st.info("Select multiple countries in the sidebar to view top/bottom countries by various metrics.")
    display_data_dictionary()

elif selected_page == "📈 Trends & Insights":
    st.header("📈 Dynamic COVID-19 Trends & Insights")
    st.write("Visualize how key COVID-19 metrics have evolved over time and explore relationships between them. All plots automatically update based on your country and date selections from the sidebar.")
    plot_time_series(filtered_data)
    plot_vaccination_progress(filtered_data)
    plot_cumulative_and_rolling_averages(filtered_data)
    plot_distributions_and_correlations(filtered_data)
    plot_interactive_scatter(filtered_data)
    plot_comparative_distributions(filtered_data) 
    plot_daily_change(filtered_data)
    plot_latest_vaccination_status_distribution(filtered_data, date_range[1])
    plot_monthly_yearly_trends(filtered_data)
    plot_growth_rate_trends(filtered_data)
    plot_lag_plot(filtered_data)
    plot_daily_metrics_comparison_for_date(filtered_data)
    plot_geographic_distribution_map(covid_data) 

elif selected_page == "🔍 Data Diagnostics":
    st.header("🔍 Data Diagnostics & Quality Check")
    st.write("This section provides insights into the data's quality, completeness, and structure, which are crucial for reliable analysis.")
    display_data_types_and_uniques(filtered_data)
    plot_missing_values_heatmap(filtered_data) 

elif selected_page == "🧪 Advanced Analysis":
    st.header("🧪 Advanced Analysis & Time Series Insights")
    st.write("Explore more sophisticated analyses for in-depth understanding of COVID-19 data patterns.")
    plot_pair_plot(filtered_data)
    plot_time_series_decomposition(filtered_data)
    plot_outliers_boxplot(filtered_data)

elif selected_page == "📈 Forecasting":
    st.header("📈 COVID-19 Forecasting")
    st.write("This section demonstrates basic and conceptual advanced forecasting capabilities for key COVID-19 metrics. Select a single country in the sidebar for best results.")
    display_simple_forecast(filtered_data)
    display_advanced_forecast(filtered_data)

elif selected_page == "📊 Scenario Analysis":
    st.header("📊 Scenario Analysis")
    st.write("Explore 'what-if' scenarios by adjusting key input parameters and observing their impact on predicted outcomes (New Deaths or Daily Vaccinations) based on our machine learning models.")
    display_scenario_analysis(filtered_data, models, model_features_dict, covid_data)

elif selected_page == "⚙️ Model Info & About":
    display_model_insights_section(models, model_features_dict, covid_data)
    display_about_section()
    st.subheader("Data Preprocessing Steps") 
    display_data_preprocessing_section(covid_data)
