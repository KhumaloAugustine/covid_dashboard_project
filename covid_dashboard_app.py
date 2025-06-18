# This script creates an interactive Streamlit dashboard
# for analyzing COVID-19 vaccination and mortality data,
# and predicting new deaths using a trained regression model.
# It now features a simplified tabbed interface and enhanced interactivity,
# with a stronger focus on comparative analysis for multiple selected countries.

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

# --- Initial User Guide / Getting Started ---
st.info(
    """
    **Welcome!** This interactive dashboard allows you to explore COVID-19 vaccination and mortality data.
    
    **To get started:**
    1.  Use the **filters on the left sidebar** to select specific countries and a date range.
    2.  Navigate through the **tabs below** to view data overviews, trends, predictions, and model insights.
    """
)
st.markdown("---")


# --- 2. Load Data and Pre-trained Model ---
@st.cache_data
def load_data():
    """Loads the COVID-19 dataset and performs initial preprocessing."""
    with st.spinner("Loading and preprocessing data..."):
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
with st.spinner("Loading machine learning model..."):
    try:
        model = joblib.load('trained_covid_model.pkl')
        model_features = joblib.load('model_features.pkl')
        st.sidebar.success("Model and data loaded successfully!")
    except FileNotFoundError:
        st.error("Error: 'trained_covid_model.pkl' or 'model_features.pkl' not found. Please run 'train_covid_model.py' first.")
        st.stop()

# --- 3. Dashboard Introduction (This content is now moved to the initial info block) ---
# st.write(...)

# --- 4. Interactive Data Filtering (Sidebar) ---
st.sidebar.header("📊 Global Data Filters")
st.sidebar.write("Use these filters to customize the data displayed in the main sections.")

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

# --- Main Content Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Trends & Insights", "🔮 Prediction", "⚙️ Model & About"])

with tab1:
    st.header("📊 Data Overview & Statistics")
    st.write(f"Showing raw data and key statistics for **{', '.join(selected_countries)}** from **{date_range[0].strftime('%Y-%m-%d')}** to **{date_range[1].strftime('%Y-%m-%d')}**.")

    st.subheader("Summary Metrics for Filtered Data")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric(label="Total New Deaths (Selected Period)",
                  value=f"{int(filtered_data['New_deaths'].sum()):,}")
    with col_metrics2:
        st.metric(label="Max Total Vaccinations (Selected Period)",
                  value=f"{int(filtered_data['total_vaccinations'].max()):,}")
    with col_metrics3:
        st.metric(label="Average Vaccination Coverage",
                  value=f"{filtered_data['vaccination_coverage'].mean():.2%}")
    st.markdown("---")

    st.subheader("Raw Filtered Data Sample")
    with st.spinner("Loading raw data sample..."):
        st.dataframe(filtered_data.head())
        st.caption(f"Displaying the first 5 rows of {len(filtered_data)} entries. This sample reflects **all active filters** (country and date range).")

    st.subheader("Summary Statistics for Numerical Data")
    with st.spinner("Calculating summary statistics..."):
        st.dataframe(filtered_data.describe().transpose()) # Transpose for better readability
        st.caption("Descriptive statistics for numerical columns in the filtered dataset.")

    # --- New: Country Comparison Table ---
    if len(selected_countries) > 1:
        st.subheader("Country Comparison: Key Aggregated Metrics")
        with st.spinner("Generating country comparison table..."):
            comparison_table = filtered_data.groupby('country').agg(
                Total_New_Deaths=('New_deaths', 'sum'),
                Max_Total_Vaccinations=('total_vaccinations', 'max'),
                Avg_Daily_New_Deaths=('New_deaths', 'mean'),
                Avg_Vaccination_Coverage=('vaccination_coverage', 'mean'),
                Max_Population=('population', 'max')
            ).reset_index()
            # Format columns for better readability
            comparison_table['Total_New_Deaths'] = comparison_table['Total_New_Deaths'].apply(lambda x: f"{int(x):,}")
            comparison_table['Max_Total_Vaccinations'] = comparison_table['Max_Total_Vaccinations'].apply(lambda x: f"{int(x):,}")
            comparison_table['Avg_Daily_New_Deaths'] = comparison_table['Avg_Daily_New_Deaths'].apply(lambda x: f"{x:.2f}")
            comparison_table['Avg_Vaccination_Coverage'] = comparison_table['Avg_Vaccination_Coverage'].apply(lambda x: f"{x:.2%}")
            comparison_table['Max_Population'] = comparison_table['Max_Population'].apply(lambda x: f"{int(x):,}")
            
            st.dataframe(comparison_table.set_index('country'))
            st.caption("Aggregated statistics for each selected country over the chosen period. 'Max' values represent the highest recorded within the period.")
    else:
        st.info("Select multiple countries in the sidebar to view a comparative summary table.")

    st.markdown("---")

    st.subheader("Key Statistics per Country (Latest Data Point)")
    with st.spinner("Calculating latest country statistics..."):
        # Get the latest data point for each country within the filtered range
        latest_country_stats = filtered_data.groupby('country').agg(
            Latest_Population=('population', 'max'), # Max population in the period
            Latest_Total_Vaccinations=('total_vaccinations', 'max'),
            Latest_People_Fully_Vaccinated=('people_fully_vaccinated', 'max'),
            Latest_Vaccination_Coverage=('vaccination_coverage', 'max'),
            Average_Daily_New_Deaths=('New_deaths', 'mean')
        ).reset_index()

        st.dataframe(latest_country_stats.set_index('country'))
        st.caption("Shows the maximum recorded values for relevant metrics within the selected date range for each country.")

    st.subheader("Total Deaths and Vaccinations for Selected Period (Bar Plots)")
    with st.spinner("Generating bar plots for totals..."):
        period_summary = filtered_data.groupby('country').agg(
            Total_New_Deaths=('New_deaths', 'sum'),
            Highest_Total_Vaccinations_Reached=('total_vaccinations', 'max')
        ).reset_index()
        
        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            # Bar plot for Total New Deaths per country
            fig_total_deaths, ax_total_deaths = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5))) # Dynamic height
            sns.barplot(data=period_summary.sort_values('Total_New_Deaths', ascending=False), x='Total_New_Deaths', y='country', palette='viridis', ax=ax_total_deaths)
            ax_total_deaths.set_title('Total New Deaths per Country (Selected Period)')
            ax_total_deaths.set_xlabel('Total New Deaths')
            ax_total_deaths.set_ylabel('Country')
            ax_total_deaths.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_total_deaths)
        
        with col_bar2:
            # Bar plot for Highest Total Vaccinations Reached per country
            fig_total_vacc, ax_total_vacc = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5))) # Dynamic height
            sns.barplot(data=period_summary.sort_values('Highest_Total_Vaccinations_Reached', ascending=False), x='Highest_Total_Vaccinations_Reached', y='country', palette='cividis', ax=ax_total_vacc)
            ax_total_vacc.set_title('Highest Total Vaccinations Reached per Country')
            ax_total_vacc.set_xlabel('Highest Total Vaccinations')
            ax_total_vacc.set_ylabel('Country')
            ax_total_vacc.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_total_vacc)

    st.markdown("---")

    st.subheader("Top/Bottom Countries by Average Daily Deaths")
    with st.spinner("Calculating top/bottom countries..."):
        avg_daily_deaths = filtered_data.groupby('country')['New_deaths'].mean().sort_values(ascending=False)
        
        col_top_bottom1, col_top_bottom2 = st.columns(2)
        with col_top_bottom1:
            st.write("#### Top 5 Countries by Average Daily New Deaths")
            st.dataframe(avg_daily_deaths.head(5).reset_index().rename(columns={'New_deaths': 'Avg. Daily Deaths'}).set_index('country'))
        with col_top_bottom2:
            st.write("#### Bottom 5 Countries by Average Daily New Deaths")
            st.dataframe(avg_daily_deaths.tail(5).reset_index().rename(columns={'New_deaths': 'Avg. Daily Deaths'}).set_index('country'))

    st.markdown("---")


with tab2:
    st.header("📈 Dynamic COVID-19 Trends & Insights")
    st.write("These visualizations update automatically based on the filters applied in the 'Data Overview' tab.")

    # 5.1 Time Series Plots - Now with selectable metric
    st.subheader("5.1 Customizable Time Series Trends")
    time_series_metric = st.selectbox(
        "Select Metric to Plot Over Time:",
        options=['New_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'vaccination_coverage'],
        index=0 # Default to New Deaths
    )
    
    with st.spinner(f"Generating time series plot for {time_series_metric.replace('_', ' ').title()}..."):
        fig_ts, ax_ts = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=filtered_data, x='date', y=time_series_metric, hue='country', marker='o', ax=ax_ts)
        ax_ts.set_title(f'{time_series_metric.replace("_", " ").title()} Over Time')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel(time_series_metric.replace('_', ' ').title())
        ax_ts.tick_params(axis='x', rotation=45)
        ax_ts.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig_ts)

    st.markdown("---")

    # --- Additional Time-Series Analysis (Cumulative & Rolling Averages) ---
    st.subheader("5.2 Additional Time-Series Analysis: Cumulative & Rolling Averages")
    st.write("These plots provide a 'survival-like' perspective by showing cumulative totals and smoothed trends.")

    # Sort data by country and date to ensure correct cumulative sums and rolling averages
    filtered_data_sorted = filtered_data.sort_values(by=['country', 'date'])

    col_add_ts1, col_add_ts2 = st.columns(2)

    with col_add_ts1:
        st.write("#### Cumulative New Deaths Over Time")
        with st.spinner("Calculating cumulative deaths..."):
            filtered_data_sorted['Cumulative_New_Deaths'] = filtered_data_sorted.groupby('country')['New_deaths'].cumsum()

            fig_cum_deaths, ax_cum_deaths = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data_sorted, x='date', y='Cumulative_New_Deaths', hue='country', marker='o', ax=ax_cum_deaths)
            ax_cum_deaths.set_title('Cumulative New Deaths Over Time')
            ax_cum_deaths.set_xlabel('Date')
            ax_cum_deaths.set_ylabel('Cumulative New Deaths')
            ax_cum_deaths.tick_params(axis='x', rotation=45)
            ax_cum_deaths.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
            plt.tight_layout()
            st.pyplot(fig_cum_deaths)

    with col_add_ts2:
        st.write("#### 7-Day Rolling Average of New Deaths")
        with st.spinner("Calculating rolling average deaths..."):
            filtered_data_sorted['Rolling_Avg_Deaths'] = filtered_data_sorted.groupby('country')['New_deaths'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

            fig_roll_deaths, ax_roll_deaths = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data_sorted, x='date', y='Rolling_Avg_Deaths', hue='country', marker='o', ax=ax_roll_deaths)
            ax_roll_deaths.set_title('7-Day Rolling Average of New Deaths')
            ax_roll_deaths.set_xlabel('Date')
            ax_roll_deaths.set_ylabel('Avg. New Deaths (Past 7 Days)')
            ax_roll_deaths.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_roll_deaths)

    st.markdown("---")

    # 5.3 Distributions & Relationships
    st.subheader("5.3 Key Distributions and Relationships")

    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        st.write("#### Distribution of New Deaths")
        with st.spinner("Generating distribution of new deaths..."):
            fig_hist_deaths, ax_hist_deaths = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_data['New_deaths'], kde=True, bins=min(30, len(filtered_data)), color='salmon', ax=ax_hist_deaths)
            ax_hist_deaths.set_title('Distribution of New Deaths')
            ax_hist_deaths.set_xlabel('New Deaths')
            ax_hist_deaths.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig_hist_deaths)

    with col_dist2:
        st.write("#### Distribution of Vaccination Coverage (People Fully Vaccinated / Population)")
        with st.spinner("Generating vaccination coverage distribution..."):
            fig_hist_coverage, ax_hist_coverage = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_data['vaccination_coverage'], kde=True, bins=min(30, len(filtered_data)), color='lightgreen', ax=ax_hist_coverage)
            ax_hist_coverage.set_title('Distribution of Vaccination Coverage')
            ax_hist_coverage.set_xlabel('Vaccination Coverage')
            ax_hist_coverage.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig_hist_coverage)

    st.write("#### Correlation Heatmap of Numerical Features")
    with st.spinner("Generating correlation heatmap..."):
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

    st.subheader("5.4 Interactive Scatter Plot: Explore Relationships")
    st.write("Select X and Y axes to visualize the relationship between different numerical metrics.")
    
    numerical_cols = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                      'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start']
    
    scatter_x = st.selectbox("Select X-axis:", options=numerical_cols, index=numerical_cols.index('vaccination_coverage'))
    scatter_y = st.selectbox("Select Y-axis:", options=numerical_cols, index=numerical_cols.index('New_deaths'))
    scatter_hue = st.selectbox("Color by (Optional):", options=['None', 'country'])
    
    # Option to add a regression line
    add_regression_line = st.checkbox("Add Regression Line", value=False)

    with st.spinner("Generating scatter plot..."):
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        
        if scatter_hue == 'country':
            sns.scatterplot(data=filtered_data, x=scatter_x, y=scatter_y, hue='country', ax=ax_scatter, s=100, alpha=0.7)
            if add_regression_line:
                # Plot regression line for each country if hue is country
                for country in filtered_data['country'].unique():
                    sns.regplot(data=filtered_data[filtered_data['country'] == country], x=scatter_x, y=scatter_y, ax=ax_scatter, scatter=False, line_kws={'linestyle':'--', 'alpha':0.6})
            ax_scatter.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(data=filtered_data, x=scatter_x, y=scatter_y, ax=ax_scatter, s=100, alpha=0.7)
            if add_regression_line:
                sns.regplot(data=filtered_data, x=scatter_x, y=scatter_y, ax=ax_scatter, scatter=False, color='red', line_kws={'alpha':0.8})
        
        ax_scatter.set_title(f'Scatter Plot: {scatter_y.replace("_", " ").title()} vs. {scatter_x.replace("_", " ").title()}')
        ax_scatter.set_xlabel(scatter_x.replace('_', ' ').title())
        ax_scatter.set_ylabel(scatter_y.replace('_', ' ').title())
        ax_scatter.ticklabel_format(style='plain', axis='both')
        plt.tight_layout()
        st.pyplot(fig_scatter)
    
    st.markdown("---")

    if len(selected_countries) > 1: # Only show box plots if multiple countries are selected
        st.subheader("5.5 Comparative Distributions Across Countries")
        st.write("Compare the spread and central tendency of key metrics for selected countries.")
        
        box_plot_metric = st.selectbox("Select Metric for Comparative Plot:", options=['New_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'population', 'ratio', 'vaccination_coverage'])
        
        with st.spinner(f"Generating comparative plot for {box_plot_metric.replace('_', ' ').title()}..."):
            # Use violin plot for richer distribution insights, or boxplot if preferred
            fig_comp, ax_comp = plt.subplots(figsize=(10, max(6, len(selected_countries) * 0.5)))
            sns.violinplot(data=filtered_data, x=box_plot_metric, y='country', palette='coolwarm', ax=ax_comp) # Changed to violinplot
            ax_comp.set_title(f'Distribution of {box_plot_metric.replace("_", " ").title()} per Country')
            ax_comp.set_xlabel(box_plot_metric.replace('_', ' ').title())
            ax_comp.set_ylabel('Country')
            ax_comp.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_comp)
    else:
        st.info("Select multiple countries in the sidebar to view comparative distribution plots (e.g., Violin Plots).")
    
    st.markdown("---")


with tab3:
    st.header("🔮 Predict New COVID-19 Deaths")
    st.write("Enter values for key metrics to predict the number of New Deaths.")

    # Create input fields for the model features
    input_col1, input_col2, input_col3 = st.columns(3)

    # Initialize input values with means from the filtered data, or sensible defaults
    default_total_vacc = filtered_data['total_vaccinations'].mean() if not filtered_data.empty else 0
    default_people_vacc = filtered_data['people_vaccinated'].mean() if not filtered_data.empty else 0
    default_fully_vacc = filtered_data['people_fully_vaccinated'].mean() if not filtered_data.empty else 0
    default_population = filtered_data['population'].mean() if not filtered_data.empty else 1000000 # Avoid 0 pop
    default_ratio = filtered_data['ratio'].mean() if not filtered_data.empty else 0
    default_vacc_coverage = filtered_data['vaccination_coverage'].mean() if not filtered_data.empty else 0.1
    default_days_since_start_date = max_date_data + pd.Timedelta(days=7)

    with input_col1:
        total_vaccinations_pred = st.number_input("Total Vaccinations:",
                                                  min_value=0.0,
                                                  value=float(default_total_vacc),
                                                  step=10000.0,
                                                  format="%.0f")
        people_vaccinated_pred = st.number_input("People Vaccinated (at least one dose):",
                                                 min_value=0.0,
                                                 value=float(default_people_vacc),
                                                 step=10000.0,
                                                 format="%.0f")
        people_fully_vaccinated_pred = st.number_input("People Fully Vaccinated:",
                                                        min_value=0.0,
                                                        value=float(default_fully_vacc),
                                                        step=10000.0,
                                                        format="%.0f")
    with input_col2:
        population_pred = st.number_input("Population:",
                                          min_value=1.0,
                                          value=float(default_population),
                                          step=100000.0,
                                          format="%.0f")
        ratio_pred = st.number_input("Ratio:",
                                     min_value=0.0,
                                     value=float(default_ratio),
                                     step=0.01,
                                     format="%.4f")
        vaccination_coverage_pred = st.number_input("Vaccination Coverage (Fully Vaccinated / Population):",
                                                    min_value=0.0,
                                                    max_value=1.0,
                                                    value=float(default_vacc_coverage),
                                                    step=0.01,
                                                    format="%.2f")
    with input_col3:
        prediction_date_input = st.date_input("Select Prediction Date:",
                                        value=default_days_since_start_date.date(),
                                        min_value=covid_data['date'].min().date())
        
        prediction_datetime = datetime.datetime.combine(prediction_date_input, datetime.time.min)

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
        with st.spinner("Generating prediction..."):
            try:
                predicted_deaths_transformed = model.predict(input_for_prediction)[0]
                predicted_deaths = np.expm1(predicted_deaths_transformed)
                predicted_deaths = max(0, predicted_deaths)

                st.success(f"**Predicted New Deaths:** {int(round(predicted_deaths))} ")
                st.caption("*(Prediction is an estimate based on the model's training data. Results may vary.)*")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input values are valid numbers and model_features.pkl is correctly loaded.")

    st.markdown("---")

with tab4:
    st.header("⚙️ Model Insights and Performance Overview")
    st.write("Understand how the prediction model works and its overall accuracy.")

    # 7.1 Feature Importance
    st.subheader("7.1 Feature Importance")
    st.write("Random Forest models indicate the relative importance of each feature in making predictions.")

    with st.spinner("Calculating feature importances..."):
        feature_importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': model_features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(importance_df.set_index('Feature'))
        st.caption("A higher 'Importance' value means the feature had more influence on the model's predicted number of deaths.")

    # 7.2 Model Performance (Accuracy Metrics)
    st.subheader("7.2 Model Performance (R-squared & RMSE)")
    st.write("These metrics indicate how well the model explains the variability in New Deaths and its typical prediction error.")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    with st.spinner("Calculating model performance metrics..."):
        # Reload data to ensure clean split for metrics, consistent with training
        temp_covid_data = load_data() 
        # The load_data function now handles basic cleaning and feature engineering,
        # so we just need to ensure the data is filtered and transformed consistent with training.
        
        # Filter for valid target values before splitting for metrics
        y_full_temp = temp_covid_data['New_deaths']
        y_clean_eval_temp = y_full_temp[y_full_temp >= 0]
        y_transformed_eval_temp = np.log1p(y_clean_eval_temp)

        valid_indices_eval_temp = y_transformed_eval_temp.dropna().index
        X_full_temp = temp_covid_data.loc[valid_indices_eval_temp, model_features]
        y_filtered_transformed_eval_temp = y_transformed_eval_temp.loc[valid_indices_eval_temp]

        if np.isinf(y_filtered_transformed_eval_temp).any():
            inf_indices_eval_temp = np.isinf(y_filtered_transformed_eval_temp)
            X_full_temp = X_full_temp.loc[~inf_indices_eval_temp]
            y_filtered_transformed_eval_temp = y_filtered_transformed_eval_temp.loc[~inf_indices_eval_temp]


        _, X_test_eval, _, y_test_transformed_eval = train_test_split(
            X_full_temp, y_filtered_transformed_eval_temp, test_size=0.2, random_state=42
        )

        test_predictions_transformed = model.predict(X_test_eval)
        test_predictions = np.expm1(test_predictions_transformed)
        test_predictions[test_predictions < 0] = 0

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

        The dataset used for this project is **"COVID vaccination vs. mortality"** from Kaggle:
        [https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality](https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality)

        Context of the dataset: The COVID-19 pandemic has caused significant global mortality. This dataset was generated to
        investigate the impact of coronavirus vaccinations on coronavirus mortality, providing data on vaccination progress
        and death counts.

        This dashboard serves as a hands-on example of applying data science to real-world health data.
        """
 )
st.markdown("---")
st.write("Developed by Augustine Khumalo | Connect with me on LinkedIn!(https://www.linkedin.com/in/augustine-khumalo)")
