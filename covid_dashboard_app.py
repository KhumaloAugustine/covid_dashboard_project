# covid_dashboard_app.py
# This script creates an interactive Streamlit dashboard
# for analyzing COVID-19 vaccination and mortality data,
# and predicting new deaths or daily vaccinations using trained regression models.
# It now features a simplified tabbed interface and enhanced interactivity,
# with a stronger focus on comparative analysis for multiple selected countries,
# and improved user-friendliness across various devices.

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
    layout="wide", # Using wide layout for more space, helps with responsiveness
    initial_sidebar_state="auto" # Sidebar state adapts automatically
)

st.title("🦠 COVID-19 Vaccination & Mortality Dashboard")
st.markdown("---")

# --- Initial User Guide / Getting Started ---
st.info(
    """
    **Welcome!** This interactive dashboard allows you to explore COVID-19 vaccination and mortality data from various countries.
    
    **To get started:**
    1.  Use the **filters on the left sidebar** to select specific countries and a date range relevant to your analysis.
    2.  Navigate through the **tabs below** to view detailed data overviews, interactive trends and insights, prediction models, and information about the models used.
    
    This dashboard is designed to be user-friendly and responsive across different devices, including mobile phones, tablets, and laptops.
    """
)
st.markdown("---")


# --- 2. Load Data and Pre-trained Models ---
@st.cache_data # Cache data to prevent reloading on every rerun, improving performance
def load_data():
    """Loads the COVID-19 dataset and performs initial preprocessing."""
    with st.spinner("Loading and preparing data... This may take a moment."):
        try:
            data = pd.read_csv('covid_vaccination_mortality.csv', index_col=0)
            data['date'] = pd.to_datetime(data['date'])

            # Fill NaNs with 0 for relevant numerical columns
            # This ensures calculations are not impacted by missing data, assuming 0 for missing counts.
            numerical_cols_to_fill_zero = [
                'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'New_deaths', 'ratio'
            ]
            for col in numerical_cols_to_fill_zero:
                data[col] = data[col].fillna(0)

            # Ensure population is numeric and handle potential zeros/NaNs
            # Rows with missing or zero population are crucial for per-capita calculations and are dropped.
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
            st.error("Error: 'covid_vaccination_mortality.csv' not found. Please ensure it's in the same directory as the script.")
            st.stop() # Stop the app if data is not found

covid_data = load_data()

# Load the trained models and their features
@st.cache_resource # Cache models to avoid reloading on every rerun, crucial for performance
def load_models():
    models = {}
    features = {}
    with st.spinner("Loading machine learning models..."):
        try:
            models['deaths'] = joblib.load('trained_deaths_model.pkl')
            features['deaths'] = joblib.load('model_features_deaths.pkl')
            st.sidebar.success("Deaths prediction model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error("Error: 'trained_deaths_model.pkl' or 'model_features_deaths.pkl' not found. Please run 'train_covid_model.py' first to generate these files.")
            st.stop()
        
        try:
            models['vaccinations'] = joblib.load('trained_vaccinations_model.pkl')
            features['vaccinations'] = joblib.load('model_features_vaccinations.pkl')
            st.sidebar.success("Daily vaccinations prediction model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.warning("Warning: 'trained_vaccinations_model.pkl' or 'model_features_vaccinations.pkl' not found. Daily vaccinations prediction will not be available. Please run 'train_covid_model.py'.")
            models['vaccinations'] = None # Set to None if not found
            features['vaccinations'] = None
    return models, features

models, model_features_dict = load_models()

# --- 4. Interactive Data Filtering (Sidebar) ---
st.sidebar.header("📊 Global Data Filters")
st.sidebar.write("Use these filters to customize the data displayed in the main sections of the dashboard.")

# Country selection multiselect
all_countries = covid_data['country'].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select Country(ies):",
    options=all_countries,
    default=['United States', 'India', 'Brazil'] if 'United States' in all_countries else all_countries[:3] # Sensible defaults
)

# Date Range Slider
min_date_data = covid_data['date'].min().to_pydatetime()
max_date_data = covid_data['date'].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date_data,
    max_value=max_date_data,
    value=(min_date_data, max_date_data), # Default to full range
    format="YYYY-MM-DD",
    help="Drag the ends of the slider to select a specific period, or click to adjust."
)

# Apply filters to the data
filtered_data = covid_data[
    covid_data['country'].isin(selected_countries) &
    (covid_data['date'] >= date_range[0]) &
    (covid_data['date'] <= date_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning, important for modifying filtered_data later

# Check if filtered data is empty and stop execution if so
if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your country and date range selections in the sidebar.")
    st.stop()

# Reset Filters Button
if st.sidebar.button("Reset Filters"):
    st.session_state.clear() # Clear all Streamlit session state
    st.experimental_rerun() # Rerun the app to apply default filters and refresh all components

# Define default_days_since_start_date for prediction tab, accessible globally
default_days_since_start_date_for_input = (max_date_data + pd.Timedelta(days=7)).date()


# --- Main Content Tabs ---
# Using tabs helps organize content and keeps the interface clean.
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Trends & Insights", "🔮 Prediction", "⚙️ Model & About"])

with tab1:
    st.header("📊 Data Overview & Key Statistics")
    st.write(f"Explore the raw data and key aggregated statistics for **{', '.join(selected_countries)}** from **{date_range[0].strftime('%Y-%m-%d')}** to **{date_range[1].strftime('%Y-%m-%d')}**.")

    st.subheader("Summary Metrics for Selected Data")
    # Using multiple columns for metrics to save space and improve layout on wider screens
    col_metrics1, col_metrics2, col_metrics3, col_metrics4, col_metrics5, col_metrics6 = st.columns(6) 
    with col_metrics1:
        st.metric(label="Total New Deaths",
                  value=f"{int(filtered_data['New_deaths'].sum()):,}",
                  help="Sum of new deaths recorded for the selected countries over the chosen period.")
    with col_metrics2:
        st.metric(label="Max Total Vaccinations",
                  value=f"{int(filtered_data['total_vaccinations'].max()):,}",
                  help="Highest recorded total vaccinations across selected countries in the period.")
    with col_metrics3:
        st.metric(label="Average Vaccination Coverage",
                  value=f"{filtered_data['vaccination_coverage'].mean():.2%}",
                  help="Average of 'people fully vaccinated' divided by 'population' for all data points in the selection.")
    with col_metrics4:
        st.metric(label="Selected Countries",
                  value=f"{len(selected_countries)}",
                  help="Number of unique countries currently selected in the sidebar filter.")
    with col_metrics5:
        st.metric(label="Overall Avg Daily Deaths",
                  value=f"{filtered_data['New_deaths'].mean():,.2f}",
                  help="Average daily new deaths across all selected countries and dates.")
    with col_metrics6:
        st.metric(label="Overall Avg Deaths per Million",
                  value=f"{filtered_data['new_deaths_per_million'].mean():,.2f}",
                  help="Average daily new deaths per million people.")
    
    # Additional key metrics for a more comprehensive overview
    col_metrics_new1, col_metrics_new2, col_metrics_new3, col_metrics_new4 = st.columns(4) 
    with col_metrics_new1:
        st.metric(label="Overall Max Daily Deaths",
                  value=f"{int(filtered_data['New_deaths'].max()):,}",
                  help="Highest number of new deaths recorded on any single day within the selected data.")
    with col_metrics_new2:
        st.metric(label="Overall Max Daily Vaccinations",
                  value=f"{int(filtered_data['daily_vaccinations'].max()):,}",
                  help="Highest number of daily vaccinations recorded on any single day within the selected data.") 
    with col_metrics_new3:
        st.metric(label="Overall Average Daily Vaccinations",
                  value=f"{filtered_data['daily_vaccinations'].mean():,.0f}",
                  help="Average daily vaccinations across all selected countries and dates.") 
    with col_metrics_new4:
        st.metric(label="Overall Avg Daily Vaccinated per Million",
                  value=f"{filtered_data['daily_vaccinated_per_million'].mean():,.2f}",
                  help="Average daily vaccinations per million people.")

    st.markdown("---")

    st.subheader("Raw Filtered Data Sample")
    st.write("A glimpse into the raw data based on your current filters.")
    with st.spinner("Loading raw data sample..."):
        st.dataframe(filtered_data.head(10)) # Display slightly more rows by default
        st.caption(f"Displaying the first 10 rows of {len(filtered_data)} entries. This sample reflects **all active filters** (country and date range).")

    st.subheader("Summary Statistics for Numerical Data")
    st.write("Descriptive statistics (count, mean, std, min, max, quartiles) for all numerical columns in your filtered dataset.")
    with st.spinner("Calculating summary statistics..."):
        st.dataframe(filtered_data.describe().transpose()) # Transpose for better readability
        st.caption("These statistics provide a quick overview of the central tendency, dispersion, and shape of the numerical features.")

    # --- Country Comparison Table (only shown if multiple countries are selected) ---
    if len(selected_countries) > 1:
        st.subheader("Country Comparison: Key Aggregated Metrics")
        st.write("Compare aggregated metrics across your selected countries for the chosen date range.")
        with st.spinner("Generating country comparison table..."):
            comparison_table = filtered_data.groupby('country').agg(
                Total_New_Deaths=('New_deaths', 'sum'),
                Max_Total_Vaccinations=('total_vaccinations', 'max'),
                Avg_Daily_New_Deaths=('New_deaths', 'mean'),
                Avg_Vaccination_Coverage=('vaccination_coverage', 'mean'),
                Max_Population=('population', 'max'),
                Avg_New_Deaths_Per_Million=('new_deaths_per_million', 'mean'), 
                Max_Total_Vaccinations_Per_Hundred=('total_vaccinations_per_hundred', 'max'),
                Avg_Daily_Vaccinations=('daily_vaccinations', 'mean'), 
                Avg_Daily_Vaccinated_Per_Million=('daily_vaccinated_per_million', 'mean') 
            ).reset_index()
            # Format columns for better readability with commas and percentages
            comparison_table['Total_New_Deaths'] = comparison_table['Total_New_Deaths'].apply(lambda x: f"{int(x):,}")
            comparison_table['Max_Total_Vaccinations'] = comparison_table['Max_Total_Vaccinations'].apply(lambda x: f"{int(x):,}")
            comparison_table['Avg_Daily_New_Deaths'] = comparison_table['Avg_Daily_New_Deaths'].apply(lambda x: f"{x:.2f}")
            comparison_table['Avg_Vaccination_Coverage'] = comparison_table['Avg_Vaccination_Coverage'].apply(lambda x: f"{x:.2%}")
            comparison_table['Max_Population'] = comparison_table['Max_Population'].apply(lambda x: f"{int(x):,}")
            comparison_table['Avg_New_Deaths_Per_Million'] = comparison_table['Avg_New_Deaths_Per_Million'].apply(lambda x: f"{x:.2f}")
            comparison_table['Max_Total_Vaccinations_Per_Hundred'] = comparison_table['Max_Total_Vaccinations_Per_Hundred'].apply(lambda x: f"{x:.2f}")
            comparison_table['Avg_Daily_Vaccinations'] = comparison_table['Avg_Daily_Vaccinations'].apply(lambda x: f"{int(x):,}") 
            comparison_table['Avg_Daily_Vaccinated_Per_Million'] = comparison_table['Avg_Daily_Vaccinated_Per_Million'].apply(lambda x: f"{x:.2f}") 
            
            st.dataframe(comparison_table.set_index('country'))
            st.caption("Aggregated statistics for each selected country over the chosen period. 'Max' values represent the highest recorded within the period.")
    else:
        st.info("Select multiple countries in the sidebar to view a comparative summary table and plots.")

    st.markdown("---")

    st.subheader("Key Statistics per Country (Latest Data Point)")
    st.write("View the most recent available data points for key metrics for each selected country within your chosen date range.")
    with st.spinner("Calculating latest country statistics..."):
        # Get the latest data point for each country within the filtered range
        filtered_data_sorted_latest = filtered_data.sort_values(by=['country', 'date'])

        latest_country_stats = filtered_data_sorted_latest.groupby('country').agg(
            Latest_Population=('population', 'last'), 
            Latest_Total_Vaccinations=('total_vaccinations', 'last'), 
            Latest_People_Fully_Vaccinated=('people_fully_vaccinated', 'last'), 
            Latest_Vaccination_Coverage=('vaccination_coverage', 'last'), 
            Latest_New_Deaths=('New_deaths', 'last'), 
            Latest_New_Deaths_Per_Million=('new_deaths_per_million', 'last'), 
            Latest_Total_Vaccinations_Per_Hundred=('total_vaccinations_per_hundred', 'last'),
            Latest_Daily_Vaccinations=('daily_vaccinations', 'last'), 
            Latest_Daily_Vaccinated_Per_Million=('daily_vaccinated_per_million', 'last') 
        ).reset_index()


        st.dataframe(latest_country_stats.set_index('country'))
        st.caption("Shows the latest recorded values for relevant metrics within the selected date range for each country. If a country had no data in the range, it will not appear.")

    st.subheader("Total Deaths and Vaccinations for Selected Period (Bar Plots)")
    st.write("Visualizations showing total new deaths, highest total vaccinations reached, and average daily vaccinations for each selected country over the chosen period.")
    with st.spinner("Generating bar plots for totals..."):
        period_summary = filtered_data.groupby('country').agg(
            Total_New_Deaths=('New_deaths', 'sum'),
            Highest_Total_Vaccinations_Reached=('total_vaccinations', 'max'),
            Latest_Vaccination_Coverage=('vaccination_coverage', 'max'),
            Avg_New_Deaths_Per_Million=('new_deaths_per_million', 'mean'), 
            Avg_Daily_Vaccinations=('daily_vaccinations', 'mean') 
        ).reset_index()
        
        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            fig_total_deaths, ax_total_deaths = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5))) # Dynamic height
            sns.barplot(data=period_summary.sort_values('Total_New_Deaths', ascending=False), x='Total_New_Deaths', y='country', palette='viridis', ax=ax_total_deaths)
            ax_total_deaths.set_title('Total New Deaths per Country (Selected Period)')
            ax_total_deaths.set_xlabel('Total New Deaths')
            ax_total_deaths.set_ylabel('Country')
            ax_total_deaths.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_total_deaths)
        
        with col_bar2:
            fig_total_vacc, ax_total_vacc = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5))) # Dynamic height
            sns.barplot(data=period_summary.sort_values('Highest_Total_Vaccinations_Reached', ascending=False), x='Highest_Total_Vaccinations_Reached', y='country', palette='cividis', ax=ax_total_vacc)
            ax_total_vacc.set_title('Highest Total Vaccinations Reached per Country')
            ax_total_vacc.set_xlabel('Highest Total Vaccinations')
            ax_total_vacc.set_ylabel('Country')
            ax_total_vacc.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_total_vacc)
        
        # New: Bar plot for Latest Vaccination Coverage
        st.subheader("Latest Vaccination Coverage per Country")
        fig_vacc_coverage_bar, ax_vacc_coverage_bar = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5)))
        sns.barplot(data=period_summary.sort_values('Latest_Vaccination_Coverage', ascending=False), x='Latest_Vaccination_Coverage', y='country', palette='rocket', ax=ax_vacc_coverage_bar)
        ax_vacc_coverage_bar.set_title('Latest Vaccination Coverage per Country (Selected Period)')
        ax_vacc_coverage_bar.set_xlabel('Latest Vaccination Coverage (%)')
        ax_vacc_coverage_bar.set_ylabel('Country')
        ax_vacc_coverage_bar.ticklabel_format(style='plain', axis='x')
        ax_vacc_coverage_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}')) # Format x-axis as percentage
        plt.tight_layout()
        st.pyplot(fig_vacc_coverage_bar)

        # New: Bar plot for Average New Deaths per Million
        st.subheader("Average Daily New Deaths per Million per Country")
        fig_deaths_per_million_bar, ax_deaths_per_million_bar = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5)))
        sns.barplot(data=period_summary.sort_values('Avg_New_Deaths_Per_Million', ascending=False), x='Avg_New_Deaths_Per_Million', y='country', palette='plasma', ax=ax_deaths_per_million_bar)
        ax_deaths_per_million_bar.set_title('Average Daily New Deaths per Million per Country (Selected Period)')
        ax_deaths_per_million_bar.set_xlabel('Average New Deaths per Million')
        ax_deaths_per_million_bar.set_ylabel('Country')
        ax_deaths_per_million_bar.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        st.pyplot(fig_deaths_per_million_bar)
        
        # New: Bar plot for Average Daily Vaccinations
        st.subheader("Average Daily Vaccinations per Country")
        fig_avg_daily_vacc, ax_avg_daily_vacc = plt.subplots(figsize=(10, max(6, len(period_summary) * 0.5)))
        sns.barplot(data=period_summary.sort_values('Avg_Daily_Vaccinations', ascending=False), x='Avg_Daily_Vaccinations', y='country', palette='cubehelix', ax=ax_avg_daily_vacc)
        ax_avg_daily_vacc.set_title('Average Daily Vaccinations per Country (Selected Period)')
        ax_avg_daily_vacc.set_xlabel('Average Daily Vaccinations')
        ax_avg_daily_vacc.set_ylabel('Country')
        ax_avg_daily_vacc.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        st.pyplot(fig_avg_daily_vacc)


    st.markdown("---")

    st.subheader("Top/Bottom Countries by Average Daily Deaths")
    st.write(f"Easily identify countries with the highest and lowest average daily new deaths within your selected timeframe.")
    top_n_countries = st.slider("Select N for Top/Bottom Countries:", min_value=1, max_value=min(10, len(all_countries)), value=5, key='top_n_countries_tab1', help="Adjust 'N' to see more or fewer top/bottom countries.") 
    with st.spinner(f"Calculating top/bottom {top_n_countries} countries..."):
        avg_daily_deaths = filtered_data.groupby('country')['New_deaths'].mean().sort_values(ascending=False)
        
        col_top_bottom1, col_top_bottom2 = st.columns(2)
        with col_top_bottom1:
            st.write(f"#### Top {top_n_countries} Countries by Average Daily New Deaths")
            st.dataframe(avg_daily_deaths.head(top_n_countries).reset_index().rename(columns={'New_deaths': 'Avg. Daily Deaths'}).set_index('country'))
        with col_top_bottom2:
            st.write(f"#### Bottom {top_n_countries} Countries by Average Daily New Deaths")
            st.dataframe(avg_daily_deaths.tail(top_n_countries).reset_index().rename(columns={'New_deaths': 'Avg. Daily Deaths'}).set_index('country'))

    st.markdown("---")
    
    st.subheader("Top/Bottom Countries by Total Vaccinations") 
    st.write(f"See which countries have administered the most or fewest total vaccinations within your selected timeframe.")
    top_n_vacc_countries = st.slider("Select N for Top/Bottom Vaccinated Countries:", min_value=1, max_value=min(10, len(all_countries)), value=5, key='top_n_vacc_countries_tab1', help="Adjust 'N' to see more or fewer top/bottom vaccinated countries.")
    with st.spinner(f"Calculating top/bottom {top_n_vacc_countries} vaccinated countries..."):
        total_vacc_by_country = filtered_data.groupby('country')['total_vaccinations'].max().sort_values(ascending=False)
        
        col_top_bottom_vacc1, col_top_bottom_vacc2 = st.columns(2)
        with col_top_bottom_vacc1:
            st.write(f"#### Top {top_n_vacc_countries} Countries by Max Total Vaccinations")
            st.dataframe(total_vacc_by_country.head(top_n_vacc_countries).reset_index().rename(columns={'total_vaccinations': 'Max Total Vaccinations'}).set_index('country'))
        with col_top_bottom_vacc2:
            st.write(f"#### Bottom {top_n_vacc_countries} Countries by Max Total Vaccinations")
            st.dataframe(total_vacc_by_country.tail(top_n_vacc_countries).reset_index().rename(columns={'total_vaccinations': 'Max Total Vaccinations'}).set_index('country'))

    st.markdown("---")

    st.subheader("Data Dictionary / Column Explanations")
    st.write("Understand the meaning of each column in the dataset to better interpret the dashboard's insights.")
    st.markdown(
        """
        * **country:** The geographical entity to which the data corresponds.
        * **iso_code:** International Organization for Standardization 3166-1 alpha-3 code for each country.
        * **date:** The specific date for which the data entry is recorded.
        * **total_vaccinations:** The cumulative number of COVID-19 vaccine doses administered up to that date in the country.
        * **people_vaccinated:** The cumulative number of individuals who have received at least one dose of a COVID-19 vaccine.
        * **people_fully_vaccinated:** The cumulative number of individuals who have completed their full vaccination course.
        * **New_deaths:** The number of new deaths attributed to COVID-19 reported on that specific day.
        * **population:** The estimated population of the country in 2021, used for per-capita calculations.
        * **ratio:** Calculated as the percentage of the population that has received at least one vaccination dose: `(people_vaccinated / population) * 100`.
        * **vaccination_coverage:** Calculated as the proportion of the population that is fully vaccinated: `(people_fully_vaccinated / population)`. This is a value between 0 and 1.
        * **new_deaths_per_million:** The number of new deaths per 1,000,000 people: `(New_deaths / population) * 1,000,000`. This normalizes death counts by population size.
        * **total_vaccinations_per_hundred:** Total vaccinations administered per 100 people: `(total_vaccinations / population) * 100`.
        * **daily_vaccinations:** The number of new vaccine doses administered on a given day. This is derived from the daily change in `total_vaccinations`.
        * **daily_vaccinated_per_million:** The number of new vaccine doses administered per 1,000,000 people on a given day.
        * **daily_deaths_growth_rate:** The daily percentage change in the number of new deaths. Helps understand the acceleration or deceleration of deaths.
        * **daily_vaccinations_growth_rate:** The daily percentage change in the number of daily vaccinations. Helps understand the pace of vaccination efforts.
        * **days_since_start:** The number of days that have passed since the earliest date in the entire dataset, used for time-series modeling.
        """
    )
    st.markdown("---")


with tab2:
    st.header("📈 Dynamic COVID-19 Trends & Insights")
    st.write("Visualize how key COVID-19 metrics have evolved over time and explore relationships between them. All plots automatically update based on your country and date selections from the 'Data Overview' tab.")

    # 5.1 Time Series Plots - Customizable Metric and Plot Type
    st.subheader("5.1 Customizable Time Series Trends")
    st.write("Select a metric and a plot type to see its trend over your chosen date range. This helps identify peaks, valleys, and overall patterns.")
    col_ts_sel1, col_ts_sel2 = st.columns(2)
    with col_ts_sel1:
        time_series_metric = st.selectbox(
            "Select Metric to Plot Over Time:",
            options=['New_deaths', 'new_deaths_per_million', 'total_vaccinations', 'total_vaccinations_per_hundred', 'daily_vaccinations', 'daily_vaccinated_per_million', 'people_vaccinated', 'people_fully_vaccinated', 'vaccination_coverage'],
            index=0, # Default to New Deaths
            key='ts_metric_select',
            help="Choose a metric to display its historical trend for the selected countries."
        )
    with col_ts_sel2:
        plot_type = st.radio("Select Plot Type:", ('Line Plot', 'Area Plot', 'Bar Plot'), key='ts_plot_type', help="Line plots show continuous trends, Area plots highlight magnitude, Bar plots show discrete values.")
    
    with st.spinner(f"Generating time series plot for {time_series_metric.replace('_', ' ').title()}..."):
        fig_ts, ax_ts = plt.subplots(figsize=(12, 6)) # Fixed figure size for consistency

        if plot_type == 'Line Plot':
            sns.lineplot(data=filtered_data, x='date', y=time_series_metric, hue='country', marker='o', ax=ax_ts)
        elif plot_type == 'Area Plot':
            sns.lineplot(data=filtered_data, x='date', y=time_series_metric, hue='country', ax=ax_ts)
            for country in filtered_data['country'].unique():
                country_data = filtered_data[filtered_data['country'] == country].sort_values('date')
                ax_ts.fill_between(country_data['date'], country_data[time_series_metric], alpha=0.3)
        else: # Bar Plot
            sns.barplot(data=filtered_data, x='date', y=time_series_metric, hue='country', dodge=True, ax=ax_ts)
            # Dynamically adjust x-tick frequency for readability on bar plots with many dates
            ax_ts.set_xticks(ax_ts.get_xticks()[::max(1, len(filtered_data['date'].unique()) // 10)]) 

        ax_ts.set_title(f'{time_series_metric.replace("_", " ").title()} Over Time')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel(time_series_metric.replace('_', ' ').title())
        ax_ts.tick_params(axis='x', rotation=45) # Rotate date labels for better readability
        ax_ts.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on Y-axis
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        st.pyplot(fig_ts)

    st.markdown("---")

    # --- Additional Time-Series Analysis (Cumulative & Rolling Averages) ---
    st.subheader("5.2 Additional Time-Series Analysis: Cumulative & Rolling Averages")
    st.write("These plots provide a 'survival-like' perspective by showing cumulative totals and smoothed trends, which can help in understanding long-term impacts and underlying trends by reducing noise.")

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
            ax_cum_deaths.ticklabel_format(style='plain', axis='y') 
            plt.tight_layout()
            st.pyplot(fig_cum_deaths)

    with col_add_ts2:
        st.write("#### 7-Day Rolling Average of New Deaths")
        with st.spinner("Calculating rolling average deaths..."):
            # Rolling average helps smooth out daily fluctuations and reveal underlying trends.
            filtered_data_sorted['Rolling_Avg_Deaths'] = filtered_data_sorted.groupby('country')['New_deaths'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

            fig_roll_deaths, ax_roll_deaths = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data_sorted, x='date', y='Rolling_Avg_Deaths', hue='country', marker='o', ax=ax_roll_deaths)
            ax_roll_deaths.set_title('7-Day Rolling Average of New Deaths')
            ax_roll_deaths.set_xlabel('Date')
            ax_roll_deaths.set_ylabel('Avg. New Deaths (Past 7 Days)')
            ax_roll_deaths.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_roll_deaths)
    
    col_add_ts3, col_add_ts4 = st.columns(2)
    with col_add_ts3:
        st.write("#### Cumulative Daily Vaccinations Over Time")
        with st.spinner("Calculating cumulative vaccinations..."):
            filtered_data_sorted['Cumulative_Daily_Vaccinations'] = filtered_data_sorted.groupby('country')['daily_vaccinations'].cumsum()

            fig_cum_vacc, ax_cum_vacc = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data_sorted, x='date', y='Cumulative_Daily_Vaccinations', hue='country', marker='o', ax=ax_cum_vacc)
            ax_cum_vacc.set_title('Cumulative Daily Vaccinations Over Time')
            ax_cum_vacc.set_xlabel('Date')
            ax_cum_vacc.set_ylabel('Cumulative Daily Vaccinations')
            ax_cum_vacc.tick_params(axis='x', rotation=45)
            ax_cum_vacc.ticklabel_format(style='plain', axis='y')
            plt.tight_layout()
            st.pyplot(fig_cum_vacc)

    with col_add_ts4:
        st.write("#### 7-Day Rolling Average of Daily Vaccinations")
        with st.spinner("Calculating rolling average vaccinations..."):
            filtered_data_sorted['Rolling_Avg_Daily_Vaccinations'] = filtered_data_sorted.groupby('country')['daily_vaccinations'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

            fig_roll_vacc, ax_roll_vacc = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=filtered_data_sorted, x='date', y='Rolling_Avg_Daily_Vaccinations', hue='country', marker='o', ax=ax_roll_vacc)
            ax_roll_vacc.set_title('7-Day Rolling Average of Daily Vaccinations')
            ax_roll_vacc.set_xlabel('Date')
            ax_roll_vacc.set_ylabel('Avg. Daily Vaccinations (Past 7 Days)')
            ax_roll_vacc.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_roll_vacc)

    st.markdown("---")

    # 5.3 Distributions & Relationships
    st.subheader("5.3 Key Distributions and Relationships")
    st.write("Understand the spread and shape of your data, and how different numerical features correlate with each other.")

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
    
    with st.spinner("Generating correlation heatmap..."):
        st.write("#### Correlation Heatmap of Numerical Features")
        st.write("A correlation heatmap shows how strongly pairs of numerical variables are related. Values closer to 1 or -1 indicate a stronger relationship (positive or negative).")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        # Select only relevant numerical columns for correlation to avoid overcrowding the map
        numerical_data_for_corr = filtered_data[
            ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
             'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start',
             'new_deaths_per_million', 'total_vaccinations_per_hundred',
             'daily_vaccinations', 'daily_vaccinated_per_million',
             'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'] # Added new growth rate metrics
        ].corr()
        sns.heatmap(numerical_data_for_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title('Correlation Matrix of Key Metrics')
        plt.tight_layout()
        st.pyplot(fig_corr)

    st.markdown("---")

    st.subheader("5.4 Interactive Scatter Plot: Explore Relationships")
    st.write("Select any two numerical metrics to visualize their relationship using a scatter plot. A regression line can be added to show the general trend.")
    
    # Define all numerical columns available for scatter plot
    numerical_cols = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                      'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start',
                      'new_deaths_per_million', 'total_vaccinations_per_hundred',
                      'daily_vaccinations', 'daily_vaccinated_per_million',
                      'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'] 
    
    scatter_x = st.selectbox("Select X-axis for Scatter Plot:", 
                             options=numerical_cols, 
                             index=numerical_cols.index('vaccination_coverage'), 
                             key='scatter_x',
                             help="Choose the metric for the horizontal axis.") 
    scatter_y = st.selectbox("Select Y-axis for Scatter Plot:", 
                             options=numerical_cols, 
                             index=numerical_cols.index('New_deaths'), 
                             key='scatter_y',
                             help="Choose the metric for the vertical axis.") 
    scatter_hue = st.selectbox("Color points by (Optional):", 
                               options=['None', 'country'], 
                               key='scatter_hue',
                               help="Color points by country to distinguish trends per nation.") 
    
    add_regression_line = st.checkbox("Add Regression Line to Scatter Plot", value=False, key='scatter_reg_line', help="Adds a best-fit line to show the general relationship.") 

    with st.spinner("Generating scatter plot..."):
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        
        if scatter_hue == 'country':
            sns.scatterplot(data=filtered_data, x=scatter_x, y=scatter_y, hue='country', ax=ax_scatter, s=100, alpha=0.7)
            if add_regression_line:
                for country_val in filtered_data['country'].unique(): 
                    sns.regplot(data=filtered_data[filtered_data['country'] == country_val], x=scatter_x, y=scatter_y, ax=ax_scatter, scatter=False, line_kws={'linestyle':'--', 'alpha':0.6})
            ax_scatter.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside to avoid overlap
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

    if len(selected_countries) > 1: # Only show comparative plots if multiple countries are selected
        st.subheader("5.5 Comparative Distributions Across Countries")
        st.write("Compare the distribution (spread and central tendency) of a selected metric across multiple countries using violin plots. This shows not just averages but also density and range of values.")
        
        box_plot_metric = st.selectbox("Select Metric for Comparative Plot:", 
                                       options=['New_deaths', 'new_deaths_per_million', 'total_vaccinations', 'total_vaccinations_per_hundred', 'people_vaccinated', 'people_fully_vaccinated', 'population', 'ratio', 'vaccination_coverage', 'daily_vaccinations', 'daily_vaccinated_per_million', 'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'], 
                                       key='comp_plot_metric',
                                       help="Choose a numerical metric to compare its distribution across selected countries.") 
        
        with st.spinner(f"Generating comparative plot for {box_plot_metric.replace('_', ' ').title()}..."):
            fig_comp, ax_comp = plt.subplots(figsize=(10, max(6, len(selected_countries) * 0.5))) # Dynamic height
            sns.violinplot(data=filtered_data, x=box_plot_metric, y='country', palette='coolwarm', ax=ax_comp) 
            ax_comp.set_title(f'Distribution of {box_plot_metric.replace("_", " ").title()} per Country')
            ax_comp.set_xlabel(box_plot_metric.replace('_', ' ').title())
            ax_comp.set_ylabel('Country')
            ax_comp.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_comp)
    else:
        st.info("Select multiple countries in the sidebar to view comparative distribution plots (e.g., Violin Plots).")
    
    st.markdown("---")

    st.subheader("5.6 Daily Change in Metrics")
    st.write("Visualize the day-to-day increase or decrease in cumulative metrics like total vaccinations, or the direct daily counts for new deaths and daily vaccinations.")

    daily_change_metric_options = ['New_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'daily_vaccinations'] 
    daily_change_metric = st.selectbox(
        "Select Metric to show Daily Change:",
        options=daily_change_metric_options,
        index=0, 
        key='daily_change_metric',
        help="Choose a metric to see its daily change trend. 'New_deaths' and 'daily_vaccinations' are already daily counts."
    )

    with st.spinner(f"Calculating daily change for {daily_change_metric.replace('_', ' ').title()}..."):
        filtered_data_sorted_for_diff = filtered_data.sort_values(by=['country', 'date'])
        
        data_to_plot_col = daily_change_metric
        y_label_diff = daily_change_metric.replace("_", " ").title()
        
        if daily_change_metric not in ['New_deaths', 'daily_vaccinations']:
            # For cumulative metrics, calculate the actual daily change
            daily_data_to_plot_temp = filtered_data_sorted_for_diff.copy()
            daily_data_to_plot_temp[f'{daily_change_metric}_calculated_daily_change'] = daily_data_to_plot_temp.groupby('country')[daily_change_metric].diff().fillna(0)
            data_to_plot_col = f'{daily_change_metric}_calculated_daily_change'
            y_label_diff = f'Daily Change in {daily_change_metric.replace("_", " ").title()}'

        fig_daily_change, ax_daily_change = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=filtered_data_sorted_for_diff, x='date', y=data_to_plot_col, hue='country', ax=ax_daily_change)
        ax_daily_change.set_title(f'Daily Trends for {y_label_diff}')
        ax_daily_change.set_xlabel('Date')
        ax_daily_change.set_ylabel(y_label_diff)
        ax_daily_change.tick_params(axis='x', rotation=45)
        ax_daily_change.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig_daily_change)
    st.markdown("---")

    st.subheader("5.7 Latest Vaccination Status Distribution")
    st.write("This pie chart shows the overall approximate distribution of vaccination status (fully vaccinated, partially vaccinated, unvaccinated) for the latest available date across all selected countries.")

    with st.spinner("Calculating vaccination status distribution..."):
        latest_date_per_country = filtered_data.groupby('country')['date'].max().reset_index()
        latest_status_data = pd.merge(latest_date_per_country, filtered_data, on=['country', 'date'], how='left')

        if not latest_status_data.empty:
            total_people_vaccinated = latest_status_data['people_vaccinated'].sum()
            total_people_fully_vaccinated = latest_status_data['people_fully_vaccinated'].sum()
            total_population = latest_status_data['population'].sum()

            # Adjust if people_fully_vaccinated is somehow higher than people_vaccinated (data anomaly)
            total_people_vaccinated_adjusted = max(total_people_vaccinated, total_people_fully_vaccinated)
            
            # Calculate the segments for the pie chart, ensuring non-negative values
            unvaccinated = max(0, total_population - total_people_vaccinated_adjusted)
            fully_vaccinated = total_people_fully_vaccinated
            partially_vaccinated = max(0, total_people_vaccinated_adjusted - total_people_fully_vaccinated)

            pie_data = pd.DataFrame({
                'Status': ['Fully Vaccinated', 'Partially Vaccinated', 'Unvaccinated'],
                'Count': [fully_vaccinated, partially_vaccinated, unvaccinated]
            })
            pie_data = pie_data[pie_data['Count'] > 0] # Filter out zero counts for clearer pie chart

            if not pie_data.empty:
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                ax_pie.pie(pie_data['Count'], labels=pie_data['Status'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                ax_pie.set_title(f'Latest Vaccination Status Distribution ({date_range[1].strftime("%Y-%m-%d")})')
                ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig_pie)
            else:
                st.info("No vaccination status data to display for the selected period. This might happen if 'people_vaccinated' or 'people_fully_vaccinated' counts are consistently zero or missing for the latest dates.")
        else:
            st.info("No latest vaccination status data found for the selected countries/date range. Please check your filters.")
    st.markdown("---")

    st.subheader("5.8 Monthly and Yearly Trends")
    st.write("Aggregate data to see broader patterns and long-term trends by month or year. This can help in understanding seasonalities or major shifts.")

    aggregation_level = st.radio("Aggregate by:", ('Month', 'Year'), key='agg_level', help="Choose to view trends aggregated by month or by year.") 
    aggregation_metric = st.selectbox(
        "Select Metric for Aggregation:",
        options=['New_deaths', 'total_vaccinations', 'people_fully_vaccinated', 'new_deaths_per_million', 'total_vaccinations_per_hundred', 'daily_vaccinations', 'daily_vaccinated_per_million'],
        index=0,
        key='agg_metric',
        help="Choose the metric to aggregate and plot over months or years." 
    )

    with st.spinner(f"Generating {aggregation_level.lower()}ly trends for {aggregation_metric.replace('_', ' ').title()}..."):
        # Ensure 'date' is a datetime type before dt.to_period or dt.year
        if not pd.api.types.is_datetime64_any_dtype(filtered_data['date']):
            filtered_data['date'] = pd.to_datetime(filtered_data['date'])
            
        temp_df_agg = filtered_data.copy() # Create a copy to avoid SettingWithCopyWarning
        if aggregation_level == 'Month':
            temp_df_agg['time_period'] = temp_df_agg['date'].dt.to_period('M')
            grouped_data = temp_df_agg.groupby(['time_period', 'country'])[aggregation_metric].sum().reset_index()
            grouped_data['time_period'] = grouped_data['time_period'].astype(str) # Convert Period to string for plotting
            x_col_name = 'time_period' 
            x_label = 'Year-Month' 
        else: # Year
            temp_df_agg['time_period'] = temp_df_agg['date'].dt.year
            grouped_data = temp_df_agg.groupby(['time_period', 'country'])[aggregation_metric].sum().reset_index()
            x_col_name = 'time_period' 
            x_label = 'Year' 

        fig_agg, ax_agg = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=grouped_data, x=x_col_name, y=aggregation_metric, hue='country', marker='o', ax=ax_agg)
        ax_agg.set_title(f'{aggregation_level}ly {aggregation_metric.replace("_", " ").title()} Trends')
        ax_agg.set_xlabel(x_label)
        ax_agg.set_ylabel(f'Total {aggregation_metric.replace("_", " ").title()}')
        ax_agg.tick_params(axis='x', rotation=45)
        ax_agg.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig_agg)
    st.markdown("---")

    if len(selected_countries) > 1:
        st.subheader("5.9 Comparative Density Plots (KDE)")
        st.write("Kernel Density Estimation (KDE) plots visualize the distribution shape of a metric across selected countries. This helps in understanding data concentration and spread without binning.")
        kde_metric = st.selectbox(
            "Select Metric for Density Plot:",
            options=['New_deaths', 'new_deaths_per_million', 'vaccination_coverage', 'total_vaccinations_per_hundred', 'daily_vaccinations', 'daily_vaccinated_per_million', 'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'],
            key='kde_metric',
            help="Choose a metric to compare its density distribution across selected countries."
        )
        with st.spinner(f"Generating density plot for {kde_metric.replace('_', ' ').title()}..."):
            fig_kde, ax_kde = plt.subplots(figsize=(10, 6))
            sns.kdeplot(data=filtered_data, x=kde_metric, hue='country', fill=True, common_norm=False, ax=ax_kde)
            ax_kde.set_title(f'Density Plot of {kde_metric.replace("_", " ").title()} by Country')
            ax_kde.set_xlabel(kde_metric.replace('_', ' ').title())
            ax_kde.set_ylabel('Density')
            ax_kde.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig_kde)
    else:
        st.info("Select multiple countries in the sidebar to view comparative density plots.")
    st.markdown("---")

    st.subheader("5.10 Growth Rate Trends")
    st.write("Examine the daily percentage change in new deaths and daily vaccinations. Positive values indicate growth, negative values indicate decline.")

    growth_rate_metric = st.selectbox(
        "Select Growth Rate Metric:",
        options=['daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'],
        index=0,
        key='growth_rate_metric',
        help="View how quickly new deaths or daily vaccinations are changing percentage-wise."
    )

    with st.spinner(f"Generating growth rate trend for {growth_rate_metric.replace('_', ' ').title()}..."):
        fig_growth, ax_growth = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=filtered_data, x='date', y=growth_rate_metric, hue='country', ax=ax_growth)
        ax_growth.set_title(f'Daily {growth_rate_metric.replace("_", " ").title()} Over Time')
        ax_growth.set_xlabel('Date')
        ax_growth.set_ylabel('Percentage Change')
        ax_growth.tick_params(axis='x', rotation=45)
        ax_growth.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')) # Format as percentage
        plt.tight_layout()
        st.pyplot(fig_growth)
    st.markdown("---")

    st.subheader("5.11 Lag Plot for Autocorrelation")
    st.write("A lag plot helps visualize autocorrelation (the relationship between a variable's value and its past values). A linear pattern suggests strong autocorrelation, which is common in time series data.")

    lag_plot_metric = st.selectbox(
        "Select Metric for Lag Plot:",
        options=['New_deaths', 'daily_vaccinations', 'vaccination_coverage', 'new_deaths_per_million'],
        index=0,
        key='lag_plot_metric',
        help="Choose a metric to analyze its autocorrelation with past values."
    )
    lag_days = st.slider("Select Lag (in days):", min_value=1, max_value=30, value=7, key='lag_days', help="The number of days back to compare the current metric value against.")

    with st.spinner(f"Generating lag plot for {lag_plot_metric.replace('_', ' ').title()} with lag {lag_days} days..."):
        lag_data = filtered_data[['date', 'country', lag_plot_metric]].copy()
        
        # Sort data to ensure correct lag calculation
        lag_data = lag_data.sort_values(by=['country', 'date'])
        
        # Calculate lagged values grouped by country to prevent mixing data from different countries
        lag_data['lagged_metric'] = lag_data.groupby('country')[lag_plot_metric].shift(lag_days)
        
        # Drop rows with NaN values (where lag could not be calculated, e.g., at the beginning of the series)
        lag_data.dropna(subset=[lag_plot_metric, 'lagged_metric'], inplace=True)

        if not lag_data.empty:
            fig_lag, ax_lag = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=lag_data, x='lagged_metric', y=lag_plot_metric, hue='country', s=50, alpha=0.6, ax=ax_lag)
            ax_lag.set_title(f'Lag Plot: {lag_plot_metric.replace("_", " ").title()} (t) vs. {lag_days}-day Lagged (t-{lag_days})')
            ax_lag.set_xlabel(f'{lag_plot_metric.replace("_", " ").title()} (at t-{lag_days} days)')
            ax_lag.set_ylabel(f'{lag_plot_metric.replace("_", " ").title()} (at time t)')
            ax_lag.ticklabel_format(style='plain', axis='both')
            plt.tight_layout()
            st.pyplot(fig_lag)
        else:
            st.info("Not enough data to generate lag plot for the selected parameters. Try a smaller lag or wider date range.")
    st.markdown("---")


with tab3:
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
    if prediction_type == 'New Deaths' and models['deaths']:
        current_model = models['deaths']
        current_features = model_features_dict['deaths']
        prediction_label = "Predicted New Deaths"
    elif prediction_type == 'Daily Vaccinations' and models['vaccinations']:
        current_model = models['vaccinations']
        current_features = model_features_dict['vaccinations']
        prediction_label = "Predicted Daily Vaccinations"
    
    if current_model is None:
        st.warning(f"The model for '{prediction_type}' is not loaded. Please ensure 'train_covid_model.py' has been run successfully to generate the necessary model files (`trained_deaths_model.pkl`, `model_features_deaths.pkl`, `trained_vaccinations_model.pkl`, `model_features_vaccinations.pkl`).")
    else:
        st.write(f"Enter values for the features below to predict **{prediction_type}**: *(Default values are based on the average of your currently filtered data.)*")

        input_values = {} # Dictionary to store user inputs
        input_cols = st.columns(3) # Use columns for a cleaner layout

        # Helper function to get a sensible default value for input fields
        def get_default_value(col_name, fallback_value):
            if not filtered_data.empty and col_name in filtered_data.columns and filtered_data[col_name].sum() > 0:
                # Use mean of filtered data if available and non-zero
                return float(filtered_data[col_name].mean())
            return float(fallback_value) # Fallback to a predefined default

        # Predefined sensible defaults for features that might be needed by models
        feature_defaults = {
            'total_vaccinations': 100000.0,
            'people_vaccinated': 50000.0,
            'people_fully_vaccinated': 25000.0,
            'population': 10000000.0, # A larger default for population
            'ratio': 0.05,
            'vaccination_coverage': 0.025,
            'New_deaths': 10.0, 
        }

        # Dynamically create input fields based on the selected model's required features
        with input_cols[0]:
            for feature in ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']:
                if feature in current_features:
                    input_values[feature] = st.number_input(
                        feature.replace('_', ' ').title() + ":",
                        min_value=0.0,
                        value=get_default_value(feature, feature_defaults.get(feature, 0.0)),
                        step=10000.0, # Larger step for large numbers
                        format="%.0f", # Format as integer
                        key=f'pred_input_{feature}',
                        help=f"Enter the {feature.replace('_', ' ').lower()} value for prediction."
                    )
        with input_cols[1]:
            for feature in ['population', 'ratio', 'vaccination_coverage']:
                if feature in current_features:
                    # Adjust min_value and format based on feature type
                    min_val = 1.0 if feature == 'population' else 0.0
                    max_val = 1.0 if feature == 'vaccination_coverage' else None
                    step_val = 0.01 if feature in ['ratio', 'vaccination_coverage'] else 100000.0
                    format_str = "%.4f" if feature in ['ratio', 'vaccination_coverage'] else "%.0f"

                    input_values[feature] = st.number_input(
                        feature.replace('_', ' ').title() + ":",
                        min_value=min_val,
                        max_value=max_val,
                        value=get_default_value(feature, feature_defaults.get(feature, min_val)),
                        step=step_val,
                        format=format_str, 
                        key=f'pred_input_{feature}',
                        help=f"Enter the {feature.replace('_', ' ').lower()} value for prediction."
                    )
        with input_cols[2]:
            # Date input for 'days_since_start' feature
            prediction_date_input = st.date_input("Select Prediction Date:",
                                            value=default_days_since_start_date_for_input, # Uses global default
                                            min_value=covid_data['date'].min().date(), # Prevents dates before dataset start
                                            key='prediction_date_input_tab3',
                                            help="Choose the date for which you want to make a prediction. 'Days since start' will be calculated automatically."
                                            )
            
            prediction_datetime = datetime.datetime.combine(prediction_date_input, datetime.time.min)
            input_values['days_since_start'] = (prediction_datetime - covid_data['date'].min().to_pydatetime()).days
            st.info(f"Calculated days since start: **{input_values['days_since_start']}** days.")
            
            # This feature is only relevant for the daily_vaccinations model in this setup
            if 'New_deaths' in current_features: 
                input_values['New_deaths'] = st.number_input("New Deaths (for Daily Vaccinations Model):",
                                                             min_value=0.0,
                                                             value=get_default_value('New_deaths', feature_defaults.get('New_deaths', 0.0)),
                                                             step=100.0,
                                                             format="%.0f", 
                                                             key='pred_input_new_deaths',
                                                             help="Input for 'New Deaths' is used as a feature when predicting 'Daily Vaccinations'."
                                                            )

        # Create a DataFrame with the correct features and their order for the selected model
        # This ensures the input DataFrame matches the training features for the model.
        final_input_for_prediction = pd.DataFrame([input_values])[current_features]


        if st.button(f"Predict {prediction_type}", key='predict_button'):
            with st.spinner(f"Generating {prediction_type.lower()} prediction..."):
                try:
                    # Make prediction using the loaded model
                    predicted_value_transformed = current_model.predict(final_input_for_prediction)[0]
                    # Inverse transform the prediction (from log1p back to original scale)
                    predicted_value = np.expm1(predicted_value_transformed)
                    # Ensure prediction is non-negative and round to a whole number as deaths/vaccinations are integers
                    predicted_value = max(0, round(predicted_value)) 

                    st.success(f"**{prediction_label}:** {int(predicted_value):,} ")
                    st.caption("*(Prediction is an estimate based on the model's training data. Results may vary and are best interpreted in the context of the model's performance metrics.)*")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}. Please check your input values.")
                    st.warning("Ensure all input values are valid numbers. If the issue persists, verify that the trained model files (`.pkl` files) are correctly generated and loaded.")

    st.markdown("---")

with tab4:
    st.header("⚙️ Model Insights and Performance Overview")
    st.write("Understand how our machine learning models work, which factors they consider most important, and how accurate their predictions are.")

    model_insight_selector = st.selectbox(
        "Select Model to View Insights:",
        options=['New Deaths Model', 'Daily Vaccinations Model'],
        key='model_insight_selector',
        help="Choose which prediction model's insights you want to explore."
    )

    selected_model_name_key = 'deaths' if model_insight_selector == 'New Deaths Model' else 'vaccinations'
    selected_model = models.get(selected_model_name_key)
    selected_features = model_features_dict.get(selected_model_name_key)

    if selected_model is None:
        st.warning(f"Model for '{model_insight_selector}' is not loaded. Please run 'train_covid_model.py' to generate model files before viewing its insights.")
    else:
        # 7.1 Feature Importance
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

        # 7.2 Model Performance (Accuracy Metrics)
        st.subheader(f"7.2 Model Performance for {model_insight_selector} (R-squared & RMSE)")
        st.write("These metrics indicate how well the model performs on unseen data. They help you understand the reliability and accuracy of the predictions.")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        with st.spinner(f"Calculating {model_insight_selector} performance metrics..."):
            # Load fresh data for evaluation to ensure an unbiased performance check
            temp_covid_data_eval = load_data() 
            
            target_col_eval = 'New_deaths' if selected_model_name_key == 'deaths' else 'daily_vaccinations'
            features_list_eval = model_features_dict[selected_model_name_key]

            y_full_temp_eval = temp_covid_data_eval[target_col_eval]
            y_clean_eval_temp = y_full_temp_eval[y_full_temp_eval >= 0] # Filter out negative target values
            y_transformed_eval_temp = np.log1p(y_clean_eval_temp) # Apply log1p transformation

            # Align X and y after filtering and transformation
            valid_indices_eval_temp = y_transformed_eval_temp.dropna().index
            X_full_temp_eval = temp_covid_data_eval.loc[valid_indices_eval_temp, features_list_eval]
            y_filtered_transformed_eval_temp = y_transformed_eval_temp.loc[valid_indices_eval_temp]

            # Handle potential inf values after log1p (though less likely with >=0 filter)
            if np.isinf(y_filtered_transformed_eval_temp).any():
                inf_indices_eval_temp = np.isinf(y_filtered_transformed_eval_temp)
                X_full_temp_eval = X_full_temp_eval.loc[~inf_indices_eval_temp]
                y_filtered_transformed_eval_temp = y_filtered_transformed_eval_temp.loc[~inf_indices_eval_temp]

            # Drop rows with NaN in features for evaluation to ensure clean data for model prediction
            initial_rows_eval = len(X_full_temp_eval)
            X_full_temp_eval.dropna(inplace=True)
            y_filtered_transformed_eval_temp = y_filtered_transformed_eval_temp.loc[X_full_temp_eval.index]
            if len(X_full_temp_eval) < initial_rows_eval:
                st.warning(f"For model evaluation of {model_insight_selector}, dropped {initial_rows_eval - len(X_full_temp_eval)} rows due to missing feature values.")
            
            if X_full_temp_eval.empty:
                st.info(f"Not enough clean data to perform robust evaluation for {model_insight_selector}. Try a wider date range or more countries.")
            else:
                # Split data for evaluation (using a 80/20 train-test split, consistent with training script)
                _, X_test_eval, _, y_test_transformed_eval = train_test_split(
                    X_full_temp_eval, y_filtered_transformed_eval_temp, test_size=0.2, random_state=42
                )

                test_predictions_transformed = selected_model.predict(X_test_eval)
                test_predictions = np.expm1(test_predictions_transformed) # Inverse transform predictions
                test_predictions[test_predictions < 0] = 0 # Ensure non-negative predictions

                y_test_original_scale = temp_covid_data_eval.loc[y_test_transformed_eval.index, target_col_eval] # Get original scale target values

                # Calculate evaluation metrics
                test_mse = mean_squared_error(y_test_original_scale, test_predictions)
                test_rmse = np.sqrt(test_mse)
                test_r2 = r2_score(y_test_original_scale, test_predictions)

                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric(label="R-squared (R²)", value=f"{test_r2:.2f}")
                    st.caption("R-squared measures how well the model's predictions fit the actual data. A value of 1.0 means perfect fit, 0.0 means no better than predicting the mean, and negative values indicate a very poor fit.")
                with col_metrics2:
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{test_rmse:,.2f}")
                    st.caption(f"RMSE represents the average magnitude of the errors in predictions. It is in the same units as the target variable ({target_col_eval.replace('_', ' ').title()}). Lower values indicate better accuracy.")

    st.markdown("---")

    # --- 8. Project Details and Footer ---
    st.header("About This Project")
    st.write(
        """
        This project demonstrates a comprehensive data science workflow, from data preparation to interactive visualization and machine learning prediction. Key components include:
        -   **Data Loading & Preprocessing:** Efficiently handling time-series data, imputing missing values, and engineering new features using `pandas`.
        -   **Exploratory Data Analysis (EDA):** Creating insightful visualizations with `matplotlib` and `seaborn` to uncover trends, distributions, and relationships within the data.
        -   **Supervised Machine Learning (Regression):** Building and evaluating robust `RandomForestRegressor` models from `scikit-learn` to predict continuous outcomes (new deaths and daily vaccinations).
        -   **Model Persistence:** Saving and loading trained models using `joblib` for efficient deployment and reuse.
        -   **Interactive Dashboard Development:** Designing a dynamic and user-friendly web application with `Streamlit`, allowing real-time interaction with data and models.

        **Data Source:**
        The dataset used for this project is **"COVID vaccination vs. mortality"** from Kaggle:
        [https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality](https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-mortality)

        **Dataset Context:**
        The COVID-19 pandemic significantly impacted global health. This dataset was compiled to help investigate the potential relationship between coronavirus vaccination efforts and mortality rates, providing valuable time-series data on vaccination progress and daily death counts across various countries.

        This dashboard serves as a practical example of applying data science methodologies to real-world public health data.
        """
    )
st.markdown("---")
st.write("Developed by Augustine Khumalo | Connect with me on LinkedIn!(https://www.linkedin.com/in/augustine-khumalo)")
