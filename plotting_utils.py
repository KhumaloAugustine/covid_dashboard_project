# This file contains all functions responsible for generating visualizations
# for the COVID-19 dashboard. This separation promotes reusability and
# keeps the main application file clean.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import STL # For time series decomposition

def plot_matplotlib_figure(fig, title=""):
    """
    Helper function to display a Matplotlib figure in Streamlit and close it.
    Args:
        fig (matplotlib.figure.Figure): The figure object to display.
        title (str): An optional title for the plot, not used directly by matplotlib
                     but good for context in calling functions.
    """
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

def display_summary_metrics(data):
    """
    Displays key aggregated metrics at the top of the Data Overview tab.
    
    Args:
        data (pandas.DataFrame): The filtered data to calculate metrics from.
    """
    st.subheader("Summary Metrics for Selected Data")
    col_metrics1, col_metrics2, col_metrics3, col_metrics4, col_metrics5, col_metrics6 = st.columns(6) 
    with col_metrics1:
        st.metric(label="Total New Deaths", value=f"{int(data['New_deaths'].sum()):,}",
                  help="Sum of new deaths recorded for the selected countries over the chosen period.")
    with col_metrics2:
        st.metric(label="Max Total Vaccinations", value=f"{int(data['total_vaccinations'].max()):,}",
                  help="Highest recorded total vaccinations across selected countries in the period.")
    with col_metrics3:
        st.metric(label="Average Vaccination Coverage", value=f"{data['vaccination_coverage'].mean():.2%}",
                  help="Average of 'people fully vaccinated' divided by 'population' for all data points in the selection.")
    with col_metrics4:
        st.metric(label="Selected Countries", value=f"{len(data['country'].unique())}",
                  help="Number of unique countries currently selected in the sidebar filter.")
    with col_metrics5:
        st.metric(label="Overall Avg Daily Deaths", value=f"{data['New_deaths'].mean():,.2f}",
                  help="Average daily new deaths across all selected countries and dates.")
    with col_metrics6:
        st.metric(label="Overall Avg Deaths per Million", value=f"{data['new_deaths_per_million'].mean():,.2f}",
                  help="Average daily new deaths per million people.")
    
    col_metrics_new1, col_metrics_new2, col_metrics_new3, col_metrics_new4 = st.columns(4) 
    with col_metrics_new1:
        st.metric(label="Overall Max Daily Deaths", value=f"{int(data['New_deaths'].max()):,}",
                  help="Highest number of new deaths recorded on any single day within the selected data.")
    with col_metrics_new2:
        st.metric(label="Overall Max Daily Vaccinations", value=f"{int(data['daily_vaccinations'].max()):,}",
                  help="Highest number of daily vaccinations recorded on any single day within the selected data.") 
    with col_metrics_new3:
        st.metric(label="Overall Average Daily Vaccinations", value=f"{data['daily_vaccinations'].mean():,.0f}",
                  help="Average daily vaccinations across all selected countries and dates.") 
    with col_metrics_new4:
        st.metric(label="Overall Avg Daily Vaccinated per Million", value=f"{data['daily_vaccinated_per_million'].mean():,.2f}",
                  help="Average daily vaccinations per million people.")
    st.markdown("---")

def display_raw_data_sample(data):
    """
    Displays a sample of the raw filtered data.
    
    Args:
        data (pandas.DataFrame): The filtered data to display.
    """
    st.subheader("Raw Filtered Data Sample")
    st.write("A glimpse into the raw data based on your current filters.")
    with st.spinner("Loading raw data sample..."):
        st.dataframe(data.head(10))
        st.caption(f"Displaying the first 10 rows of {len(data)} entries. This sample reflects **all active filters** (country and date range).")
    st.markdown("---")


def display_summary_statistics(data):
    """
    Displays descriptive statistics for numerical columns in the filtered data.
    
    Args:
        data (pandas.DataFrame): The filtered data to analyze.
    """
    st.subheader("Summary Statistics for Numerical Data")
    st.write("Descriptive statistics (count, mean, std, min, max, quartiles) for all numerical columns in your filtered dataset.")
    with st.spinner("Calculating summary statistics..."):
        st.dataframe(data.describe().transpose())
        st.caption("These statistics provide a quick overview of the central tendency, dispersion, and shape of the numerical features.")
    st.markdown("---")

def display_country_comparison_table(data):
    """
    Displays a table comparing key aggregated metrics across selected countries.
    Only shown if multiple countries are selected.
    
    Args:
        data (pandas.DataFrame): The filtered data for comparison.
    """
    if len(data['country'].unique()) > 1:
        st.subheader("Country Comparison: Key Aggregated Metrics")
        st.write("Compare aggregated metrics across your selected countries for the chosen date range.")
        with st.spinner("Generating country comparison table..."):
            comparison_table = data.groupby('country').agg(
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
            
            # Format columns for better readability
            for col in ['Total_New_Deaths', 'Max_Total_Vaccinations', 'Max_Population', 'Avg_Daily_Vaccinations']:
                comparison_table[col] = comparison_table[col].apply(lambda x: f"{int(x):,}")
            for col in ['Avg_Daily_New_Deaths', 'Avg_New_Deaths_Per_Million', 'Max_Total_Vaccinations_Per_Hundred', 'Avg_Daily_Vaccinated_Per_Million']:
                comparison_table[col] = comparison_table[col].apply(lambda x: f"{x:.2f}")
            comparison_table['Avg_Vaccination_Coverage'] = comparison_table['Avg_Vaccination_Coverage'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(comparison_table.set_index('country'))
            st.caption("Aggregated statistics for each selected country over the chosen period. 'Max' values represent the highest recorded within the period.")
    else:
        st.info("Select multiple countries in the sidebar to view a comparative summary table and plots.")
    st.markdown("---")

def display_latest_country_stats(data):
    """
    Displays the latest available key statistics for each selected country within the filtered date range.
    
    Args:
        data (pandas.DataFrame): The filtered data.
    """
    st.subheader("Key Statistics per Country (Latest Data Point)")
    st.write("View the most recent available data points for key metrics for each selected country within your chosen date range.")
    with st.spinner("Calculating latest country statistics..."):
        latest_country_stats = data.sort_values(by=['country', 'date']).groupby('country').agg(
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
    st.markdown("---")

def generate_bar_plots_for_period_summary(data):
    """
    Generates various bar plots summarizing data (Total Deaths, Highest Total Vaccinations,
    Latest Vaccination Coverage, Average New Deaths per Million, Average Daily Vaccinations)
    for the selected period and countries.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Total Deaths and Vaccinations for Selected Period (Bar Plots)")
    st.write("Visualizations showing total new deaths, highest total vaccinations reached, and average daily vaccinations for each selected country over the chosen period.")
    
    period_summary = data.groupby('country').agg(
        Total_New_Deaths=('New_deaths', 'sum'),
        Highest_Total_Vaccinations_Reached=('total_vaccinations', 'max'),
        Latest_Vaccination_Coverage=('vaccination_coverage', 'max'),
        Avg_New_Deaths_Per_Million=('new_deaths_per_million', 'mean'), 
        Avg_Daily_Vaccinations=('daily_vaccinations', 'mean') 
    ).reset_index()

    if period_summary.empty:
        st.info("No data to generate bar plots for the selected period.")
        return

    with st.spinner("Generating bar plots for totals..."):
        # Helper for individual bar plots
        def plot_single_bar(df, x_col, y_col, title, x_label, y_label, palette, format_func=None):
            fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))
            sns.barplot(data=df.sort_values(x_col, ascending=False), x=x_col, y=y_col, hue=y_col, palette=palette, legend=False, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.ticklabel_format(style='plain', axis='x')
            if format_func:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            plot_matplotlib_figure(fig)

        plot_single_bar(period_summary, 'Total_New_Deaths', 'country', 'Total New Deaths per Country (Selected Period)', 'Total New Deaths', 'Country', 'viridis')
        plot_single_bar(period_summary, 'Highest_Total_Vaccinations_Reached', 'country', 'Highest Total Vaccinations Reached per Country', 'Highest Total Vaccinations', 'Country', 'cividis')
        
        st.subheader("Latest Vaccination Coverage per Country")
        plot_single_bar(period_summary, 'Latest_Vaccination_Coverage', 'country', 'Latest Vaccination Coverage per Country (Selected Period)', 'Latest Vaccination Coverage (%)', 'Country', 'rocket', format_func=lambda x, _: f'{x:.1%}')

        st.subheader("Average Daily New Deaths per Million per Country")
        plot_single_bar(period_summary, 'Avg_New_Deaths_Per_Million', 'country', 'Average Daily New Deaths per Million per Country (Selected Period)', 'Average New Deaths per Million', 'Country', 'plasma')
        
        st.subheader("Average Daily Vaccinations per Country")
        plot_single_bar(period_summary, 'Avg_Daily_Vaccinations', 'country', 'Average Daily Vaccinations per Country (Selected Period)', 'Average Daily Vaccinations', 'Country', 'cubehelix')
    st.markdown("---")

def display_top_bottom_countries(data, metric, title_prefix, help_text):
    """
    Displays tables for top and bottom N countries based on a given metric.
    
    Args:
        data (pandas.DataFrame): The filtered data.
        metric (str): The column name of the metric to sort by.
        title_prefix (str): Prefix for the subheaders (e.g., "Deaths Growth").
        help_text (str): Help text for the slider.
    """
    st.subheader(f"Top/Bottom Countries by {title_prefix}")
    st.write(f"Easily identify countries with the highest and lowest average daily {title_prefix.lower()} within your selected timeframe.")
    
    all_countries_count = len(data['country'].unique())
    if all_countries_count == 0:
        st.info(f"No data available to display top/bottom countries for {title_prefix}.")
        return

    top_n_countries = st.slider(
        f"Select N for Top/Bottom {title_prefix.replace('Growth Rate', 'Growth')} Countries:", 
        min_value=1, 
        max_value=min(10, all_countries_count), 
        value=min(5, all_countries_count), 
        key=f'top_n_{metric}', 
        help=help_text
    ) 
    
    with st.spinner(f"Calculating top/bottom {top_n_countries} countries..."):
        avg_metric = data.groupby('country')[metric].mean().sort_values(ascending=False)
        
        col_top_bottom1, col_top_bottom2 = st.columns(2)
        with col_top_bottom1:
            st.write(f"#### Top {top_n_countries} Countries by Average {title_prefix}")
            st.dataframe(avg_metric.head(top_n_countries).reset_index().rename(columns={metric: f'Avg. {title_prefix}'}).set_index('country'))
        with col_top_bottom2:
            st.write(f"#### Bottom {top_n_countries} Countries by Average {title_prefix}")
            st.dataframe(avg_metric.tail(top_n_countries).reset_index().rename(columns={metric: f'Avg. {title_prefix}'}).set_index('country'))
    st.markdown("---")

def display_data_dictionary():
    """Displays the data dictionary explaining each column."""
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
        * **days_since_start:** The number of days that have passed since the earliest date in the entire dataset. This time-based feature can capture overall trends in time-series models.
        """
    )
    st.markdown("---")


def plot_time_series(data):
    """
    Generates customizable time series plots (Line, Area, Bar) for selected metrics.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Customizable Time Series Trends")
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
        fig_ts, ax_ts = plt.subplots(figsize=(12, 6))

        if plot_type == 'Line Plot':
            sns.lineplot(data=data, x='date', y=time_series_metric, hue='country', marker='o', ax=ax_ts)
        elif plot_type == 'Area Plot':
            sns.lineplot(data=data, x='date', y=time_series_metric, hue='country', ax=ax_ts)
            for country in data['country'].unique():
                country_data = data[data['country'] == country].sort_values('date')
                ax_ts.fill_between(country_data['date'], country_data[time_series_metric], alpha=0.3)
        else: # Bar Plot
            sns.barplot(data=data, x='date', y=time_series_metric, hue='country', dodge=True, ax=ax_ts)
            # Dynamically adjust x-tick frequency for readability on bar plots with many dates
            ax_ts.set_xticks(ax_ts.get_xticks()[::max(1, len(data['date'].unique()) // 10)]) 

        ax_ts.set_title(f'{time_series_metric.replace("_", " ").title()} Over Time')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel(time_series_metric.replace('_', ' ').title())
        ax_ts.tick_params(axis='x', rotation=45) # Rotate date labels for better readability
        ax_ts.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on Y-axis
        plot_matplotlib_figure(fig_ts)
    st.markdown("---")

def plot_vaccination_progress(data):
    """
    Plots vaccination coverage or total vaccinations per hundred over time.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Vaccination Progress Over Time")
    st.write("Track the cumulative vaccination coverage and total vaccinations over time for the selected countries.")

    vacc_progress_metric = st.selectbox(
        "Select Vaccination Progress Metric:",
        options=['vaccination_coverage', 'total_vaccinations_per_hundred'],
        index=0,
        key='vacc_progress_metric_select'
    )

    with st.spinner(f"Generating vaccination progress plot for {vacc_progress_metric.replace('_', ' ').title()}..."):
        fig_vacc_progress, ax_vacc_progress = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data, x='date', y=vacc_progress_metric, hue='country', marker='o', ax=ax_vacc_progress)
        ax_vacc_progress.set_title(f'{vacc_progress_metric.replace("_", " ").title()} Over Time')
        ax_vacc_progress.set_xlabel('Date')
        ax_vacc_progress.set_ylabel(vacc_progress_metric.replace('_', ' ').title())
        ax_vacc_progress.tick_params(axis='x', rotation=45)
        if vacc_progress_metric == 'vaccination_coverage':
            ax_vacc_progress.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plot_matplotlib_figure(fig_vacc_progress)
    st.markdown("---")

def plot_cumulative_and_rolling_averages(data):
    """
    Plots cumulative sums and rolling averages for new deaths and daily vaccinations.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Additional Time-Series Analysis: Cumulative & Rolling Averages")
    st.write("These plots provide a 'survival-like' perspective by showing cumulative totals and smoothed trends, which can help in understanding long-term impacts and underlying trends by reducing noise.")

    data_sorted = data.sort_values(by=['country', 'date'])

    col_add_ts1, col_add_ts2 = st.columns(2)
    with col_add_ts1:
        st.write("#### Cumulative New Deaths Over Time")
        with st.spinner("Calculating cumulative deaths..."):
            data_sorted['Cumulative_New_Deaths'] = data_sorted.groupby('country')['New_deaths'].cumsum()
            fig_cum_deaths, ax_cum_deaths = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data_sorted, x='date', y='Cumulative_New_Deaths', hue='country', marker='o', ax=ax_cum_deaths)
            ax_cum_deaths.set_title('Cumulative New Deaths Over Time')
            ax_cum_deaths.set_xlabel('Date')
            ax_cum_deaths.set_ylabel('Cumulative New Deaths')
            ax_cum_deaths.tick_params(axis='x', rotation=45)
            ax_cum_deaths.ticklabel_format(style='plain', axis='y') 
            plot_matplotlib_figure(fig_cum_deaths)

    with col_add_ts2:
        st.write("#### 7-Day Rolling Average of New Deaths")
        with st.spinner("Calculating rolling average deaths..."):
            data_sorted['Rolling_Avg_Deaths'] = data_sorted.groupby('country')['New_deaths'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            fig_roll_deaths, ax_roll_deaths = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data_sorted, x='date', y='Rolling_Avg_Deaths', hue='country', marker='o', ax=ax_roll_deaths)
            ax_roll_deaths.set_title('7-Day Rolling Average of New Deaths')
            ax_roll_deaths.set_xlabel('Date')
            ax_roll_deaths.set_ylabel('Avg. New Deaths (Past 7 Days)')
            ax_roll_deaths.tick_params(axis='x', rotation=45)
            plot_matplotlib_figure(fig_roll_deaths)
    
    col_add_ts3, col_add_ts4 = st.columns(2)
    with col_add_ts3:
        st.write("#### Cumulative Daily Vaccinations Over Time")
        with st.spinner("Calculating cumulative vaccinations..."):
            data_sorted['Cumulative_Daily_Vaccinations'] = data_sorted.groupby('country')['daily_vaccinations'].cumsum()
            fig_cum_vacc, ax_cum_vacc = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data_sorted, x='date', y='Cumulative_Daily_Vaccinations', hue='country', marker='o', ax=ax_cum_vacc)
            ax_cum_vacc.set_title('Cumulative Daily Vaccinations Over Time')
            ax_cum_vacc.set_xlabel('Date')
            ax_cum_vacc.set_ylabel('Cumulative Daily Vaccinations')
            ax_cum_vacc.tick_params(axis='x', rotation=45)
            ax_cum_vacc.ticklabel_format(style='plain', axis='y')
            plot_matplotlib_figure(fig_cum_vacc)

    with col_add_ts4:
        st.write("#### 7-Day Rolling Average of Daily Vaccinations")
        with st.spinner("Calculating rolling average vaccinations..."):
            data_sorted['Rolling_Avg_Daily_Vaccinations'] = data_sorted.groupby('country')['daily_vaccinations'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            fig_roll_vacc, ax_roll_vacc = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data_sorted, x='date', y='Rolling_Avg_Daily_Vaccinations', hue='country', marker='o', ax=ax_roll_vacc)
            ax_roll_vacc.set_title('7-Day Rolling Average of Daily Vaccinations')
            ax_roll_vacc.set_xlabel('Date')
            ax_roll_vacc.set_ylabel('Avg. Daily Vaccinations (Past 7 Days)')
            ax_roll_vacc.tick_params(axis='x', rotation=45)
            plot_matplotlib_figure(fig_roll_vacc)
    st.markdown("---")

def plot_distributions_and_correlations(data):
    """
    Plots distributions of key metrics (histograms) and a correlation heatmap for numerical features.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Key Distributions and Relationships")
    st.write("Understand the spread and shape of your data, and how different numerical features correlate with each other.")

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.write("#### Distribution of New Deaths")
        with st.spinner("Generating distribution of new deaths..."):
            fig_hist_deaths, ax_hist_deaths = plt.subplots(figsize=(8, 5))
            sns.histplot(data['New_deaths'], kde=True, bins=min(30, len(data)), color='salmon', ax=ax_hist_deaths)
            ax_hist_deaths.set_title('Distribution of New Deaths')
            ax_hist_deaths.set_xlabel('New Deaths')
            ax_hist_deaths.set_ylabel('Frequency')
            plot_matplotlib_figure(fig_hist_deaths)

    with col_dist2:
        st.write("#### Distribution of Vaccination Coverage (People Fully Vaccinated / Population)")
        with st.spinner("Generating vaccination coverage distribution..."):
            fig_hist_coverage, ax_hist_coverage = plt.subplots(figsize=(8, 5))
            sns.histplot(data['vaccination_coverage'], kde=True, bins=min(30, len(data)), color='lightgreen', ax=ax_hist_coverage)
            ax_hist_coverage.set_title('Distribution of Vaccination Coverage')
            ax_hist_coverage.set_xlabel('Vaccination Coverage')
            ax_hist_coverage.set_ylabel('Frequency')
            plot_matplotlib_figure(fig_hist_coverage)
    
    with st.spinner("Generating correlation heatmap..."):
        st.write("#### Correlation Heatmap of Numerical Features")
        st.write("A correlation heatmap shows how strongly pairs of numerical variables are related. Values closer to 1 or -1 indicate a stronger relationship (positive or negative).")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        numerical_data_for_corr = data[
            ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
             'New_deaths', 'population', 'ratio', 'vaccination_coverage', 'days_since_start',
             'new_deaths_per_million', 'total_vaccinations_per_hundred',
             'daily_vaccinations', 'daily_vaccinated_per_million',
             'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate']
        ].corr()
        sns.heatmap(numerical_data_for_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title('Correlation Matrix of Key Metrics')
        plot_matplotlib_figure(fig_corr)
    st.markdown("---")

def plot_interactive_scatter(data):
    """
    Generates an interactive scatter plot to explore relationships between numerical metrics.
    Allows selection of X, Y axes, color by country, and optional regression line.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Interactive Scatter Plot: Explore Relationships")
    st.write("Select any two numerical metrics to visualize their relationship using a scatter plot. A regression line can be added to show the general trend.")
    
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
            sns.scatterplot(data=data, x=scatter_x, y=scatter_y, hue='country', ax=ax_scatter, s=100, alpha=0.7)
            if add_regression_line:
                for country_val in data['country'].unique(): 
                    # Use regplot for each country's regression line
                    sns.regplot(data=data[data['country'] == country_val], x=scatter_x, y=scatter_y, ax=ax_scatter, scatter=False, line_kws={'linestyle':'--', 'alpha':0.6})
            ax_scatter.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(data=data, x=scatter_x, y=scatter_y, ax=ax_scatter, s=100, alpha=0.7)
            if add_regression_line:
                sns.regplot(data=data, x=scatter_x, y=scatter_y, ax=ax_scatter, scatter=False, color='red', line_kws={'alpha':0.8})
        
        ax_scatter.set_title(f'Scatter Plot: {scatter_y.replace("_", " ").title()} vs. {scatter_x.replace("_", " ").title()}')
        ax_scatter.set_xlabel(scatter_x.replace('_', ' ').title())
        ax_scatter.set_ylabel(scatter_y.replace('_', ' ').title())
        ax_scatter.ticklabel_format(style='plain', axis='both')
        plot_matplotlib_figure(fig_scatter)
    st.markdown("---")

def plot_comparative_distributions(data):
    """
    Plots comparative distributions (Kernel Density Estimation - KDE plots) across countries
    if multiple countries are selected.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    if len(data['country'].unique()) > 1:
        st.subheader("Comparative Density Plots (KDE)")
        st.write("Kernel Density Estimation (KDE) plots visualize the distribution shape of a metric across selected countries. This helps in understanding data concentration and spread without binning.")
        kde_metric = st.selectbox(
            "Select Metric for Density Plot:",
            options=['New_deaths', 'new_deaths_per_million', 'vaccination_coverage', 'total_vaccinations_per_hundred', 'daily_vaccinations', 'daily_vaccinated_per_million', 'daily_deaths_growth_rate', 'daily_vaccinations_growth_rate'],
            key='kde_metric',
            help="Choose a metric to compare its density distribution across selected countries."
        )
        with st.spinner(f"Generating density plot for {kde_metric.replace('_', ' ').title()}..."):
            fig_kde, ax_kde = plt.subplots(figsize=(10, 6))
            sns.kdeplot(data=data, x=kde_metric, hue='country', fill=True, common_norm=False, ax=ax_kde)
            ax_kde.set_title(f'Density Plot of {kde_metric.replace("_", " ").title()} by Country')
            ax_kde.set_xlabel(kde_metric.replace('_', ' ').title())
            ax_kde.set_ylabel('Density')
            ax_kde.ticklabel_format(style='plain', axis='x')
            plot_matplotlib_figure(fig_kde)
    else:
        st.info("Select multiple countries in the sidebar to view comparative distribution plots.")
    st.markdown("---")

def plot_daily_change(data):
    """
    Plots daily change in selected metrics over time for selected countries.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Daily Change in Metrics")
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
        data_sorted_for_diff = data.sort_values(by=['country', 'date'])
        
        data_to_plot_col = daily_change_metric
        y_label_diff = daily_change_metric.replace("_", " ").title()
        
        if daily_change_metric not in ['New_deaths', 'daily_vaccinations']:
            daily_data_to_plot_temp = data_sorted_for_diff.copy()
            daily_data_to_plot_temp[f'{daily_change_metric}_calculated_daily_change'] = daily_data_to_plot_temp.groupby('country')[daily_change_metric].diff().fillna(0)
            data_to_plot_col = f'{daily_change_metric}_calculated_daily_change'
            y_label_diff = f'Daily Change in {daily_change_metric.replace("_", " ").title()}'

        fig_daily_change, ax_daily_change = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data_sorted_for_diff, x='date', y=data_to_plot_col, hue='country', ax=ax_daily_change)
        ax_daily_change.set_title(f'Daily Trends for {y_label_diff}')
        ax_daily_change.set_xlabel('Date')
        ax_daily_change.set_ylabel(y_label_diff)
        ax_daily_change.tick_params(axis='x', rotation=45)
        ax_daily_change.ticklabel_format(style='plain', axis='y')
        plot_matplotlib_figure(fig_daily_change)
    st.markdown("---")

def plot_latest_vaccination_status_distribution(data, date_range_end):
    """
    Plots a pie chart showing the overall approximate distribution of vaccination status
    (fully vaccinated, partially vaccinated, unvaccinated) for the latest available date
    across all selected countries.
    
    Args:
        data (pandas.DataFrame): The filtered data.
        date_range_end (datetime.date): The end date of the selected date range, for display in title.
    """
    st.subheader("Latest Vaccination Status Distribution")
    st.write("This pie chart shows the overall approximate distribution of vaccination status (fully vaccinated, partially vaccinated, unvaccinated) for the latest available date across all selected countries.")

    with st.spinner("Calculating vaccination status distribution..."):
        latest_date_per_country = data.groupby('country')['date'].max().reset_index()
        latest_status_data = pd.merge(latest_date_per_country, data, on=['country', 'date'], how='left')

        if not latest_status_data.empty:
            total_people_vaccinated = latest_status_data['people_vaccinated'].sum()
            total_people_fully_vaccinated = latest_status_data['people_fully_vaccinated'].sum()
            total_population = latest_status_data['population'].sum()

            total_people_vaccinated_adjusted = max(total_people_vaccinated, total_people_fully_vaccinated)
            
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
                ax_pie.set_title(f'Latest Vaccination Status Distribution ({date_range_end.strftime("%Y-%m-%d")})')
                ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                plot_matplotlib_figure(fig_pie)
            else:
                st.info("No vaccination status data to display for the selected period. This might happen if 'people_vaccinated' or 'people_fully_vaccinated' counts are consistently zero or missing for the latest dates.")
        else:
            st.info("No latest vaccination status data found for the selected countries/date range. Please check your filters.")
    st.markdown("---")

def plot_monthly_yearly_trends(data):
    """
    Plots monthly and yearly aggregated trends for selected metrics.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Monthly and Yearly Trends")
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
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        temp_df_agg = data.copy()
        if aggregation_level == 'Month':
            temp_df_agg['time_period'] = temp_df_agg['date'].dt.to_period('M')
            grouped_data = temp_df_agg.groupby(['time_period', 'country'])[aggregation_metric].sum().reset_index()
            grouped_data['time_period'] = grouped_data['time_period'].astype(str)
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
        plot_matplotlib_figure(fig_agg)
    st.markdown("---")

def plot_growth_rate_trends(data):
    """
    Plots daily growth rate trends for new deaths and daily vaccinations.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Growth Rate Trends")
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
        sns.lineplot(data=data, x='date', y=growth_rate_metric, hue='country', ax=ax_growth)
        ax_growth.set_title(f'Daily {growth_rate_metric.replace("_", " ").title()} Over Time')
        ax_growth.set_xlabel('Date')
        ax_growth.set_ylabel('Percentage Change')
        ax_growth.tick_params(axis='x', rotation=45)
        ax_growth.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')) # Format as percentage
        plot_matplotlib_figure(fig_growth)
    st.markdown("---")

def plot_lag_plot(data):
    """
    Generates a lag plot to visualize autocorrelation for a selected metric.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Lag Plot for Autocorrelation")
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
        lag_data = data[['date', 'country', lag_plot_metric]].copy()
        
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
            plot_matplotlib_figure(fig_lag)
        else:
            st.info("Not enough data to generate lag plot for the selected parameters. Try a smaller lag or wider date range.")
    st.markdown("---")

def plot_daily_metrics_comparison_for_date(data):
    """
    Compares daily metrics (New Deaths or Daily Vaccinations) across selected countries
    for a specific chosen date using a bar chart.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Daily Metrics Comparison for a Specific Date")
    st.write("Select a specific date to compare daily deaths or daily vaccinations across your chosen countries using a bar chart.")

    single_date_metric = st.selectbox(
        "Select Daily Metric for Date Comparison:",
        options=['New_deaths', 'daily_vaccinations'],
        index=0,
        key='single_date_metric_select'
    )
    
    # Ensure the date picker respects the overall filtered data's min/max dates
    single_comparison_date = st.date_input(
        "Select Comparison Date:",
        value=data['date'].max().date(), # Default to the latest date in filtered data
        min_value=data['date'].min().date(),
        max_value=data['date'].max().date(),
        key='single_comparison_date_picker'
    )

    with st.spinner(f"Generating daily {single_date_metric.replace('_', ' ').lower()} comparison for {single_comparison_date}..."):
        daily_comparison_data = data[data['date'].dt.date == single_comparison_date]
        
        if not daily_comparison_data.empty:
            fig_daily_comp, ax_daily_comp = plt.subplots(figsize=(10, max(6, len(daily_comparison_data) * 0.5)))
            sns.barplot(data=daily_comparison_data.sort_values(single_date_metric, ascending=False), 
                        x=single_date_metric, y='country', hue='country', palette='viridis', legend=False, ax=ax_daily_comp)
            ax_daily_comp.set_title(f'Daily {single_date_metric.replace("_", " ").title()} on {single_comparison_date.strftime("%Y-%m-%d")}')
            ax_daily_comp.set_xlabel(single_date_metric.replace('_', ' ').title())
            ax_daily_comp.set_ylabel('Country')
            ax_daily_comp.ticklabel_format(style='plain', axis='x')
            plot_matplotlib_figure(fig_daily_comp)
        else:
            st.info(f"No data available for {single_date_metric.replace('_', ' ').lower()} on {single_comparison_date.strftime('%Y-%m-%d')} for the selected countries.")
    st.markdown("---")

def plot_geographic_distribution_map(full_data):
    """
    Generates a choropleth map to visualize spatial distribution of COVID-19 metrics.
    
    Args:
        full_data (pandas.DataFrame): The complete, unfiltered COVID-19 dataset. This is used
                                      to ensure all countries are available for mapping, even
                                      if not currently selected in the sidebar filter.
    """
    st.subheader("Geographic Distribution of Key Metrics")
    st.write("Visualize the spatial distribution of COVID-19 metrics on a world map. Select a metric to display its value across countries.")

    # Prepare data for map: get the latest data point for each country in the full dataset
    map_base_data = full_data.sort_values('date').groupby('iso_code').last().reset_index()
    map_base_data = map_base_data[map_base_data['iso_code'].notna()] # Ensure ISO codes are present

    # Calculate cumulative deaths for the map from the full dataset
    cumulative_deaths_for_map = full_data.groupby(['iso_code', 'country'])['New_deaths'].sum().reset_index()
    cumulative_deaths_for_map.rename(columns={'New_deaths': 'Cumulative_New_Deaths'}, inplace=True)

    # Merge cumulative deaths into the map_base_data
    map_final_data = pd.merge(map_base_data, cumulative_deaths_for_map[['iso_code', 'Cumulative_New_Deaths']], on='iso_code', how='left')

    # Define map metric options with the exact column names available in map_final_data
    map_metric_options = {
        'Cumulative New Deaths': 'Cumulative_New_Deaths',
        'Latest Total Vaccinations': 'total_vaccinations',
        'Latest Vaccination Coverage (%)': 'vaccination_coverage',
        'Latest New Deaths per Million': 'new_deaths_per_million',
        'Latest Daily Vaccinations': 'daily_vaccinations',
        'Latest Population': 'population'
    }
    
    selected_map_metric_display = st.selectbox(
        "Select Metric for Map:",
        options=list(map_metric_options.keys()),
        index=0,
        key='map_metric_select'
    )
    selected_map_metric_column = map_metric_options[selected_map_metric_display]

    # Ensure selected metric column exists and fill NaNs
    if selected_map_metric_column in map_final_data.columns:
        map_final_data[selected_map_metric_column] = map_final_data[selected_map_metric_column].fillna(0)
    else:
        # Fallback if column is genuinely missing (e.g., if a new metric is added to options but not data)
        map_final_data[selected_map_metric_column] = 0 

    # Apply percentage formatting for display on map if it's a coverage metric
    # This 'vaccination_coverage_for_color' column is for coloring purposes only, not for hover_data.
    if selected_map_metric_column == 'vaccination_coverage':
        map_final_data['vaccination_coverage_for_color'] = map_final_data['vaccination_coverage'] * 100 
        color_column_for_plot = 'vaccination_coverage_for_color'
    else:
        color_column_for_plot = selected_map_metric_column

    with st.spinner(f"Generating map for {selected_map_metric_display}..."):
        # Check if the necessary columns are present before plotting
        if not map_final_data['iso_code'].empty and color_column_for_plot in map_final_data.columns:
            fig_map = px.choropleth(
                map_final_data,
                locations="iso_code",
                color=color_column_for_plot, # Use the potentially new column for display
                hover_name="country",
                # Update hover_data to use actual column names from map_final_data
                # The 'vaccination_coverage_for_color' column should not be in hover_data directly,
                # as it's a temporary column for color mapping. We show 'vaccination_coverage' with formatting instead.
                hover_data={
                    selected_map_metric_column: True, # Show selected metric's raw value
                    'total_vaccinations': True, 
                    'people_vaccinated': True,
                    'people_fully_vaccinated': True,
                    'vaccination_coverage': ':.2%', # Format raw coverage as percentage
                    'New_deaths': True,
                    'new_deaths_per_million': ':.2f',
                    'population': ':,',
                    'daily_vaccinations': True,
                    'daily_vaccinated_per_million': ':.2f',
                    'Cumulative_New_Deaths': True 
                },
                color_continuous_scale=px.colors.sequential.Plasma if 'Vaccination' in selected_map_metric_display else px.colors.sequential.Viridis,
                title=f'{selected_map_metric_display} by Country',
                template="plotly_dark" # Dark theme for better contrast
            )
            fig_map.update_geos(
                showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGrey",
                showocean=True, oceancolor="LightBlue",
                showlakes=True, lakecolor="LightBlue",
                showrivers=True, rivercolor="LightBlue",
                projection_type="natural earth" # Good general-purpose projection
            )
            fig_map.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0}) # Adjust margins and height for better fit
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No data available to render the map for the selected metric.")
    st.markdown("---")

def plot_pair_plot(data):
    """
    Generates a Seaborn pair plot for selected numerical columns.
    Useful for visualizing relationships between multiple variables.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Pair Plot of Key Numerical Metrics")
    st.write("Visualize the pairwise relationships and distributions of several key numerical metrics. This can reveal correlations and patterns that single plots might miss.")
    
    # Select a subset of numerical columns to avoid an overly dense pairplot
    default_pair_cols = ['New_deaths', 'daily_vaccinations', 'vaccination_coverage', 'population']
    
    # Filter for columns that actually exist in the DataFrame
    available_numerical_cols = [col for col in default_pair_cols if col in data.columns]
    
    if not available_numerical_cols:
        st.info("No suitable numerical columns available for pair plot. Please check data or filters.")
        return

    selected_pair_cols = st.multiselect(
        "Select Metrics for Pair Plot:",
        options=[col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col not in ['days_since_start']], # Exclude days_since_start for clarity
        default=available_numerical_cols,
        key='pair_plot_cols'
    )

    if not selected_pair_cols:
        st.info("Please select at least one metric for the pair plot.")
        return

    with st.spinner("Generating pair plot (this may take a while for large datasets)..."):
        # Ensure data is not too large for pairplot performance
        if len(data) > 5000: # Arbitrary threshold, adjust as needed
            st.warning("Pair plot might be slow for large datasets. Consider narrowing your date range or countries.")
        
        # Add 'country' as hue if multiple countries are selected
        hue_param = 'country' if len(data['country'].unique()) > 1 else None
        
        fig_pair = sns.pairplot(data[selected_pair_cols + ([hue_param] if hue_param else [])].dropna(), hue=hue_param, diag_kind='kde')
        st.pyplot(fig_pair)
        # Corrected line: plt.close() should be called with the Figure object, which is fig_pair.fig for a PairGrid
        plt.close(fig_pair.fig) # Access the underlying Figure object
    st.markdown("---")

def plot_time_series_decomposition(data):
    """
    Performs and plots time series decomposition (STL) for a selected metric.
    Decomposes a time series into trend, seasonal, and residual components.
    
    Args:
        data (pandas.DataFrame): The filtered data for analysis.
    """
    st.subheader("Time Series Decomposition (STL)")
    st.write("Decompose a time series into its trend, seasonal, and residual components to better understand underlying patterns. This works best for single-country data to avoid mixing trends.")

    time_series_decomposition_metric = st.selectbox(
        "Select Metric for Decomposition:",
        options=['New_deaths', 'daily_vaccinations', 'total_vaccinations'],
        index=0,
        key='decomposition_metric'
    )

    # Time series decomposition is best for a single, continuous series.
    # So, we'll only allow it if one country is selected.
    if len(data['country'].unique()) == 1:
        selected_country_data = data[data['country'] == data['country'].unique()[0]].set_index('date')[time_series_decomposition_metric].dropna()
        
        if not selected_country_data.empty:
            # Resample to daily frequency and fill missing dates for consistent time series
            selected_country_data = selected_country_data.asfreq('D', fill_value=0) # Use 0 for missing daily counts

            if len(selected_country_data) < 2 * 7: # Need at least two seasonal cycles (e.g., 2 weeks for daily data)
                st.info(f"Not enough data points ({len(selected_country_data)} days) for robust STL decomposition (requires at least two seasonal periods). Please select a longer date range.")
                return

            try:
                # Determine seasonal period. Assuming daily data, common periods are 7 (weekly) or 30/365
                # Let's use a 7-day period for daily data to capture weekly seasonality
                period = 7 # Weekly seasonality for daily data
                
                # Try to run STL decomposition
                stl = STL(selected_country_data, seasonal=period, robust=True)
                res = stl.fit()

                fig_stl = res.plot()
                fig_stl.set_size_inches(10, 8)
                fig_stl.suptitle(f'STL Decomposition of {time_series_decomposition_metric.replace("_", " ").title()} for {data["country"].unique()[0]}', y=1.02) # Adjust title position
                plot_matplotlib_figure(fig_stl)
            except Exception as e:
                st.error(f"Error during STL decomposition: {e}. This might happen if the data has too many zeros or is too short for the chosen period. Try selecting a different metric or country/date range.")
        else:
            st.info("No data available for time series decomposition for the selected metric and country.")
    else:
        st.info("Please select exactly one country in the sidebar filters to enable Time Series Decomposition.")
    st.markdown("---")

def plot_outliers_boxplot(data):
    """
    Generates box plots to visualize potential outliers for selected numerical metrics.
    
    Args:
        data (pandas.DataFrame): The filtered data for plotting.
    """
    st.subheader("Outlier Detection with Box Plots")
    st.write("Box plots display the distribution of data and highlight potential outliers (points beyond the 'whiskers').")

    numerical_cols = ['New_deaths', 'daily_vaccinations', 'new_deaths_per_million', 'daily_vaccinated_per_million', 'total_vaccinations', 'people_fully_vaccinated', 'vaccination_coverage']
    
    outlier_metric = st.selectbox(
        "Select Metric for Outlier Analysis:",
        options=numerical_cols,
        index=0,
        key='outlier_metric'
    )

    with st.spinner(f"Generating box plot for {outlier_metric.replace('_', ' ').title()}..."):
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, y=outlier_metric, x='country', hue='country', palette='coolwarm', legend=False, ax=ax_box)
        ax_box.set_title(f'Box Plot of {outlier_metric.replace("_", " ").title()} by Country')
        ax_box.set_xlabel('Country')
        ax_box.set_ylabel(outlier_metric.replace('_', ' ').title())
        ax_box.ticklabel_format(style='plain', axis='y')
        plot_matplotlib_figure(fig_box)
    st.markdown("---")

def display_data_types_and_uniques(data):
    """
    Displays a table of data types and unique value counts for each column.
    Useful for understanding the structure and cardinality of the data.
    
    Args:
        data (pandas.DataFrame): The filtered data to analyze.
    """
    st.subheader("Data Types and Unique Values")
    st.write("Examine the data type of each column and the number of unique values. This helps identify categorical vs. numerical data and potential data quality issues like unexpected cardinality.")
    
    with st.spinner("Analyzing data types and unique values..."):
        info_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        }).set_index('Column')
        st.dataframe(info_df)
    st.markdown("---")

def plot_missing_values_heatmap(data):
    """
    Generates a heatmap visualizing the distribution of missing values across columns.
    
    Args:
        data (pandas.DataFrame): The data to analyze for missing values.
    """
    st.subheader("Missing Values Heatmap")
    st.write("This heatmap provides a visual representation of missing (NaN) values in your dataset. Rows or columns with large white spaces indicate a significant amount of missing data.")
    
    with st.spinner("Generating missing values heatmap..."):
        fig_missing, ax_missing = plt.subplots(figsize=(12, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax_missing)
        ax_missing.set_title("Missing Values Pattern (White indicates Missing Data)")
        ax_missing.set_xlabel("Columns")
        ax_missing.set_ylabel("Rows")
        plot_matplotlib_figure(fig_missing)
    st.markdown("---")

def display_simple_forecast(data):
    """
    Displays a simple illustrative forecast based on a rolling average.
    This is a conceptual placeholder for more complex forecasting models.
    
    Args:
        data (pandas.DataFrame): The filtered data for forecasting.
    """
    st.subheader("Simple Illustrative Forecast (Moving Average)")
    st.write("This section provides a very basic forward-looking projection using a simple rolling average. For more accurate and robust forecasts, advanced time-series models (e.g., ARIMA, Prophet) are required and can be integrated here.")

    forecast_metric = st.selectbox(
        "Select Metric for Simple Forecast:",
        options=['New_deaths', 'daily_vaccinations'],
        index=0,
        key='simple_forecast_metric' # Changed key to avoid conflict
    )
    forecast_days = st.slider("Number of Days to Forecast:", min_value=1, max_value=30, value=7, key='simple_forecast_days') # Changed key

    # Simple forecast logic (e.g., last 7-day average)
    if len(data['country'].unique()) == 1:
        # Check for adequate historical data before proceeding with rolling average
        if data.empty or data['date'].nunique() < 7: # Need at least 7 days for a 7-day rolling average
            st.info("Not enough historical data (at least 7 days) to perform a meaningful simple forecast for the selected country and metric. Please ensure a longer data range.")
            return

        country_data = data[data['country'] == data['country'].unique()[0]].set_index('date')[forecast_metric].dropna()
        
        if len(country_data) > 7:
            last_avg = country_data.tail(7).mean()
            
            forecast_dates = pd.date_range(start=country_data.index.max() + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            forecast_values = [last_avg] * forecast_days # Simple constant forecast

            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                forecast_metric: forecast_values
            })

            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=country_data.reset_index(), x='date', y=forecast_metric, label='Historical Data', marker='o', ax=ax_forecast)
            sns.lineplot(data=forecast_df, x='date', y=forecast_metric, label='Simple Forecast', linestyle='--', color='red', marker='x', ax=ax_forecast)
            
            ax_forecast.set_title(f'Simple {forecast_metric.replace("_", " ").title()} Forecast for {data["country"].unique()[0]}')
            ax_forecast.set_xlabel('Date')
            ax_forecast.set_ylabel(forecast_metric.replace('_', ' ').title())
            ax_forecast.tick_params(axis='x', rotation=45)
            ax_forecast.ticklabel_format(style='plain', axis='y')
            ax_forecast.legend()
            plot_matplotlib_figure(fig_forecast)
            st.caption("This forecast uses the average of the last 7 days of historical data. More sophisticated models would incorporate trend, seasonality, and exogenous variables.")
        else:
            st.info("Not enough historical data (at least 7 days) to perform a meaningful simple forecast for the selected country and metric. Please ensure a longer data range.")
    else:
        st.info("To use the simple forecast, please select only one country in the sidebar filters.")
    st.markdown("---")

def display_advanced_forecast(data):
    """
    Placeholder for an advanced forecasting section.
    Users can interact with parameters for a more complex model.
    This function will guide the user on where a more robust forecasting model
    (e.g., ARIMA, Prophet, Neural Networks) could be integrated.
    
    Args:
        data (pandas.DataFrame): The filtered data for forecasting.
    """
    st.subheader("Advanced Forecasting (Conceptual)")
    st.write("This section is a conceptual placeholder for integrating more sophisticated time-series forecasting models (e.g., ARIMA, SARIMA, Prophet, or recurrent neural networks like LSTMs). These models can capture complex patterns, seasonality, and trends more accurately.")
    
    st.info("""
    **To implement advanced forecasting:**
    1.  **Choose a Model:** Select a time-series model (e.g., `statsmodels.tsa.arima.model.ARIMA`, `prophet.Prophet`, or a custom PyTorch/TensorFlow model).
    2.  **Train the Model:** Train the chosen model on historical data. This typically involves defining model parameters (e.g., p, d, q for ARIMA; seasonality for Prophet).
    3.  **Generate Forecasts:** Use the trained model to predict future values.
    4.  **Visualize Results:** Plot the historical data alongside the forecasted values and confidence intervals.
    5.  **Evaluate:** Provide metrics like MAE, RMSE, MAPE to assess forecast accuracy.
    
    **Considerations for advanced models:**
    -   **Data Requirements:** Advanced models often require continuous, regularly spaced time series data. Missing dates might need to be imputed.
    -   **Computational Cost:** Training and forecasting with complex models can be computationally intensive, especially for large datasets or many countries.
    -   **Hyperparameter Tuning:** Optimal model performance often requires careful tuning of hyperparameters.
    -   **Uncertainty:** Displaying prediction intervals is crucial for understanding the uncertainty around forecasts.
    """)
    
    if len(data['country'].unique()) == 1:
        selected_country = data['country'].unique()[0]
        st.write(f"For **{selected_country}**, you would feed its time series data (e.g., `New_deaths` or `daily_vaccinations`) into an advanced forecasting model.")
        st.code(f"""
# Example for a hypothetical ARIMA model
# import pmdarima as pm # Auto ARIMA for automatic parameter selection

# # Prepare data for a single country
# country_ts_data = data[data['country'] == '{selected_country}'].set_index('date')['New_deaths'].dropna()
# country_ts_data = country_ts_data.asfreq('D', fill_value=0) # Ensure daily frequency

# if len(country_ts_data) > 30: # Example threshold for model
#     # Fit a simple auto_arima model (conceptual)
#     # model = pm.auto_arima(country_ts_data, seasonal=True, m=7,  # m=7 for weekly seasonality
#     #                       suppress_warnings=True, stepwise=True, trace=False)
#     # forecast = model.predict(n_periods=forecast_days)
#     # st.line_chart(pd.Series(forecast, index=forecast_dates))
#     st.info("Placeholder for advanced model's output (e.g., ARIMA forecast plot).")
# else:
#     st.info("Not enough data for advanced forecasting model for the selected country. Needs more historical data.")
""", language='python')
    else:
        st.info("Advanced forecasting is best demonstrated on a single country's time series. Please select one country in the sidebar to visualize advanced forecasting concepts.")
    st.markdown("---")

def display_scenario_analysis(filtered_data, models, model_features_dict, covid_data): # Added covid_data parameter
    """
    Provides a framework for scenario analysis based on model predictions.
    Allows users to adjust key input features and observe the predicted outcome.
    
    Args:
        filtered_data (pandas.DataFrame): The currently filtered data for default values.
        models (dict): Dictionary of loaded ML models.
        model_features_dict (dict): Dictionary of features required by each model.
        covid_data (pandas.DataFrame): The full data, needed for date calculations.
    """
    st.subheader("Scenario Analysis: Impact on Predictions")
    st.write("Adjust input parameters to create different scenarios and observe their predicted impact on **New Deaths** or **Daily Vaccinations** based on our machine learning models.")

    if not models.get('deaths') or not models.get('vaccinations'):
        st.warning("Prediction models are not loaded. Scenario analysis requires loaded models. Please ensure 'train_covid_model.py' has been run.")
        return

    # Select which prediction model to use for scenario analysis
    scenario_prediction_type = st.radio(
        "Choose Model for Scenario Analysis:",
        ('New Deaths Model', 'Daily Vaccinations Model'),
        key='scenario_model_selector',
        help="Select which model's predictions you want to analyze under different scenarios."
    )

    model_key = 'deaths' if scenario_prediction_type == 'New Deaths Model' else 'vaccinations'
    current_model = models[model_key]
    current_features = model_features_dict[model_key]
    prediction_label = f"Predicted {scenario_prediction_type.replace(' Model', '')}"
    
    st.markdown("---")
    st.write("#### Adjust Scenario Parameters:")

    # Prepare input values, using median from filtered data as a sensible default
    # Handle cases where filtered_data might be empty for a feature
    input_scenario_values = {}
    default_vals = {
        'total_vaccinations': filtered_data['total_vaccinations'].median() if 'total_vaccinations' in filtered_data.columns and not filtered_data['total_vaccinations'].isnull().all() else 1_000_000,
        'people_vaccinated': filtered_data['people_vaccinated'].median() if 'people_vaccinated' in filtered_data.columns and not filtered_data['people_vaccinated'].isnull().all() else 500_000,
        'people_fully_vaccinated': filtered_data['people_fully_vaccinated'].median() if 'people_fully_vaccinated' in filtered_data.columns and not filtered_data['people_fully_vaccinated'].isnull().all() else 250_000,
        'population': filtered_data['population'].median() if 'population' in filtered_data.columns and not filtered_data['population'].isnull().all() else 10_000_000,
        'ratio': filtered_data['ratio'].median() if 'ratio' in filtered_data.columns and not filtered_data['ratio'].isnull().all() else 0.1,
        'vaccination_coverage': filtered_data['vaccination_coverage'].median() if 'vaccination_coverage' in filtered_data.columns and not filtered_data['vaccination_coverage'].isnull().all() else 0.05,
        'New_deaths': filtered_data['New_deaths'].median() if 'New_deaths' in filtered_data.columns and not filtered_data['New_deaths'].isnull().all() else 50,
        'days_since_start': (pd.to_datetime(filtered_data['date'].max()) - pd.to_datetime(covid_data['date'].min())).days if not filtered_data['date'].empty else 365 # Approx days in a year
    }
    # Ensure default_vals are not NaN for features required by model
    for feature in current_features:
        if feature not in default_vals or pd.isna(default_vals[feature]):
            # Use a hardcoded sensible default if median is NaN or column missing
            if 'total_vaccinations' in feature: default_vals[feature] = 1_000_000
            elif 'people_vaccinated' in feature: default_vals[feature] = 500_000
            elif 'people_fully_vaccinated' in feature: default_vals[feature] = 250_000
            elif 'population' in feature: default_vals[feature] = 10_000_000
            elif 'ratio' in feature: default_vals[feature] = 0.1
            elif 'vaccination_coverage' in feature: default_vals[feature] = 0.05
            elif 'New_deaths' in feature: default_vals[feature] = 50
            elif 'days_since_start' in feature: default_vals[feature] = 365 # Default to roughly a year from start if no data

    scenario_cols = st.columns(3)
    
    # Common features across both models
    if 'population' in current_features:
        with scenario_cols[0]:
            input_scenario_values['population'] = st.slider(
                "Population:",
                min_value=1_000_000, max_value=2_000_000_000, 
                value=int(default_vals['population']),
                step=1_000_000, format="%d", key='scenario_population'
            )
    
    if 'total_vaccinations' in current_features:
        with scenario_cols[1]:
            input_scenario_values['total_vaccinations'] = st.slider(
                "Total Vaccinations:",
                min_value=0, max_value=int(default_vals['population'] * 2), # Can exceed population if multiple doses
                value=int(default_vals['total_vaccinations']),
                step=10_000, format="%d", key='scenario_total_vaccinations'
            )
    
    if 'people_fully_vaccinated' in current_features:
        with scenario_cols[2]:
            input_scenario_values['people_fully_vaccinated'] = st.slider(
                "People Fully Vaccinated:",
                min_value=0, max_value=int(input_scenario_values.get('population', default_vals['population'])),
                value=int(default_vals['people_fully_vaccinated']),
                step=10_000, format="%d", key='scenario_people_fully_vaccinated'
            )

    # Derived features based on the above sliders, ensuring consistency
    # Only calculate if parent features exist and are used by the model
    if 'vaccination_coverage' in current_features:
        pop = input_scenario_values.get('population', default_vals['population'])
        pfv = input_scenario_values.get('people_fully_vaccinated', default_vals['people_fully_vaccinated'])
        input_scenario_values['vaccination_coverage'] = pfv / pop if pop > 0 else 0.0
        # Display derived value (optional, as it's an input for model, not direct user slider)
        st.markdown(f"**Derived Vaccination Coverage:** `{input_scenario_values['vaccination_coverage']:.2%}`")


    if 'days_since_start' in current_features:
        st.markdown("---")
        st.write("#### Time-Related Parameters:")
        days_slider_col = st.columns(1)
        with days_slider_col[0]:
            input_scenario_values['days_since_start'] = st.slider(
                "Days Since Start of Data Collection:",
                min_value=0, max_value=int(default_vals['days_since_start'] + 365), # Allow extending a year
                value=int(default_vals['days_since_start']),
                step=7, key='scenario_days_since_start'
            )
            # Use covid_data here directly for calculation
            if not covid_data['date'].empty:
                st.info(f"Corresponds to a date roughly: {covid_data['date'].min().date() + pd.Timedelta(days=input_scenario_values['days_since_start'])}")
            else:
                st.info("Cannot calculate corresponding date as 'covid_data' is empty or dates are missing.")
    
    # Specific feature for Daily Vaccinations Model if selected
    if model_key == 'vaccinations' and 'New_deaths' in current_features:
        st.markdown("---")
        st.write("#### Deaths-Related Parameter (for Vaccination Model):")
        deaths_slider_col = st.columns(1)
        with deaths_slider_col[0]:
            input_scenario_values['New_deaths'] = st.slider(
                "New Deaths (Impact on Vaccinations):",
                min_value=0, max_value=5000, 
                value=int(default_vals['New_deaths']),
                step=10, format="%d", key='scenario_new_deaths'
            )


    # Create a DataFrame for prediction, ensuring all `current_features` are present
    predict_df = pd.DataFrame([input_scenario_values])
    
    # Ensure all required features are present, fill missing if any were not in sliders with their defaults
    for feature in current_features:
        if feature not in predict_df.columns:
            predict_df[feature] = default_vals.get(feature, 0.0) # Fallback to 0 or a sensible default

    # Reorder columns to match the model's expected feature order
    predict_df = predict_df[current_features]

    # Perform prediction
    with st.spinner(f"Running scenario analysis for {scenario_prediction_type.lower()}..."):
        try:
            predicted_value_transformed = current_model.predict(predict_df)[0]
            predicted_value = np.expm1(predicted_value_transformed)
            predicted_value = max(0, round(predicted_value))
            
            st.markdown("---")
            st.subheader(f"📊 Scenario Result: {prediction_label}")
            st.success(f"Under this scenario, the **{prediction_label}** is predicted to be: **{int(predicted_value):,}**")
            st.caption("*(This prediction is based on the selected model and the parameters you've adjusted. It helps explore 'what-if' scenarios.)*")

        except Exception as e:
            st.error(f"An error occurred during scenario prediction: {e}")
            st.warning("Please ensure all input parameters are within reasonable bounds for the model. If the issue persists, check the data preprocessing and model training steps.")
    st.markdown("---")
