# 🦠 COVID-19 Vaccination & Mortality Dashboard

This project is an interactive Streamlit dashboard designed to explore and predict trends in COVID-19 vaccination efforts and their relationship with mortality rates. It leverages a dataset containing vaccination counts, population figures, and new death counts across various countries over time.

## ✨ Features

-   **Interactive Data Filtering:** Filter data by country and date range to focus on specific regions and periods.
-   **Dynamic Visualizations:** Explore time-series trends (New Deaths, Total Vaccinations), distributions of key metrics, and correlation heatmaps. All graphs dynamically update based on applied filters.
-   **Predictive Modeling:** Utilize a trained Machine Learning Regression model (`RandomForestRegressor`) to predict `New_deaths` based on user-input vaccination and population metrics.
-   **Model Insights:** View feature importances, indicating which factors are most influential in the death prediction.
-   **Model Performance:** Assess the model's accuracy using R-squared and Root Mean Squared Error (RMSE).

## 🚀 Technologies Used

* **Python 3.x**
* **pandas:** For efficient data loading, cleaning, and manipulation.
* **scikit-learn:** For building and evaluating the `RandomForestRegressor` machine learning model.
* **Streamlit:** For creating the interactive and visually appealing web dashboard.
* **matplotlib & seaborn:** For generating comprehensive data visualizations.
* **joblib:** For saving and loading the trained machine learning model and its features.
* **numpy:** For numerical operations, including data transformation (e.g., `log1p`, `expm1`).

## ⚙️ How It Works

The dashboard operates in two main parts:

1.  **Descriptive Analysis:** Users can apply filters to the dataset (country, date range) and observe how various metrics (vaccinations, new deaths) change over time and how they are distributed. Correlation heatmaps provide insights into relationships between variables.
2.  **Predictive Modeling:** A pre-trained `RandomForestRegressor` model takes user-defined inputs (e.g., total vaccinations, population, vaccination coverage) and predicts the estimated number of new deaths. The model's feature importance and performance metrics (R-squared, RMSE) are also displayed for transparency.

## 📦 Project Structure


covid_dashboard_project/
├── covid_vaccination_mortality.csv # Your dataset
├── train_covid_model.py            # Script to train and save the ML model
├── covid_dashboard_app.py          # Main Streamlit dashboard application
├── trained_covid_model.pkl         # Saved RandomForestRegressor model
├── model_features.pkl              # Saved list of features used by the model
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project README file


## ▶️ How to Run Locally

Follow these steps to get the dashboard running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/covid_dashboard_project.git](https://github.com/KhumaloAugustine/covid_dashboard_project.git)
    cd covid_dashboard_project
    ```

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv_covid
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv_covid\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv_covid/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Train the Machine Learning Model:**
    This crucial step creates the `trained_covid_model.pkl` and `model_features.pkl` files.
    ```bash
    python train_covid_model.py
    ```
    You should see output confirming the model training and saving.

6.  **Run the Streamlit Dashboard:**
    ```bash
    streamlit run covid_dashboard_app.py
    ```
    Your default web browser should automatically open a new tab with the dashboard (usually at `http://localhost:8501`).

## 🌐 Live Demo

[Link to your deployed Streamlit App will go here once deployed, e.g., `https://yourusername-covid-dashboard.streamlit.app/`]

## ✍️ Author

* **Augustine Khumalo** - [LinkedIn Profile](https://www.linkedin.com/in/augustine-khumalo)

## 🤝 Contributing

Feel free to fork this repository, suggest improvements, or open issues. Any contributions are welcome!
