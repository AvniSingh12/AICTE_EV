# AICTE_EV
EV Vehicle Demand Prediction

# EV Adoption Forecasting for Washington State
This project offers a data-intensive solution to predict electric vehicle (EV) adoption on a county-by-county basis for Washington State. By applying a machine learning model, it reads through historical registration data to forecast future patterns, allowing policymakers, urban planners, and businesses to make informed decisions about infrastructure investment and resource allocations.

The heart of the project is a Python forecasting pipeline consisting of data cleaning, feature engineering, model training, and an interactive web application using Streamlit.

# Features
County-Level Forecasting: Forecasts EV adoption (number of EVs) for individual counties.
Interactive Web Application: An easy-to-use interface designed with Streamlit for dynamic forecasting and visualization.
Historical & Forecasted Trends: Shows past and future EV adoption trends on easy-to-understand charts.
Multi-County Comparison: Enables users to compare EV growth patterns between up to three various counties.
Data-Driven Insights: Determines the most significant factors driving EV uptake using feature importance analysis.
Model Persistence: Saves the trained model and loads it easily for quick predictions without re-training.

# Project Structure

├── EV_prediction.ipynb                              # Jupyter Notebook containing the complete data science workflow
├── Electric_Vehicle_Population_Countrywise.csv      # Raw data file
├── preprocessed_ev_data.csv                         # Cleaned and feature-engineered data
├── forecasting_ev_model.pkl                         # The saved trained machine learning model
├── app.py                                           # Streamlit web application
└── README.md                                        # This document

# Methodology 
Data Preprocessing & Feature Engineering
The raw data was heavily cleaned and feature engineered in preparation for a time-series forecasting model. Important features were derived in order to capture temporal relations and growth trends:

Lag Features: EV sums for the last 1, 2, and 3 months.
Rolling Averages: A rolling mean of 3 months to reduce noise in the data.
Growth Metrics: 1-month and 3-month percentage growth and a 6-month growth slope.
Categorical Encoding: Counties were converted to numerical representations.

**Model**
A Random Forest Regressor was used due to its capacity to fit non-linear relationships and its resistance to overfitting. RandomizedSearchCV was used for optimization in order to determine the optimal hyperparameters.

# Evaluation
The model performed very well on the test set with a high R squared Score of 1.00 and a MAE of 0.01, reflecting a highly accurate model and excellent predictive strength.

# Key Findings
Top Predictors: The model's predictions are mainly dictated by lagged EV totals (ev_total_lag1, ev_total_lag2) and the ev_growth_slope, pointing to the significance of recent past and growth momentum.

County-Specific Trends: The projection exhibits high degree of variance of EV adoption across counties, with leaders such as Santa Clara and Fairfax likely to continue robust growth.

Informed Planning: The visualizations offer intuitive insights which can inform strategic deployment of charging infrastructure and development of local policy.

# Acknowledgements
The project was done under the AICTE Internship by S4F.
Special thanks to the developers of the libraries used: Pandas, Scikit-learn, and Streamlit
