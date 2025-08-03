import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load model ===
model = joblib.load('forecasting_ev_model.pkl')

def run_forecast(model, county_df, county_code, latest_date, months_since_start, forecast_horizon=36):
    # Get the last 6 months of historical EV totals
    initial_historical_ev_window = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    
    # Calculate the cumulative sum for the historical window for slope calculation
    # This also needs to be a running sum of the actual EV totals, not just predictions.
    # The 'cumulative_ev' for slope is the cumulative sum of the 'ev_total' (historical + predicted)
    # in the rolling window.
    
    # Start the running cumulative total from the last known historical cumulative
    last_historical_cumulative_total = county_df['Electric Vehicle (EV) Total'].sum()

    # The list 'current_ev_window' will hold the last 6 'Electric Vehicle (EV) Total' (actual or predicted)
    # for calculating lags, rolling mean, and percentage changes.
    current_ev_window = list(initial_historical_ev_window)

    # The list 'current_cumulative_window_for_slope' will hold the *cumulative* sum of EVs
    # for the last 6 periods, crucial for the 'ev_growth_slope'.
    # Initialize it with the cumulative sums of the last 6 historical periods.
    # This might require a bit more context if `county_df` doesn't have a direct cumulative column easily accessible.
    # For simplicity, let's derive it here based on the `initial_historical_ev_window`:
    
    temp_cumulative_for_slope = []
    current_sum = last_historical_cumulative_total - sum(county_df['Electric Vehicle (EV) Total'].values[-(len(county_df) - 6):]) if len(county_df) > 6 else 0 # Starting point for the 6-month window
    for val in initial_historical_ev_window:
        current_sum += val
        temp_cumulative_for_slope.append(current_sum)
    current_cumulative_window_for_slope = temp_cumulative_for_slope[-6:] # Ensure it's the last 6 cumulative sums


    future_rows = []

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        
        # Ensure enough values for lags. If fewer than 3, use 0 or handle accordingly.
        lag1 = current_ev_window[-1] if len(current_ev_window) >= 1 else 0
        lag2 = current_ev_window[-2] if len(current_ev_window) >= 2 else 0
        lag3 = current_ev_window[-3] if len(current_ev_window) >= 3 else 0

        roll_mean = np.mean([lag1, lag2, lag3]) if len(current_ev_window) >= 3 else (lag1 + lag2 + lag3) / max(1, len(current_ev_window)) # Handle cases with less than 3
        
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        
        # Calculate slope based on the *cumulative values within the rolling window*
        # Ensure current_cumulative_window_for_slope always has 6 values before calling polyfit
        ev_growth_slope = 0
        if len(current_cumulative_window_for_slope) == 6:
            ev_growth_slope = np.polyfit(range(len(current_cumulative_window_for_slope)), current_cumulative_window_for_slope, 1)[0]
        # else: handle if there are fewer than 6 cumulative points for slope calculation (e.g., in very early periods of the historical data if it's short)

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        predicted_ev_total_current_month = max(0, round(pred)) # Ensure predictions are not negative
        future_rows.append({"Date": forecast_date, "Predicted EV Total": predicted_ev_total_current_month})

        # Update the rolling window for next iteration's lags/mean/pct_change
        current_ev_window.append(predicted_ev_total_current_month)
        if len(current_ev_window) > 6:
            current_ev_window.pop(0)

        # Update the rolling cumulative window for next iteration's slope
        last_historical_cumulative_total += predicted_ev_total_current_month # Update the running total
        current_cumulative_window_for_slope.append(last_historical_cumulative_total)
        if len(current_cumulative_window_for_slope) > 6:
            current_cumulative_window_for_slope.pop(0)

    forecast_df = pd.DataFrame(future_rows)
    # The cumulative sum for the final plot should start from the absolute historical cumulative total
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + county_df['Electric Vehicle (EV) Total'].sum()
    
    return forecast_df

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: #fcfcfc;
            color: #000000;
        }
        .stApp {
            background: linear-gradient(to right, #8fdbcc, #6eb5a7);
        }
    </style>
""", unsafe_allow_html=True)

# Display image after config and styles
# Stylized title using markdown + HTML
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

# Welcome subtitle
st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

# Image
st.image("ev-car.jpg", use_container_width=True)

# Instruction line
st.markdown("""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: #FFFFFF;'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
""", unsafe_allow_html=True)


# === Load data (must contain historical values, features, etc.) ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === County dropdown ===
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# === Combine Historical + Forecast for Cumulative Plot ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)


# === New: Plot Monthly Forecast ===
st.subheader(f"📈 Monthly EV Adoption Forecast for {county} County")

# Prepare data for monthly plot
historical_monthly = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_monthly['Source'] = 'Historical'
historical_monthly = historical_monthly.rename(columns={'Electric Vehicle (EV) Total': 'EV Total'})

forecast_monthly = forecast_df.copy()
forecast_monthly['Source'] = 'Forecast'
forecast_monthly = forecast_monthly.rename(columns={'Predicted EV Total': 'EV Total'})

combined_monthly = pd.concat([
    historical_monthly,
    forecast_monthly
], ignore_index=True)

fig_monthly, ax_monthly = plt.subplots(figsize=(12, 6))
# Plot historical data in one color
ax_monthly.plot(combined_monthly[combined_monthly['Source'] == 'Historical']['Date'],
                combined_monthly[combined_monthly['Source'] == 'Historical']['EV Total'],
                label='Historical Monthly EV', marker='o', color='skyblue')
# Plot forecasted data in another color
ax_monthly.plot(combined_monthly[combined_monthly['Source'] == 'Forecast']['Date'],
                combined_monthly[combined_monthly['Source'] == 'Forecast']['EV Total'],
                label='Forecasted Monthly EV', marker='o', linestyle='--', color='lightcoral')

ax_monthly.set_title(f"Monthly EV Total Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
ax_monthly.set_xlabel("Date", color='white')
ax_monthly.set_ylabel("Monthly EV Count", color='white')
ax_monthly.grid(True, alpha=0.3)
ax_monthly.set_facecolor("#1c1c1c")
fig_monthly.patch.set_facecolor('#1c1c1c')
ax_monthly.tick_params(colors='white')
ax_monthly.legend()
st.pyplot(fig_monthly)
st.download_button(
    label="Download Forecasted Monthly EV Data (CSV)",
    data=forecast_df.to_csv(index=False).encode('utf-8'),
    file_name=f"{county}_EV_Monthly_Forecast.csv",
    mime="text/csv",
    help="Download the predicted monthly EV totals for the selected county."
)

st.markdown(f"""
    <div style='text-align: left; font-size: 18px; padding-top: 10px; color: #FFFFFF;'>
        The chart above shows the predicted number of new EV registrations each month.
    </div>
    """, unsafe_allow_html=True) 

# === Plot Cumulative Graph ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend(title="Source", facecolor='#1c1c1c', edgecolor='white', labelcolor='white') # For single plot
st.pyplot(fig)

# === Compare historical and forecasted cumulative EVs ===
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(f"Based on the graph, EV adoption in **{county}** is expected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
else:
    st.warning("Historical EV total is zero, so percentage forecast change can't be computed.")


# === New: Compare up to 3 counties ===
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]

        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)

            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    # Combine all counties data for plotting
    comp_df = pd.concat(comparison_data, ignore_index=True)

    # Plot
    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County", facecolor='#1c1c1c', edgecolor='white', labelcolor='white')
    st.pyplot(fig)
    
    # Display % growth for selected counties ===
    growth_summaries = []
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecasted_total = cty_df['Cumulative EV'].iloc[-1]

        if historical_total > 0:
            growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A (no historical data)")

    # Join all in one sentence and show with st.success
    growth_sentence = " | ".join(growth_summaries)
    st.success(f"Forecasted EV adoption growth over next 3 years — {growth_sentence}")

st.success("Forecast complete")


st.markdown("---")
st.header("About This Application")
st.info("""
This app forecasts Electric Vehicle (EV) adoption within several counties of Washington State.
It employs a pre-trained machine learning model (`forecasting_ev_model.pkl`) to predict future EV counts
based on historical data and engineered features such as lagged values, rolling means, and growth slopes.

**Data Source:** The historical data comes from the Washington State Department of Licensing EV registrations.
The model considers features such as `months_since_start`, `county_encoded`, and different lagged EV counts 
into account to capture county-specific patterns and time-series dependencies.

**Forecasting Methodology:** The model is trained from historical EV registration data for months. For the 
purpose of making predictions into the future, it continuously generates new features from its own past 
predictions, enabling the possibility of rolling forecast over a 3-year (36-month) horizon.

**Limitations:**
* The accuracy of the forecast relies to a great extent on the patterns and quality of the historical data.
* Unanticipated external forces (e.g., large policy shifts, economic transformation, emergence of new EV models, 
    development of charging infrastructure)
   not reflected in the historical attributes might influence future adoption rates.
* The model forecasts new registrations and does not include EVs exiting the county or being decommissioned.
""")


st.markdown("---")
st.markdown("App developed by **Avni Singh**  , for the **AICTE Internship by S4F**  ")

