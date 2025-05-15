# -*- coding: utf-8 -*-
"""
Simplified Streamlit GUI for Demand Forecasting

Provides an interface to upload weekly data, run the simplified forecasting
logic, and view/download the best model results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import sim_forecasting_logic as sim_logic # Import the simplified logic
# Force reload the module to ensure latest changes are used
importlib.reload(sim_logic)
import logging
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# --- Page Configuration ---
st.set_page_config(page_title="Demand Forecast Comparison", layout="wide")
st.title("📊 Demand Forecast Comparison")
st.markdown("""Upload your **weekly** historical demand data (Excel format) to find the best forecasting model.

The analysis uses a 70/30 train/test split and compares Moving Average, Exponential Smoothing, Linear Regression, MLR, Croston, and ARIMA models to identify the best fit for each item based on the selected error metric.""")

# --- Session State Initialization ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0 # Key to reset file uploader
if 'best_forecasts' not in st.session_state:
    st.session_state.best_forecasts = None
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'selected_result_item' not in st.session_state:
    # remembers the choice in the Results filter
    st.session_state.selected_result_item = 'All Items'
# --- Helper Functions ---
def display_error(message):
    """Displays an error message in the Streamlit app."""
    st.error(f"🚨 Error: {message}")
    st.session_state.error_message = message
    st.session_state.results_df = None # Clear results on error

def display_success(message):
    """Displays a success message."""
    st.success(f"✅ {message}")

def plot_forecast(item_name, forecast_data):
    """Generate a plot of historical data and the best model forecast."""
    if not forecast_data:
        st.warning(f"No forecast data available for {item_name}")
        return
    
    try:
        # Extract data
        train_data = forecast_data['train_data']
        test_data = forecast_data['test_data']
        forecast = forecast_data['forecast']
        model_name = forecast_data['model']
        error_value = forecast_data['error']
        error_metric = forecast_data['error_metric']
        params = forecast_data['params']
        
        # Debug info - can be removed in production
        # st.write(f"Train data shape: {train_data.shape if hasattr(train_data, 'shape') else 'N/A'}")
        # st.write(f"Test data shape: {test_data.shape if hasattr(test_data, 'shape') else 'N/A'}")
        # st.write(f"Forecast data shape: {forecast.shape if hasattr(forecast, 'shape') else 'N/A'}")
        
        # Create a clean parameter string
        param_str = ", ".join([f"{k.replace('param_', '')}: {v}" for k, v in params.items()])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        if not train_data.empty:
            ax.plot(train_data.index, train_data.values, 'b-', label='Training Data')
        if not test_data.empty:
            ax.plot(test_data.index, test_data.values, 'g-', label='Test Data')
        if not forecast.empty:
            ax.plot(forecast.index, forecast.values, 'r--', label='Forecast')
        
        # Set title and labels
        ax.set_title(f"{item_name} - {model_name} ({error_metric.upper()}: {error_value:.2f})\n{param_str}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        st.write("Forecast data keys:", list(forecast_data.keys()))
        return None

def run_analysis(uploaded_file, sheet_name, skiprows, date_col, value_col, item_col, fill_missing_weeks_flag=True, skip_leading_zeros=False, show_visualization=True, error_metric='smape'):
    """Wrapper function to run the forecasting logic and handle errors."""
    st.session_state.results_df = None
    st.session_state.error_message = None
    st.session_state.best_forecasts = None
    output_buffer = io.BytesIO() # Use BytesIO for in-memory Excel file

    try:
        with st.spinner('Analyzing data and finding best models... Please wait.'):
            # Ensure item_col is None if the checkbox isn't selected or field is empty
            effective_item_col = item_col if item_col else None
            print(f"Effective item column: {effective_item_col}")
            # Call the simplified logic function with forecast data for visualization
            results, best_forecasts = sim_logic.run_simplified_forecast(
                file_path=uploaded_file, # Pass the uploaded file object directly
                sheet_name=sheet_name if sheet_name else 0, # Default to first sheet if empty
                skiprows=skiprows,
                date_col=date_col,
                value_col=value_col,
                item_col=effective_item_col,
                output_path=output_buffer, # Save to buffer instead of file
                model_params=sim_logic.DEFAULT_PARAMS, # Use defaults for now
                fill_missing_weeks_flag=fill_missing_weeks_flag,
                skip_leading_zeros=skip_leading_zeros,
                return_best_forecasts=show_visualization,
                error_metric=error_metric
            )

        if results is not None and not results.empty:
            st.session_state.results_df = results
            st.session_state.best_forecasts = best_forecasts
            
            # Set the selected item to the first item if we have forecasts
            if best_forecasts and len(best_forecasts) > 0:
                st.session_state.selected_item = list(best_forecasts.keys())[0]
                
            display_success("Analysis complete! Best models identified.")
        elif results is not None and results.empty:
             st.session_state.error_message = "Analysis completed, but no models could be successfully generated or evaluated. Check data quality and length."
             st.warning(st.session_state.error_message)
        # else: # The logic inside run_simplified_forecast should handle errors/empty cases
        #     st.session_state.error_message = "Analysis failed to produce results. Check logs or input data."
        #     st.warning(st.session_state.error_message)

    except FileNotFoundError:
        display_error("Input file not found during processing. This should not happen with uploads.")
    except KeyError as e:
        display_error(f"Column name mismatch during processing: '{e}'. Please verify the column names entered below match your Excel file exactly.")
    except ValueError as e:
         display_error(f"Data type or value error during processing: {e}. Ensure date/value columns have correct data.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        display_error(f"An unexpected error occurred: {e}. Check the format and content of your data file.")

# --- Sidebar: File Upload and Configuration ---
st.sidebar.header("⚙️ Configuration")

# Data processing options - moved to top level for visibility
st.sidebar.markdown("**Data Processing Options:**")
fill_missing_weeks_flag = st.sidebar.checkbox(
    "Fill missing weeks with zeros",
    value=True,
    help="If checked, any missing weekly dates in your data will be filled with a value of 0. If unchecked, missing weeks will be omitted (no zeros inserted)."
)
st.sidebar.caption(
    "If you skip this option, the forecasting will only use the weeks present in your data. No zeros will be inserted for missing weeks. This is a default option in RR."
)

skip_leading_zeros = st.sidebar.checkbox(
    "Skip leading zeros in historical data",
    value=True,
    help="If checked, leading zeros in the historical data will be skipped before fitting the models."
)
st.sidebar.caption(
    "This option removes initial leading zeros from the historical data before fitting models."
)

uploaded_file = st.sidebar.file_uploader(
    "1. Upload Weekly Data (Excel)",
    type=["xlsx"],
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

if uploaded_file:
    st.sidebar.info(f"File uploaded: `{uploaded_file.name}`")

    st.sidebar.markdown("**2. Specify Data Details:**")
    sheet_name = st.sidebar.text_input("Sheet Name (optional, defaults to first sheet)", placeholder="e.g., Outliers")
    skiprows = st.sidebar.number_input("Rows to Skip at Top", min_value=0, value=2, step=1)

    st.sidebar.markdown("**Column Names (Case-Sensitive):**")
    date_col = st.sidebar.text_input("Date Column Name", value="Date", placeholder="Enter exact column name")
    value_col = st.sidebar.text_input("Value Column Name", value="Actual.1", placeholder="Enter exact column name")
    has_item_col = st.sidebar.checkbox("My data has a 'Forecast Item' column", value=True)
    item_col = st.sidebar.text_input("Item Column Name", value="Forecast Item", placeholder="Enter exact item column name", disabled=not has_item_col)

    # Ensure item_col is effectively None if checkbox is unchecked
    if not has_item_col:
        item_col = None
        
    # Add visualization option
    show_visualization = st.sidebar.checkbox("Show visualization of best models", value=True)
    
    # Add error metric selection BEFORE running the analysis
    st.sidebar.markdown("**Error Metric for Model Selection:**")
    error_metric = st.sidebar.selectbox(
        "Select error metric",
        options=['smape', 'mape', 'rmse'],
        index=0,
        help="SMAPE: Symmetric Mean Absolute Percentage Error (balanced)\n"
             "MAPE: Mean Absolute Percentage Error (traditional)\n"
             "RMSE: Root Mean Square Error (penalizes large errors more)"
    )

    run_button = st.sidebar.button("🚀 Run Analysis", type="primary")

    if run_button:
        if not date_col or not value_col or (has_item_col and not item_col):
            st.sidebar.warning("Please fill in all required column names.")
        else:
            run_analysis(uploaded_file, sheet_name, skiprows, date_col, value_col, item_col, 
                       fill_missing_weeks_flag, skip_leading_zeros, show_visualization, error_metric)
else:
    st.sidebar.warning("Please upload an Excel file containing your weekly demand data.")

# Add a button to clear results and reset the file uploader 
if st.sidebar.button("Clear Results & Reset"):
    st.session_state.results_df = None
    st.session_state.error_message = None
    st.session_state.best_forecasts = None
    st.session_state.selected_item = None
    st.session_state.file_uploader_key += 1 # Increment key to force re-render
    st.rerun()

# --- Main Area: Display Results ---
st.header("📈 Results")

if st.session_state.error_message:
    st.warning(st.session_state.error_message) # Show persistent error if one occurred

if st.session_state.results_df is not None and not st.session_state.results_df.empty:
    results = st.session_state.results_df

    # ────────────────────────────────────────────────
    # NEW ▸ let the user filter the summary table
    # ────────────────────────────────────────────────
    if 'Forecast Item' in results.columns and results['Forecast Item'].nunique() > 1:
        items = sorted(results['Forecast Item'].unique())
        item_options = ['All Items'] + items

        selected_result_item = st.selectbox(
            "Filter by Forecast Item",
            item_options,
            index=item_options.index(st.session_state.selected_result_item)
            if st.session_state.selected_result_item in item_options else 0
        )
        st.session_state.selected_result_item = selected_result_item

        if selected_result_item != 'All Items':
            results_to_show = results[results['Forecast Item'] == selected_result_item]
        else:
            results_to_show = results
    else:
        # Single item → no selector needed
        results_to_show = results

    st.dataframe(results_to_show)
    
    # Visualization section
    if st.session_state.best_forecasts and len(st.session_state.best_forecasts) > 0:
        st.header("📊 Visualization")
        st.markdown("View the historical data and best model forecast for each item.")
        
        # Item selector
        items = list(st.session_state.best_forecasts.keys())
        selected_item = st.selectbox(
            "Select Item to Visualize", 
            items,
            index=items.index(st.session_state.selected_item) if st.session_state.selected_item in items else 0
        )
        st.session_state.selected_item = selected_item
        
        # Plot the selected item's forecast
        if selected_item in st.session_state.best_forecasts:
            forecast_data = st.session_state.best_forecasts[selected_item]
            fig = plot_forecast(selected_item, forecast_data)
            if fig:
                st.pyplot(fig)

    # Provide download link for the results
    # Need to re-run the logic to get the buffer if not stored, or store the buffer
    # For simplicity, let's re-run quickly to get the excel bytes
    @st.cache_data # Cache the generation of download data
    def get_excel_download(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Best_Models')
        return output.getvalue()

    excel_bytes = get_excel_download(results_to_show)

    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_bytes,
        file_name='simplified_best_models_summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
elif not uploaded_file:
    st.info("Upload a data file and configure parameters in the sidebar to see results.")
elif uploaded_file and not run_button and st.session_state.results_df is None and st.session_state.error_message is None:
     st.info("Click 'Run Analysis' in the sidebar after configuring your data.")
# If results are None but no error, it means analysis hasn't run or finished yet.
# If results are empty DataFrame, the logic inside run_analysis handles the message.
