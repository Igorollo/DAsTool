import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
from forecasting_logic import load_and_prepare_history, load_and_prepare_actuals, generate_experiment_splits
from forecasting_logic import forecast_moving_average, forecast_exponential_smoothing
from forecasting_logic import forecast_linear_regression, forecast_multiple_linear_regression
from forecasting_logic import prepare_mlr_features, run_backtest, summarize_results
import io
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
st.sidebar.title("Configuration")

# Function to create default config
def get_default_config():
    return {
        # File Paths (will be replaced with uploaded files)
        "raw_data_path": None,  # Historical weekly data
        "actuals_path": None,   # Actual monthly shipments
        "output_path": "best_forecast_models_summary.xlsx", # Output file for results

        # Data Loading Options
        "raw_data_skiprows": 2,
        "actuals_skiprows": 2,
        "history_col_name": "Actual.1", # Column name for history in raw_data_path

        # Forecasting Parameters
        "forecast_lag": 1,           # Lag in months for forecasting
        "num_experiments": 3,        # Number of rolling forecast validation periods
        "forecast_horizon_weeks": 52,# How many weeks ahead models should predict internally
        "min_weeks_in_month": 4,     # Minimum weeks to consider a month valid for splitting

        # Moving Average Parameters
        "ma_window_min": 2,
        "ma_window_max": 52,

        # Exponential Smoothing Parameters
        "es_alphas": np.arange(0, 1.02, 0.02).tolist(),

        # Multiple Linear Regression Parameters
        "mlr_trend_decay_factors": [0.05, 1.0], # 0.05 maps to Kinaxis 0, 1.0 maps to Kinaxis 1
        "mlr_seasonality_min": 2,
        "mlr_seasonality_max": 52,
        "mlr_train_weeks": 104, # Use last N weeks of training data for MLR fit

        # Evaluation & Output
        "results_top_n_params": 3, # Keep top N parameter sets per model type
    }

# Initialize or get config from session state
if 'config' not in st.session_state:
    st.session_state.config = get_default_config()

# File uploader components in sidebar
st.sidebar.subheader("Upload Data Files")
raw_data_file = st.sidebar.file_uploader("Upload Historical Weekly Data (Excel)", type=["xlsx", "xls"], key="raw_data")
actuals_file = st.sidebar.file_uploader("Upload Actual Monthly Shipments (Excel)", type=["xlsx", "xls"], key="actuals")

# Allow customization of main configuration parameters
st.sidebar.subheader("Forecasting Parameters")
st.session_state.config["forecast_lag"] = st.sidebar.number_input(
    "Forecast Lag (months)", 
    min_value=1, 
    max_value=12, 
    value=st.session_state.config["forecast_lag"]
)
st.session_state.config["num_experiments"] = st.sidebar.number_input(
    "Number of Experiments", 
    min_value=1, 
    max_value=10, 
    value=st.session_state.config["num_experiments"]
)
st.session_state.config["results_top_n_params"] = st.sidebar.number_input(
    "Top N Results to Show", 
    min_value=1, 
    max_value=10, 
    value=st.session_state.config["results_top_n_params"]
)

# Data configuration
st.sidebar.subheader("Data Configuration")
st.session_state.config["raw_data_skiprows"] = st.sidebar.number_input(
    "Skip Rows in Historical Data", 
    min_value=0, 
    max_value=10, 
    value=st.session_state.config["raw_data_skiprows"]
)
st.session_state.config["actuals_skiprows"] = st.sidebar.number_input(
    "Skip Rows in Actuals Data", 
    min_value=0, 
    max_value=10, 
    value=st.session_state.config["actuals_skiprows"]
)
st.session_state.config["history_col_name"] = st.sidebar.text_input(
    "History Column Name", 
    value=st.session_state.config["history_col_name"]
)

# Advanced configuration expander
with st.sidebar.expander("Advanced Configuration"):
    st.session_state.config["ma_window_min"] = st.number_input(
        "Min Moving Average Window", 
        min_value=1, 
        max_value=20, 
        value=st.session_state.config["ma_window_min"]
    )
    st.session_state.config["ma_window_max"] = st.number_input(
        "Max Moving Average Window", 
        min_value=st.session_state.config["ma_window_min"], 
        max_value=100, 
        value=st.session_state.config["ma_window_max"]
    )
    
    st.session_state.config["mlr_seasonality_min"] = st.number_input(
        "Min MLR Seasonality", 
        min_value=1, 
        max_value=20, 
        value=st.session_state.config["mlr_seasonality_min"]
    )
    st.session_state.config["mlr_seasonality_max"] = st.number_input(
        "Max MLR Seasonality", 
        min_value=st.session_state.config["mlr_seasonality_min"], 
        max_value=100, 
        value=st.session_state.config["mlr_seasonality_max"]
    )
    
    st.session_state.config["mlr_train_weeks"] = st.number_input(
        "MLR Training Weeks", 
        min_value=10, 
        max_value=500, 
        value=st.session_state.config["mlr_train_weeks"]
    )

# Main content area
st.title("Demand Forecasting Model Comparison Tool")
st.markdown("""
This tool evaluates different time-series forecasting models (Moving Average, 
Exponential Smoothing, Linear Regression, Multiple Linear Regression) to find 
the best fit for demand forecasting based on historical weekly data.

**Instructions:**
1. Upload your historical weekly data and monthly actuals files using the sidebar
2. Adjust configuration parameters as needed
3. Click 'Run Forecasting' to begin the analysis
4. View and download the results
""")

# Function to save uploaded files to temporary files and update config
def save_uploaded_files():
    config = st.session_state.config
    temp_files = []
    
    if raw_data_file is not None:
        temp_raw_data = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_raw_data.write(raw_data_file.getvalue())
        config["raw_data_path"] = temp_raw_data.name
        temp_files.append(temp_raw_data.name)
        st.session_state.raw_data_name = raw_data_file.name
    
    if actuals_file is not None:
        temp_actuals = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_actuals.write(actuals_file.getvalue())
        config["actuals_path"] = temp_actuals.name
        temp_files.append(temp_actuals.name)
        st.session_state.actuals_name = actuals_file.name
    
    return temp_files

# Function to run the forecasting process
def run_forecasting():
    temp_files = save_uploaded_files()
    config = st.session_state.config
    
    try:
        with st.spinner('Running backtesting process...'):
            # Run the entire backtesting process
            raw_results = run_backtest(config)
            
            if not raw_results.empty:
                # Summarize the results
                summary_df = summarize_results(raw_results, config)
                
                # Fix potential duplicate column issue by renaming columns
                if not summary_df.empty:
                    # Get list of duplicate columns
                    cols = summary_df.columns.tolist()
                    duplicates = set([x for x in cols if cols.count(x) > 1])
                    
                    # Rename duplicates with a suffix
                    if duplicates:
                        for dup in duplicates:
                            # Find all indices of the duplicate column
                            indices = [i for i, x in enumerate(cols) if x == dup]
                            
                            # Rename all but the first occurrence
                            for i, idx in enumerate(indices[1:], 1):
                                new_name = f"{dup}_{i}"
                                cols[idx] = new_name
                        
                        # Apply the new column names
                        summary_df.columns = cols
                
                # Store results in session state
                st.session_state.raw_results = raw_results
                st.session_state.summary_results = summary_df
                st.session_state.forecast_success = True
                
                # Clean up temp files after successful run
                # for file in temp_files:
                #     if os.path.exists(file):
                #         os.unlink(file)
                        
                return True
            else:
                st.error("Backtesting generated no results. Please check your data and configuration.")
                return False
                
    except FileNotFoundError as fnf_error:
        st.error(f"File Not Found Error: {fnf_error}. Please check uploaded files.")
    except ValueError as val_error:
        st.error(f"Value Error: {val_error}. Check configuration or data consistency.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    
    # Clean up temp files after error
    for file in temp_files:
        if os.path.exists(file):
            os.unlink(file)
    
    return False

# Check if both files are uploaded
files_ready = raw_data_file is not None and actuals_file is not None

# Start forecasting button
if st.button('Run Forecasting', disabled=not files_ready):
    success = run_forecasting()
    if success:
        st.success("Forecasting completed successfully!")

# Display warning if files are missing
if not files_ready:
    st.warning("Please upload both data files to run the forecasting process.")

# Display results if available
if 'forecast_success' in st.session_state and st.session_state.forecast_success:
    st.header("Forecasting Results")
    
    # Show the summary results
    st.subheader("Top Model Configurations")
    st.dataframe(st.session_state.summary_results, use_container_width=True)
    
    # Create plots
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot average MAPE by model type
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        model_mape = st.session_state.summary_results.groupby('model')['Average_MAPE'].mean().reset_index()
        model_mape = model_mape.sort_values('Average_MAPE')
        sns.barplot(x='model', y='Average_MAPE', data=model_mape, ax=ax1)
        ax1.set_title('Average MAPE by Model Type')
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Average MAPE')
        st.pyplot(fig1)
    
    with col2:
        # Plot best model per item
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        best_per_item = st.session_state.summary_results.loc[st.session_state.summary_results.groupby('Forecast Item')['Average_MAPE'].idxmin()]
        sns.barplot(x='Forecast Item', y='Average_MAPE', hue='model', data=best_per_item, ax=ax2)
        ax2.set_title('Best Model Performance per Item')
        ax2.set_xlabel('Forecast Item')
        ax2.set_ylabel('Average MAPE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Create Excel download button
    st.subheader("Download Results")
    
    # Function to create Excel with both raw and summary data
    def create_excel():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.summary_results.to_excel(writer, sheet_name='Summary Results', index=False)
            st.session_state.raw_results.to_excel(writer, sheet_name='Detailed Results', index=False)
        return output.getvalue()
    
    excel_file = create_excel()
    st.download_button(
        label="Download Excel Results",
        data=excel_file,
        file_name="forecast_models_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Footer
st.markdown("---")
st.caption("Demand Forecasting Model Comparison Tool")
