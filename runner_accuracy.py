# main_runner.py
# -*- coding: utf-8 -*-
"""
Main script to run the simplified demand forecasting pipeline
and calculate average accuracies per item for different internal error metrics.
"""

import pandas as pd
import numpy as np
import logging

# Assuming your provided code is in 'forecasting_logic.py'
from forecasting_logic import run_simplified_forecast, DEFAULT_PARAMS, load_monthly_data

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_and_print_average_accuracies(results_df: pd.DataFrame, internal_metric_name: str):
    """
    Calculates and prints the average 'Forecast Accuracy (1 - |F-A|/|A|)'
    for each Forecast Item from the results DataFrame.
    """
    if results_df.empty:
        logging.warning(f"No results to analyze for internal metric: {internal_metric_name}.")
        return

    # Ensure the accuracy column is numeric, coercing errors to NaN
    results_df['Forecast Accuracy (1 - |F-A|/|A|)'] = pd.to_numeric(
        results_df['Forecast Accuracy (1 - |F-A|/|A|)'], errors='coerce'
    )

    # Calculate average accuracy, ignoring NaNs
    average_accuracies = results_df.groupby('Forecast Item')['Forecast Accuracy (1 - |F-A|/|A|)'].mean()

    logging.info(f"\n--- Average Forecast Accuracy (1 - |F-A|/|A|) per Item (using {internal_metric_name} for model selection) ---")
    if average_accuracies.empty:
        logging.info("No average accuracies could be calculated.")
    else:
        for item, avg_acc in average_accuracies.items():
            logging.info(f"Forecast Item: {item}, Average Accuracy: {avg_acc:.4f}")
    logging.info("------------------------------------------------------------------------------------\n")
    return average_accuracies

def main():
    """
    Main function to execute the forecasting pipeline.
    """
    # --- User-Defined Parameters ---
    # !!! REPLACE 'your_data_file.xlsx' WITH THE ACTUAL PATH TO YOUR DATA FILE !!!
    input_file_path = 'your_data_file.xlsx'
    # output_base_name = "rolling_forecast_accuracy" # Base for output file names

    # Parameters for load_monthly_data as requested
    sheet_name_param = 0         # Default sheet
    skiprows_param = 2
    date_col_param = 'Date'
    value_col_param = 'Actual.1'
    item_col_param = 'Forecast Item'
    fill_missing_months_param = True # User asked for 'fill_missing_weeks_with_zero', adapted to monthly
    skip_leading_zeros_param = True

    # Error metrics to test for internal model selection
    internal_error_metrics_to_test = ['smape', 'mape', 'rmse']
    all_average_accuracies = {}

    # First, try to load the data to see if the file and basic columns are okay
    try:
        logging.info(f"Attempting to pre-load data from: {input_file_path} with specified columns to check validity.")
        # We call load_monthly_data directly here just to catch early errors
        # The actual data loading for the forecast happens within run_simplified_forecast
        _ = load_monthly_data(
            file_path=input_file_path,
            sheet_name=sheet_name_param,
            skiprows=skiprows_param,
            date_col=date_col_param,
            value_col=value_col_param,
            item_col=item_col_param,
            fill_missing_months_flag=fill_missing_months_param,
            skip_leading_zeros=skip_leading_zeros_param
        )
        logging.info("Pre-load check successful. Basic file and column configuration seems OK.")
    except FileNotFoundError:
        logging.error(f"CRITICAL: Input data file not found at '{input_file_path}'. Please specify the correct path.")
        return
    except KeyError as e:
        logging.error(f"CRITICAL: Column name error during pre-load check: {e}. Please verify 'date_col', 'value_col', 'item_col'.")
        return
    except Exception as e:
        logging.error(f"CRITICAL: An unexpected error occurred during data pre-load check: {e}")
        return

    for metric in internal_error_metrics_to_test:
        logging.info(f"\n===== Running pipeline with INTERNAL error metric for model selection: {metric.upper()} =====")
        
        output_file_name = f"rolling_forecast_accuracy_{metric}.xlsx"

        # Run the main forecasting process
        # `run_simplified_forecast` uses the `error_metric` for its internal `find_best_model` call.
        # The "Forecast Accuracy (1 - |F-A|/|A|)" in its output is a separate, fixed accuracy calculation.
        results_df = run_simplified_forecast(
            file_path=input_file_path,
            sheet_name=sheet_name_param,
            skiprows=skiprows_param,
            date_col=date_col_param,
            value_col=value_col_param,
            item_col=item_col_param,
            output_path=output_file_name,
            model_params=DEFAULT_PARAMS, # Using default model hyperparameter search spaces
            fill_missing_months_flag=fill_missing_months_param,
            skip_leading_zeros=skip_leading_zeros_param,
            error_metric=metric # This sets the metric for find_best_model's internal evaluation
        )

        if not results_df.empty:
            avg_acc = calculate_and_print_average_accuracies(results_df.copy(), metric.upper())
            if avg_acc is not None:
                 all_average_accuracies[metric.upper()] = avg_acc
        else:
            logging.warning(f"No results generated for internal metric: {metric.upper()}. Cannot calculate average accuracies.")

    logging.info("\n\n========== Summary of Average Accuracies Across All Internal Metrics ==========")
    if not all_average_accuracies:
        logging.info("No average accuracies were successfully calculated across any metric runs.")
    else:
        summary_avg_acc_df = pd.DataFrame(all_average_accuracies)
        logging.info("Average 'Forecast Accuracy (1 - |F-A|/|A|)' per Item, by Internal Model Selection Metric:\n")
        print(summary_avg_acc_df.to_string())
        summary_avg_acc_df.to_excel("summary_average_accuracies_by_internal_metric.xlsx")
        logging.info("\nSummary of average accuracies saved to summary_average_accuracies_by_internal_metric.xlsx")
    logging.info("==================================================================================")


if __name__ == "__main__":
    # Create a dummy Excel file for testing if it doesn't exist
    # In a real scenario, you would replace 'your_data_file.xlsx' with your actual file.
    try:
        pd.read_excel('your_data_file.xlsx')
    except FileNotFoundError:
        logging.warning("Dummy 'your_data_file.xlsx' not found. Creating a sample file for demonstration.")
        sample_dates = pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01',
                                       '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01',
                                       '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01'] * 2)
        sample_values = np.abs(np.random.randn(30) * 100 + 50).astype(int)
        sample_items = ['ItemA'] * 15 + ['ItemB'] * 15
        
        # Create a DataFrame with extra rows and columns to simulate `skiprows=2` and specific column names
        header_df = pd.DataFrame([["Title of Report"], ["Data extracted on X Date"]]) # 2 rows to skip
        data_df = pd.DataFrame({
            'Forecast Item': sample_items,
            'SomeOtherCol': np.random.rand(30),
            'Date': sample_dates,
            'Actual.1': sample_values,
            'Notes': ['note'] * 30
        })
        
        with pd.ExcelWriter('your_data_file.xlsx', engine='openpyxl') as writer:
            header_df.to_excel(writer, index=False, header=False, sheet_name='Sheet1') # Default sheet is 0
            data_df.to_excel(writer, index=False, header=True, sheet_name='Sheet1', startrow=2)
        logging.info("Sample 'your_data_file.xlsx' created. Please replace with your actual data for meaningful results.")
        logging.info("If you have an existing 'your_data_file.xlsx', it will be used instead of creating a new one.")
        
    main()