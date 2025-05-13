# Script to generate forecasts based on specific configurations and compare with consensus

import pandas as pd
import numpy as np
import logging
from sim_forecasting_logic import (
    load_weekly_data,
    forecast_moving_average,
    forecast_exponential_smoothing,
    forecast_linear_regression,
    forecast_multiple_linear_regression,
    # Add other forecast functions if needed (Croston, ARIMA etc.) based on config file
    forecast_arima,
    forecast_croston
)

# --- Configuration ---
HISTORICAL_DATA_FILE = "Outliers-2.xlsx"
CONFIG_FILE = "Arima/Forecast Item Configuration ARIMA.xlsx" #"/Users/igor/DemandForecasting/Forecast Item Configuration Exp.xlsx"#
CONFIG_FILE_OVERRIDE = "Arima/Forecast Item Configuration ARIMA.xlsx"
CONSENSUS_FILE = "Arima/Consensus Demand Plan - Next Level Summary ARIMA.xlsx"
OUTPUT_FILE = "forecast_comparison ARIMA.xlsx"

# ### ADJUST_CONFIG ### - Define column names for the Configuration file
CONFIG_SHEET_NAME = 0 # Or the specific sheet name as string
CONFIG_ITEM_COL = 'Forecast Item' # Column with the item identifier
CONFIG_MODEL_COL = 'Model'        # Column specifying the model type (e.g., 'Moving Average')
CONFIG_PARAM_NAME_COL = 'Name'    # Column with the parameter name (e.g., 'alpha', 'window')
CONFIG_PARAM_VALUE_COL = 'Value'  # Column with the parameter value

# ### ADJUST_CONSENSUS ### - Define column names for the Consensus file
CONSENSUS_SHEET_NAME = 0 # Or the specific sheet name as string
CONSENSUS_ITEM_COL = 'Part'  # User-defined original item column name in consensus file (e.g., 'Part' or 'Forecast ID')
CONSENSUS_DATE_COL = 'Date'  # This will be the name of the column holding dates after melting
CONSENSUS_VALUE_COL = 'Consensus Forecast' # Column with the consensus forecast value after melting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Logic ---
def main():
    logging.info("Starting forecast generation and comparison...")
    comparison_mape ={}
    # 1. Load Data
    try:
        # Load historical data using defaults from sim_forecasting_logic
        historical_data = load_weekly_data(
            HISTORICAL_DATA_FILE,
            item_col='Forecast Item', # Assumed based on sim_forecasting_logic defaults
            value_col='Actual.1',   # Corrected based on debug output
            date_col='Date',         # Assumed based on sim_forecasting_logic defaults
            skiprows=2,              # Added based on debugging output showing unnamed columns
            fill_missing_weeks_flag=False # Load raw data first, handle duplicates before filling
        )
        logging.info(f"Loaded raw historical data from {HISTORICAL_DATA_FILE}")

        # --- Add aggregation step to handle potential duplicate dates per item ---
        logging.info("Aggregating historical data by item and date (summing values)...")
        # Reset index because load_weekly_data(..., fill_missing_weeks_flag=False) returns Date as index
        historical_data = historical_data.reset_index()

        # Ensure the value column is numeric before summing
        historical_data['Value'] = pd.to_numeric(historical_data['Value'], errors='coerce').fillna(0)
        # Group by item and date, sum the values
        historical_data_agg = historical_data.groupby([CONFIG_ITEM_COL, 'Date'])['Value'].sum().reset_index()
        # Step 1: Group by Forecast Item and check if all Values are zero
        non_zero_items = historical_data_agg.groupby('Forecast Item')['Value'].transform(lambda x: not all(x == 0))

        # Step 2: Filter the original DataFrame to keep only the non-zero groups
        historical_data_agg = historical_data_agg[non_zero_items].reset_index(drop=True)

        logging.info("Aggregation complete.")
        # --- End aggregation step ---

        # --- Manually fill missing weeks after aggregation ---
        logging.info("Filling missing weeks after aggregation...")
        all_items_filled = []
        for item_id, group in historical_data_agg.groupby(CONFIG_ITEM_COL):
            group = group.set_index('Date').sort_index()
            if not group.empty:
                # Determine frequency (assume weekly, 'W-MON' as in sim_logic)
                # A more robust way might be needed if frequency varies
                freq = 'W-MON'
                all_weeks = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
                group_filled = group.reindex(all_weeks)
                group_filled['Value'] = group_filled['Value'].fillna(0)
                group_filled[CONFIG_ITEM_COL] = item_id # Add item column back
                group_filled.index.name = "Date"
                all_items_filled.append(group_filled.reset_index()[[CONFIG_ITEM_COL, 'Date', 'Value']])

        if not all_items_filled:
             raise ValueError("Historical data became empty after aggregation and filling.")

        historical_data = pd.concat(all_items_filled, ignore_index=True)
        mask = (
            historical_data
            .groupby(CONFIG_ITEM_COL)['Value']
            .transform(lambda x: x.ne(0).cummax())
        )

        historical_data = historical_data[mask].reset_index(drop=True)
        def drop_edges(group):
            # This will return rows 3 through (n-2-1) inclusive
            return group.iloc[3:-2]

        # 3. Apply per‐group
        trimmed = (
            historical_data
            .groupby('Forecast Item', group_keys=False)
            .apply(drop_edges)
            .reset_index(drop=True)
        )
        historical_data = trimmed
        logging.info("Missing weeks filled.")
        # --- End manual fill step ---
        # Load configuration data
        if CONSENSUS_FILE[:-5].split()[-1] in ["Crostons", "Exp", "Linear"]:
            config_df = pd.read_excel(CONFIG_FILE, sheet_name=CONFIG_SHEET_NAME, skiprows=1)
            logging.info(f"Loaded configuration from {CONFIG_FILE} (Croston path)")
            # Standardize item column name
            config_df.rename(columns={"Item": CONFIG_ITEM_COL}, inplace=True) # Assuming 'Item' is the name in this sheet

            # Check if required config columns exist
            required_config_cols = [CONFIG_ITEM_COL, CONFIG_PARAM_NAME_COL, CONFIG_PARAM_VALUE_COL]
            if CONFIG_MODEL_COL in config_df.columns:
                required_config_cols.append(CONFIG_MODEL_COL)
            missing_config_cols = [col for col in required_config_cols if col not in config_df.columns]
            if missing_config_cols:
                raise KeyError(f"Missing required columns in config file: {missing_config_cols}")

            # Strip whitespace from column headers and relevant string columns
            config_df.columns = config_df.columns.str.strip()
            cols_to_strip = [CONFIG_ITEM_COL, CONFIG_MODEL_COL, CONFIG_PARAM_NAME_COL]
            for col in cols_to_strip:
                if col in config_df.columns:
                    config_df[col] = config_df[col].astype(str).str.strip()
                else:
                    logging.warning(f"Configuration file missing expected column for stripping: {col}")

            # Pivot config for easier lookup: index='Forecast Item', columns='Parameter Name', values='Parameter Value'
            config_pivot = config_df.pivot_table(index=CONFIG_ITEM_COL,
                                                columns=CONFIG_PARAM_NAME_COL,
                                                values=CONFIG_PARAM_VALUE_COL,
                                                aggfunc='first')

            # Merge Model type back if it was in a separate column
            if CONFIG_MODEL_COL in config_df.columns:
                model_map = config_df[[CONFIG_ITEM_COL, CONFIG_MODEL_COL]].drop_duplicates().set_index(CONFIG_ITEM_COL)
                config_pivot = config_pivot.join(model_map)
            else:
                logging.warning(f"No '{CONFIG_MODEL_COL}' column found in config. Model type might not be explicitly available.")
                # Add a placeholder 'Model' column if it's missing but needed later
                if 'Model' not in config_pivot.columns:
                    config_pivot['Model'] = 'Unknown' # Placeholder
        else: 
            config_file_path = CONFIG_FILE
            config_df = pd.read_excel(config_file_path, sheet_name=CONFIG_SHEET_NAME, skiprows=2)
            logging.info(f"Loaded configuration from {config_file_path} (Non-Croston path)")
            
            config_df.columns = config_df.columns.str.strip() # Strip all column headers first
            config_df.rename(columns={"Item": CONFIG_ITEM_COL, "Model": CONFIG_MODEL_COL}, inplace=True, errors='ignore') # Rename common variations

            # Standardize relevant string column values by stripping whitespace
            cols_to_strip_nc = [CONFIG_ITEM_COL]
            if CONFIG_MODEL_COL in config_df.columns: # Only strip if 'Model' column exists
                cols_to_strip_nc.append(CONFIG_MODEL_COL)
            
            for col in cols_to_strip_nc:
                if col in config_df.columns:
                    config_df[col] = config_df[col].astype(str).str.strip()
                else:
                    # This case should ideally not be hit if rename worked or columns existed with right names
                    logging.warning(f"Configuration file (Non-Croston path) missing expected column for stripping: {col}")
            
            if CONFIG_ITEM_COL not in config_df.columns:
                raise KeyError(f"'{CONFIG_ITEM_COL}' not found in config file (Non-Croston path) after attempting rename. Columns found: {list(config_df.columns)}")

            config_pivot = config_df.set_index(CONFIG_ITEM_COL)

            # Ensure 'Model' column exists in the pivot, otherwise add a default
            if CONFIG_MODEL_COL not in config_pivot.columns:
                logging.warning(f"'{CONFIG_MODEL_COL}' column not found in config_pivot (Non-Croston path). Model type will be 'Unknown'. Available columns: {list(config_pivot.columns)}")
                config_pivot[CONFIG_MODEL_COL] = 'Unknown' # Assign default 'Unknown' to all rows
            else:
                # If 'Model' column exists, ensure its string values are also stripped
                config_pivot[CONFIG_MODEL_COL] = config_pivot[CONFIG_MODEL_COL].astype(str).str.strip()
            
            logging.info(f"Loaded configuration from {config_file_path} (Non-Croston path)")
        ################ EXCLUDING ITEMS THAT HAVE OVERRIDE ##################
        # Load configuration file
        cfg_path = CONFIG_FILE_OVERRIDE
        df_config = pd.read_excel(cfg_path, sheet_name=CONFIG_SHEET_NAME, skiprows=2)
        logging.info(f"Loaded configuration from {cfg_path} (CONFIG)")

        # Strip whitespace from all column headers
        df_config.columns = df_config.columns.str.strip()
        # Rename common variations to our standard constant names
        df_config.rename(
            columns={"Item": CONFIG_ITEM_COL, "Model": CONFIG_MODEL_COL},
            inplace=True,
            errors="ignore"
        )

        # Decide which columns to strip string values on
        strip_columns_nc = [CONFIG_ITEM_COL]
        if CONFIG_MODEL_COL in df_config.columns:
            strip_columns_nc.append(CONFIG_MODEL_COL)

        for column in strip_columns_nc:
            if column in df_config.columns:
                df_config[column] = df_config[column].astype(str).str.strip()
            else:
                logging.warning(
                    f"Configuration file (CONIFG) missing expected column for stripping: {column}"
                )

        # Ensure the mandatory item column is present
        if CONFIG_ITEM_COL not in df_config.columns:
            raise KeyError(
                f"'{CONFIG_ITEM_COL}' not found in config file (CONFIG) "
                f"after attempting rename. Columns found: {list(df_config.columns)}"
            )
        target_date = pd.to_datetime("2020-11-02")
        # 1. Build a mask of rows where Stop.1 is not missing
        mask = df_config["Stop.1"].notna() | (df_config["Start Date"] != target_date) | (df_config["Change"] != 0.0) | df_config["Other Items"].notna() | (df_config['Forecast.1'] != 106) #| (df_config["Factor"] != 1)

        # 2. Use .loc to select only the Forecast Item column for those rows
        forecast_items_with_dates = df_config.loc[mask, "Forecast Item"]

        # 3. Turn that Series into a plain Python list
        forecast_list = forecast_items_with_dates.tolist()

        ######################### END ##########################

        # Load consensus data
        logging.info(f"Loading consensus data from {CONSENSUS_FILE} with skiprows=1...")
        try:
            consensus_df_raw = pd.read_excel(CONSENSUS_FILE, sheet_name=CONSENSUS_SHEET_NAME, skiprows=1)
        except Exception as e:
            logging.error(f"Error loading consensus file raw: {e}")
            raise
        date_cols = consensus_df_raw.columns[4:]   # all columns from '7-Apr-25' onward
        df_long = (
                consensus_df_raw
                .melt(
                    id_vars=['Forecast Item', 'Unnamed: 1'],
                    value_vars=date_cols,
                    var_name='date',
                    value_name='value'
                )
            )
            # Quick sanity‐check:
        df_long['Date'] = pd.to_datetime(df_long['date'], format='%d-%b-%y')
        tidy = (
            df_long
            .pivot_table(
                index=['Forecast Item', 'Date'],
                columns='Unnamed: 1',
                values='value'
            )
            .reset_index()
            .rename_axis(columns=None)   # drop the leftover name on the columns axis
        )
        consensus_df = tidy

    except FileNotFoundError as e:
        logging.error(f"Error: Configuration file not found at {config_file_path}")
        return
    except KeyError as e:
        logging.error(f"Error: Column not found in one of the files: {e}. Please check file structures.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return
    # 2. Identify Common Items & Forecast Horizon
    # Use the consistent item column name (CONFIG_ITEM_COL)
    item_names_hist = set(historical_data[CONFIG_ITEM_COL].unique())
    item_names_config = set(config_pivot.index) # Index should be CONFIG_ITEM_COL
    item_names_consensus = set(consensus_df[CONFIG_ITEM_COL].unique()) # Item col is now CONFIG_ITEM_COL

    common_items = list(item_names_hist.intersection(item_names_config).intersection(item_names_consensus))
    common_items = [item
                   for item in common_items
                   if item not in set(forecast_list)]
    print(len(common_items))
    if not common_items:
        logging.warning("No common items found between historical data, config, and consensus. Exiting.")
        return
    # Calculate horizon length (number of weeks/periods)
    # This assumes weekly data consistent with sim_forecasting_logic
    # Use the number of unique dates in the consensus plan for the specific item
    # forecast_horizon = len(pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='W-MON')) # Example for weekly
    # A safer way might be to count unique dates per item in consensus

    # 3. Generate Forecasts per Item
    all_forecasts = []
    for item in common_items:
        logging.info(f"Processing item: {item}")
        # Ensure we use the correct item column name
        # Filter for the item, then set Date as index and select the Value series
        item_hist_data_df = historical_data[historical_data[CONFIG_ITEM_COL] == item]
        if item_hist_data_df.empty:
             logging.warning(f"Skipping {item}: No historical data found after filtering.")
             continue
        item_hist_data = item_hist_data_df.set_index('Date')['Value'].sort_index()

        item_config = config_pivot.loc[item]
        item_consensus = consensus_df[consensus_df[CONFIG_ITEM_COL] == item]["Baseline"].sort_index()

        if item_hist_data.empty:
            logging.warning(f"Skipping {item}: No historical data found.")
            continue

        # Determine forecast horizon specific to this item from its consensus data
        item_forecast_horizon = len(item_consensus)
        if item_forecast_horizon == 0:
            logging.warning(f"Skipping {item}: No consensus data found to determine horizon.")
            continue

        # Get model type and parameters
        # Model name should be in a column named as per CONFIG_MODEL_COL ('Model') in item_config
        model_name = item_config.get(CONFIG_MODEL_COL, 'Unknown') # Get model name, default to 'Unknown'

        generated_forecast = pd.Series(dtype=float) # Initialize an empty series

        try:
            # Extract parameters - ensure keys match CONFIG_PARAM_NAME_COL values
            if model_name == 'MovingAverage': # ### ADJUST_MODEL_NAME ### Check exact name in config file
                param_name = 'Average' # ### ADJUST_PARAM_NAME ### Check exact parameter name in config file
                if param_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_name}' not found in config for item {item}")
                window = int(item_config[param_name])
                generated_forecast = forecast_moving_average(item_hist_data, item_forecast_horizon, window)
                params_used = {param_name: window}
            elif model_name == 'ExponentialSmoothing': # ### ADJUST_MODEL_NAME ###
                param_name = 'Alpha' # ### ADJUST_PARAM_NAME ###
                if param_name not in item_config.index: # Check if param_name is in the columns of item_config
                    print(item_config)
                    raise KeyError(f"Parameter '{param_name}' not found in config for item {item}")
                alpha = float(item_config[param_name])
                generated_forecast = forecast_exponential_smoothing(item_hist_data, item_forecast_horizon, alpha)
                params_used = {param_name: alpha}
            elif model_name == 'Linear': # ### ADJUST_MODEL_NAME ###
                # No specific parameters needed from config based on sim_forecasting_logic
                generated_forecast = forecast_linear_regression(item_hist_data, item_forecast_horizon)
                params_used = {}
            elif model_name == 'MultipleLinearRegression': # ### ADJUST_MODEL_NAME ###
                print(len(item_hist_data))
                param_name = 'Cycle' # ### ADJUST_PARAM_NAME ###
                if param_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_name}' not found in config for item {item}")
                seasonality = int(item_config[param_name])
                generated_forecast = forecast_multiple_linear_regression(item_hist_data, item_forecast_horizon, seasonality)
                params_used = {param_name: seasonality}
            elif model_name == 'ARIMA': # ### ADJUST_MODEL_NAME ###
                param_ar_name = 'Auto Regressive' # ### ADJUST_PARAM_NAME ###
                param_ma_name = 'Moving Average' # ### ADJUST_PARAM_NAME ###
                param_d_name = 'Level' # ### ADJUST_PARAM_NAME ###
                param_constant_name = "Constant" # ### ADJUST_PARAM_NAME ###
                print(item_config[[param_ar_name, param_ma_name, param_d_name, param_constant_name]])
                if param_ar_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_ar_name}' not found in config for item {item}")
                if param_ma_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_ma_name}' not found in config for item {item}")
                if param_d_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_d_name}' not found in config for item {item}")
                if param_constant_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_constant_name}' not found in config for item {item}")
                ar_int = item_config[param_ar_name]
                ma_int = item_config[param_ma_name]
                d_int = item_config[param_d_name]
                if np.isnan(ar_int):
                    ar_int = 2
                else:
                    ar_int = int(item_config[param_ar_name])
                if np.isnan(ma_int):
                    ma_int = 2
                else:
                    ma_int = int(item_config[param_ma_name])
                if np.isnan(d_int):
                    d_int = 1
                else:
                    d_int = int(item_config[param_d_name])
                if param_constant_name == "Igonre":
                    constant_str = "n"
                elif param_constant_name == "Constant":
                    constant_str = "c"
                else:
                    raise ValueError(f"Invalid constant value for {item}: {constant_str}")
                try:
                    # Convert string '(p,d,q)' to tuple
                    order_tuple = tuple(map(int, (ar_int, d_int, ma_int)))
                except:
                    logging.warning(f"Invalid ARIMA order format for {item}: {order_str}. Using default (1,1,1).")
                    order_tuple = (1,1,1)
                generated_forecast = forecast_arima(item_hist_data, item_forecast_horizon, order_tuple, constant_str)
                params_used = {param_ar_name: ar_int, param_ma_name: ma_int, param_d_name: d_int, param_constant_name: constant_str}

            # Add other models (Croston, etc.) here if needed, checking model/param names
            elif model_name == 'CrostonsMethod':
                param_alpha_name = 'Alpha' # ### ADJUST_PARAM_NAME ###
                if param_alpha_name not in item_config.index: # Check if param_name is in the columns of item_config
                    raise KeyError(f"Parameter '{param_alpha_name}' not found in config for item {item}")
                alpha = float(item_config[param_alpha_name])
                generated_forecast = forecast_croston(item_hist_data, item_forecast_horizon, alpha)
                params_used = {param_alpha_name: alpha}
            elif model_name == 'Unknown' or model_name is None:
                 logging.warning(f"Skipping {item}: Model type is '{model_name}'. Check config file or logic.")
                 continue
            else:
                logging.warning(f"Skipping {item}: Unknown or unhandled model type '{model_name}' in config.")
                continue

            if generated_forecast is not None:
                # Align forecast index with consensus index
                if len(generated_forecast) != len(item_consensus):
                     logging.warning(f"Forecast length ({len(generated_forecast)}) does not match consensus length ({len(item_consensus)}) for {item}. Truncating/padding forecast index.")
                generated_forecast.index = item_consensus.index[:len(generated_forecast)] # Ensure index matches consensus dates

                # Ensure item_consensus is aligned to generated_forecast for MAPE calculation
                # Taking the 'Consensus Forecast' column for actuals
                actuals_for_mape = item_consensus.loc[generated_forecast.index]

                # Combine generated forecast and consensus using the consistent item column name
                item_comparison = pd.DataFrame({
                    'Generated Forecast': generated_forecast,
                    'Consensus Forecast': actuals_for_mape # Use aligned actuals
                })

                # Calculate MAPE
                mape = np.nan # Default in case of issues
                if not actuals_for_mape.empty and not generated_forecast.empty:
                    # Ensure both series are float and handle potential NaNs from alignment
                    fcst_aligned = generated_forecast.reindex(actuals_for_mape.index).astype(float)
                    act_aligned = actuals_for_mape.astype(float)
                    
                    # Filter out NaNs that might have been introduced by reindexing or were already present
                    valid_indices = act_aligned.notna() & fcst_aligned.notna()
                    act_aligned_valid = act_aligned[valid_indices]
                    fcst_aligned_valid = fcst_aligned[valid_indices]

                    if not act_aligned_valid.empty:
                        # Avoid division by zero: filter out zero actuals
                        non_zero_actuals_mask = act_aligned_valid != 0
                        if non_zero_actuals_mask.any(): # Check if there are any non-zero actuals
                            mape = np.mean(np.abs((act_aligned_valid[non_zero_actuals_mask] - fcst_aligned_valid[non_zero_actuals_mask]) / act_aligned_valid[non_zero_actuals_mask])) * 100
                        else:
                            logging.warning(f"MAPE not calculated for {item}: all actuals in forecast period are zero.")
                            mape = 0 # Or np.nan, depending on desired representation for all zero actuals
                    else:
                        logging.warning(f"MAPE not calculated for {item}: no valid overlapping actuals and forecasts after alignment.")

                item_comparison[CONFIG_ITEM_COL] = item # Add item identifier column
                item_comparison['Model Used'] = model_name
                item_comparison['Parameters'] = str(params_used)
                comparison_mape[item] = mape # Add MAPE to the dictionary for this item

                all_forecasts.append(item_comparison.reset_index())

        except KeyError as e:
            logging.warning(f"Skipping {item}: Missing parameter key {e} for model '{model_name}' in config. "
                            f"Check config file contents and column names ({CONFIG_PARAM_NAME_COL}). "
                            f"Available parameters for this item in config: {list(item_config.index) if isinstance(item_config, pd.Series) else 'N/A (item_config not a Series)'}")
        except ValueError as e:
             logging.warning(f"Skipping {item}: Error converting parameter value for model '{model_name}'. Check config file contents. Error: {e}")
        except Exception as e:
            logging.error(f"Error generating forecast for {item} (Model: {model_name}): {e}", exc_info=True) # Add traceback

    # 4. Combine and Save Results
    if not all_forecasts:
        logging.warning("No forecasts were generated.")
        return

    final_comparison_df = pd.concat(all_forecasts, ignore_index=True)

    # Reorder columns for clarity (using CONFIG_ITEM_COL and 'Date')
    # Ensure 'MAPE' is included in the desired column order
    cols_order = [CONFIG_ITEM_COL, 'Date', 'Generated Forecast', 'Consensus Forecast', 'Model Used', 'Parameters', 'MAPE']
    # Filter out any columns not present in final_comparison_df to avoid KeyErrors if some items didn't produce all cols (e.g. MAPE)
    final_comparison_df = final_comparison_df[[col for col in cols_order if col in final_comparison_df.columns]]
    import statistics

    # 1. Get the item with max and min MAPE
    max_item = max(comparison_mape, key=comparison_mape.get)
    min_item = min(comparison_mape, key=comparison_mape.get)

    # 2. Look up their MAPE values
    max_mape = comparison_mape[max_item]
    min_mape = comparison_mape[min_item]

    # 3. Compute the mean MAPE
    mean_mape = statistics.mean(comparison_mape.values())

    # 4. Print results
    # 1. Sort items by MAPE descending and take the first 10
    top_10 = sorted(
        comparison_mape.items(),
        key=lambda kv: kv[1],
        reverse=True
    )[:10]

    # 2. Print them out
    print("Top 10 highest MAPE values:")
    for rank, (item, mape) in enumerate(top_10, start=1):
        print(f"{rank:2}. {item} → {mape:.2f}%")
    print(f"Lowest  MAPE: {min_item} → {min_mape:.2f}%")
    print(f"Mean    MAPE: {mean_mape:.2f}%")

    print(comparison_mape)
    try:
        final_comparison_df.to_excel(OUTPUT_FILE, index=False)
        logging.info(f"Forecast comparison saved to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Error saving results to {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()
