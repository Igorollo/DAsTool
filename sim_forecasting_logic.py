# -*- coding: utf-8 -*-
"""
Simplified Demand Forecasting Logic

Focuses on weekly data, 70/30 train/test split, and selects the best model 
based on weekly MAPE on the test set.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import logging
import warnings
from typing import Dict, Tuple, Any, Optional, List, Literal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

# --- Default Parameters (can be overridden) ---
# Default error metric to use for model selection
DEFAULT_ERROR_METRIC = 'smape'  # Options: 'mape', 'smape', 'rmse'

DEFAULT_PARAMS = {
    "ma_windows": list(range(2, 27)), # Expanded: 2 to 26 weeks
    "es_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(), # Kept standard range
    "croston_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(), # Added for clarity, same as ES
    "mlr_seasonalities": list(range(4, 27)), # Expanded: 4 to 26 weeks
    "arima_orders": [
        # Include both d=0 and d=1 models
        # d=0 models (can use 'c' trend)
        (p, 0, q) for p in range(3) for q in range(3) 
        # d=1 models (need 'n' or 'ct' trend)
        ] + [(p, 1, q) for p in range(3) for q in range(3)
        # Full expansion would be:
        # (0,0,0), (1,0,0), (2,0,0), (0,0,1), (1,0,1), (2,0,1), (0,0,2), (1,0,2), (2,0,2),
        # (0,1,0), (1,1,0), (2,1,0), (0,1,1), (1,1,1), (2,1,1), (0,1,2), (1,1,2), (2,1,2)
    ], # Expanded ARIMA orders with both d=0 and d=1
    "arima_trends": ['n', 'c', 'ct'], # 'none', 'constant', and 'constant+trend'
}

# --- Helper Functions ---

def remove_leading_zeros(series: pd.Series) -> pd.Series:
    """
    Removes leading zeros from a time series.
    Returns a series starting from the first non-zero value.
    If all values are zero, returns the original series.
    """
    if series.empty:
        return series
        
    # Find the first non-zero value index
    first_nonzero_idx = series[series != 0].index.min() if (series != 0).any() else None
    
    # If all values are zero or found no non-zero values, return the original
    if first_nonzero_idx is None:
        return series
        
    # Return series from the first non-zero value onwards
    return series.loc[first_nonzero_idx:]    

def fill_missing_weeks(df: pd.DataFrame, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ensures there are no missing weekly records for each item between the global first and last date.
    If a week is missing for an item, adds it with the specified value_col set to 0 for that item.
    Assumes df is indexed by date_col (e.g., 'Date') and sorted by this index.
    The date_col parameter refers to the name of the index.
    """
    if df.empty:
        return df

    # Determine overall date range from the input DataFrame's index
    min_date = df.index.min()
    max_date = df.index.max()
    
    current_index_name = df.index.name if df.index.name else date_col

    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = 'W-MON' # Default weekly frequency, assuming Monday starts

    all_weeks_idx = pd.date_range(start=min_date, end=max_date, freq=freq)
    all_weeks_idx.name = current_index_name

    if item_col and item_col in df.columns:
        filled_dfs = []
        # groupby preserves the order of groups as they first appear
        for item, group_df in df.groupby(item_col):
            # group_df comes with date_col as index. Create a copy for modification.
            group_df_copy = group_df.copy()
            group_reindexed = group_df_copy.reindex(all_weeks_idx)
            
            if value_col in group_reindexed.columns:
                group_reindexed[value_col] = group_reindexed[value_col].fillna(0)
            # If value_col was not in original df, it won't be in group_reindexed.
            # If it's critical, it might need to be created: group_reindexed[value_col] = 0

            group_reindexed[item_col] = item # Fill item_col for new rows
            
            # Propagate other column values for the item if they were static
            for col in group_df_copy.columns:
                if col not in [value_col, item_col] and col != current_index_name: # current_index_name is date_col
                    if col in group_reindexed.columns: # Ensure column exists in reindexed frame
                        unique_vals = group_df_copy[col].dropna().unique()
                        if len(unique_vals) == 1:
                            group_reindexed[col] = group_reindexed[col].fillna(unique_vals[0])
                        # NaNs in other columns (non-static or all NaN in original group) will persist for new rows.
            filled_dfs.append(group_reindexed)
        
        if filled_dfs:
            df_final_filled = pd.concat(filled_dfs)
        else: 
            # This case implies input df was empty or item_col resulted in no groups.
            # Create a shell df with the full date range.
            df_final_filled = pd.DataFrame(index=all_weeks_idx)
            if value_col: # Add value_col if specified
                 df_final_filled[value_col] = 0
            # item_col cannot be filled meaningfully here.
            
    else:
        # Original behavior: process as a single series
        df_final_filled = df.reindex(all_weeks_idx)
        if value_col in df_final_filled.columns:
            df_final_filled[value_col] = df_final_filled[value_col].fillna(0)
        else:
            # If value_col not present, create it and fill with 0.
            df_final_filled[value_col] = 0 # Create and fill if not existing
    
    # Ensure DataFrame is sorted by index (date)
    # If concat was used, it stacks groups; sort_index will interleave by date correctly.
    df_final_filled = df_final_filled.sort_index()
    df_final_filled.index.name = current_index_name # Ensure index name is set
    
    return df_final_filled

def load_weekly_data(file_path: str, sheet_name: Optional[str] = 0, skiprows: int = 0, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = 'Forecast Item', fill_missing_weeks_flag: bool = True, skip_leading_zeros: bool = False) -> pd.DataFrame:
    """
    Loads and prepares weekly data from an Excel file.
    If fill_missing_weeks_flag is True (default), fills missing weekly dates with zeros.
    If False, missing weeks are omitted (no zeros inserted).
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
        logging.info(f"Loaded weekly data from {file_path}")

        # Basic cleaning and type conversion
        df = df.rename(columns={date_col: 'Date', value_col: 'Value'})
        if item_col and item_col in df.columns:
             df = df[[item_col, 'Date', 'Value']]
        else: # Handle case without item column
            df = df[['Date', 'Value']]
            df[item_col] = 'Single_Item' # Assign a default item name

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        # Optionally fill missing weekly records with 0
        if fill_missing_weeks_flag:
            df = fill_missing_weeks(df, item_col=item_col)
        print("FILLING COMPLETE")
        # Skip leading zeros if requested
        # We do this after fill_missing_weeks to ensure proper date continuity
        if skip_leading_zeros and not df.empty:
            # If we have item column, process each item separately
            if item_col and item_col in df.columns and len(df[item_col].unique()) > 1:
                # Group by item and process each series
                groups = []
                for name, group in df.groupby(item_col):
                    # Process this item's value series
                    processed_series = remove_leading_zeros(group['Value'])
                    # Only keep the rows from the processed series
                    processed_group = group.loc[processed_series.index]
                    groups.append(processed_group)
                # Recombine all items
                if groups:
                    df = pd.concat(groups)
            else:
                # No item column or only one item, process the whole series
                # Get the processed value series
                processed_series = remove_leading_zeros(df['Value'])
                # Only keep rows from the processed series
                df = df.loc[processed_series.index]

        return df

    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise
    except KeyError as e:
        logging.error(f"Error: Column not found - {e}. Check column names (date_col, value_col, item_col).")
        raise
    except Exception as e:
        logging.error(f"Error loading weekly data: {e}")
        raise

def split_data(df: pd.DataFrame, train_split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and testing sets."""
    split_point = int(len(df) * train_split_ratio)
    train_data = df.iloc[:split_point]
    test_data = df.iloc[split_point:]
    logging.info(f"Data split: Train {len(train_data)} rows, Test {len(test_data)} rows")
    return train_data, test_data

def calculate_error_metric(actual: pd.Series, forecast: pd.Series, metric: Literal['mape', 'smape', 'rmse'] = 'smape') -> float:
    """Calculates specified error metric ensuring alignment and handling zeros.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecast : pd.Series
        Forecast values
    metric : str
        Error metric to use: 'mape', 'smape', or 'rmse'
        - mape: Mean Absolute Percentage Error (favors underforecasting)
        - smape: Symmetric Mean Absolute Percentage Error (balanced)
        - rmse: Root Mean Square Error (penalizes large errors more)
    
    Returns:
    --------
    float
        The calculated error metric
    """
    actual = actual.copy()
    forecast = forecast.copy()
    
    # Align forecast index to actual index
    forecast = forecast.reindex(actual.index)
    
    # Drop NA resulting from alignment or original data
    mask = actual.notna() & forecast.notna()
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        logging.warning("No valid points for error calculation after alignment.")
        return np.inf  # Return infinity if no comparable points
    
    if metric == 'mape':
        # Avoid division by zero for MAPE
        mask_zero = actual != 0
        if not mask_zero.all():
            logging.warning(f"Zeros found in actual values during MAPE calculation. These points ({sum(~mask_zero)}) will be excluded.")
            actual = actual[mask_zero]
            forecast = forecast[mask_zero]
            
            if len(actual) == 0:
                logging.warning("No valid points for MAPE calculation after zero handling.")
                return np.inf  # Return infinity if no comparable points
                
        # Calculate MAPE manually to ensure precision
        abs_percentage_errors = np.abs((actual - forecast) / actual) * 100
        return abs_percentage_errors.mean()
    
    elif metric == 'smape':
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        # Works better with zeros and is symmetric for over/under forecasting
        # Formula: 200 * |forecast - actual| / (|forecast| + |actual|)
        numerator = np.abs(forecast - actual)
        denominator = np.abs(forecast) + np.abs(actual)
        
        # Handle division by zero (when both forecast and actual are zero)
        mask_zero = denominator != 0
        if not mask_zero.all():
            logging.warning(f"Zero denominators found during SMAPE calculation. These points ({sum(~mask_zero)}) will be excluded.")
            numerator = numerator[mask_zero]
            denominator = denominator[mask_zero]
            
            if len(numerator) == 0:
                logging.warning("No valid points for SMAPE calculation after zero handling.")
                return np.inf  # Return infinity if no comparable points
                
        smape_values = 200 * numerator / denominator
        return smape_values.mean()
    
    elif metric == 'rmse':
        # RMSE (Root Mean Square Error)
        # Less sensitive to outliers and doesn't have percentage issues
        return np.sqrt(mean_squared_error(actual, forecast))
    
    else:
        logging.warning(f"Unknown error metric: {metric}. Using SMAPE instead.")
        # Default to SMAPE if unknown metric
        return calculate_error_metric(actual, forecast, 'smape')

# --- Model Implementations (Simplified) ---

def forecast_moving_average(train_data: pd.Series, test_len: int, window: int) -> Optional[pd.Series]:
    """Generates forecasts using a simple moving average."""
    if len(train_data) < window:
        logging.warning(f"MA: Not enough training data ({len(train_data)}) for window {window}")
        return None
    # Use rolling mean on training data, then forward fill for the forecast period
    # A simple approach: forecast based on the last available window average
    last_train_avg = train_data.iloc[-window:].mean()
    forecast = pd.Series([last_train_avg] * test_len, index=pd.date_range(start=train_data.index[-1] + pd.Timedelta(weeks=1), periods=test_len, freq=train_data.index.freq))
    return forecast

def forecast_exponential_smoothing(train_data: pd.Series, test_len: int, alpha: float) -> Optional[pd.Series]:
    """Generates forecasts using simple exponential smoothing."""
    try:
        model = sm.tsa.SimpleExpSmoothing(train_data, initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
        forecast = model.forecast(steps=test_len)
        # Ensure forecast index matches expected test period frequency
        forecast.index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(weeks=1), periods=test_len, freq=train_data.index.freq)
        return forecast
    except Exception as e:
        logging.error(f"ES Error (alpha={alpha}): {e}")
        return None

def forecast_linear_regression(train_data: pd.Series, test_len: int) -> Optional[pd.Series]:
    """Generates forecasts using linear regression (trend)."""
    try:
        # Create time trend feature
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values

        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Linear Regression Coefficients:")
        print(model.coef_*7)
        print("Linear Regression Intercept:")
        print(model.intercept_ - model.coef_*20213)

        # Create future time trend for forecasting
        X_test = np.arange(len(train_data), len(train_data) + test_len).reshape(-1, 1)
        forecast_values = model.predict(X_test)

        forecast = pd.Series(forecast_values, index=pd.date_range(start=train_data.index[-1] + pd.Timedelta(weeks=1), periods=test_len, freq=train_data.index.freq))
        return forecast
    except Exception as e:
        logging.error(f"LR Error: {e}")
        return None


def forecast_multiple_linear_regression(train_data: pd.Series, test_len: int, seasonality: int) -> pd.Series:
    """
    Forecast next `test_len` periods using multiple linear regression
    with weekly seasonality and linear trend, ignoring decay and evaluation.
    """
    test_len = test_len+2
    train_data.index = pd.DatetimeIndex(train_data.index, freq='W-MON')

    # Ensure DateTimeIndex with freq
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        raise ValueError("train_data must have a regular DateTimeIndex with a defined freq.")
    
    # Compute numeric date (days since start)
    numeric_date = (train_data.index - train_data.index.min()).days
    
    # Seasonality feature: week number within cycle
    week_num = (numeric_date // 7) % seasonality
    
    # One-hot encode seasonality
    X_train = pd.get_dummies(week_num, prefix='week')
    for i in range(seasonality):
        col = f'week_{i}'
        if col not in X_train.columns:
            X_train[col] = 0
    X_train = X_train[[f'week_{i}' for i in range(seasonality)]]
    
    # Add trend feature
    X_train['trend'] = numeric_date
    
    # Train target
    y_train = train_data.values
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Build test set dates
    last_date = train_data.index.max()
    freq = train_data.index.freq
    future_index = pd.date_range(start=last_date + freq, periods=test_len, freq=freq)
    numeric_future = (future_index - train_data.index.min()).days
    week_future = (numeric_future // 7) % seasonality
    
    X_test = pd.get_dummies(week_future, prefix='week')
    for i in range(seasonality):
        col = f'week_{i}'
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[[f'week_{i}' for i in range(seasonality)]]
    X_test['trend'] = numeric_future
    
    # Predict and enforce non-negativity
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)
    
    # Return forecast series
    return pd.Series(preds[2:], index=future_index[2:], name='forecast')


def forecast_croston(train_data: pd.Series, test_len: int, alpha: float = 0.1) -> Optional[pd.Series]: 
    """Generates forecasts using Croston's method for intermittent demand."""
    # Need at least two points to establish intervals
    if len(train_data) < 2:
        logging.warning(f"Croston: Not enough data ({len(train_data)})")
        return None

    # Extract raw values
    demand = train_data.values
    n = len(demand)

    # Prepare arrays
    Z = np.zeros(n)      # demand-size estimates
    X = np.zeros(n)      # interval-size estimates
    fitted = np.zeros(n) # in‐sample fitted values

    # Find first non-zero demand
    first_idx = next((i for i, v in enumerate(demand) if v != 0), None)
    if first_idx is None:
        logging.warning("Croston: No non-zero demands in training data")
        return None

    # Initialize at first demand
    Z[first_idx] = demand[first_idx]
    X[first_idx] = 1
    q = 1  # periods since last demand

    # Run the Croston update
    for i in range(first_idx + 1, n):
        if demand[i] != 0:
            Z[i] = alpha * demand[i] + (1 - alpha) * Z[i - 1]
            X[i] = alpha * q + (1 - alpha) * X[i - 1]
            q = 1
        else:
            Z[i] = Z[i - 1]
            X[i] = X[i - 1]
            q += 1
        # fitted value uses the *previous* estimates
        # Avoid division by zero if X[i-1] is zero
        if X[i - 1] > 0:
             fitted[i] = Z[i - 1] / X[i - 1]
        else:
             fitted[i] = 0 # Or handle as appropriate, e.g., Z[i-1]


    # Final “steady‐state” forecast value
    # Avoid division by zero if X[-1] is zero
    if X[-1] > 0:
        forecast_value = Z[-1] / X[-1]
    else:
        forecast_value = 0 # Or handle appropriately, e.g., Z[-1]


    # Build forecast index one step ahead, matching training freq
    # Ensure index frequency is available
    freq = pd.infer_freq(train_data.index)
    if freq is None:
        logging.warning("Croston: Cannot determine frequency from training data index.")
        # Attempt to use days if frequency cannot be inferred
        try:
             delta = train_data.index[1] - train_data.index[0]
             if delta.days > 0:
                 freq = f'{delta.days}D'
             else:
                 # Fallback or raise error if specific frequency handling is needed
                 logging.error("Croston: Could not infer frequency and time delta is not in days.")
                 return None
        except IndexError:
             logging.error("Croston: Not enough data points to infer frequency.")
             return None
        except Exception as e:
             logging.error(f"Croston: Error determining frequency: {e}")
             return None

    start = train_data.index[-1] + pd.Timedelta(days=1) if freq == 'D' else train_data.index[-1] + pd.tseries.frequencies.to_offset(freq)

    # Ensure start is correctly calculated
    # freq = train_data.index.freq or pd.infer_freq(train_data.index)
    # if freq is None:
    #    logging.warning("Croston: Cannot determine frequency from training data index.")
    #    return None # Or handle differently
    # start = train_data.index[-1] + freq

    idx = pd.date_range(start=start, periods=test_len, freq=freq)

    # Return constant forecast series
    return pd.Series([forecast_value] * test_len, index=idx)

def forecast_arima(
    train_data: pd.Series,
    test_len: int,
    arima_order: Tuple[int, int, int],
    trend_param: str = "n" #for constant c, ct
) -> Optional[pd.Series]:
    """Generates forecasts using an ARIMA model."""
    # Handle trend parameter based on differencing
    p, d, q = arima_order
    # If d > 0, we can only use 'n' (none) or 'ct' (constant + trend) but not 'c' (constant) alone
    if d > 0 and trend_param == 'c':
        logging.warning(f"ARIMA: Changing trend from 'c' to 'ct' for order {arima_order} with d={d} to include constant with trend")
        trend_param = 'ct'  # Change to 'constant+trend' which is valid with differencing
    
    # Need enough data to fit
    min_data_needed = sum(arima_order) + 1 # Basic check
    if len(train_data) < min_data_needed:
        logging.warning(
            f"ARIMA: Not enough training data ({len(train_data)}) for order={arima_order}, need at least {min_data_needed}"
        )
        return None
        
    # If test_len is 0, we can't generate a forecast, so return None
    if test_len <= 0:
        return None

    try:
        # Instantiate & fit
        model = ARIMA(train_data, order=arima_order, trend=trend_param)
        model_fit = model.fit()
        
        # Determine frequency for the forecast index
        freq = pd.infer_freq(train_data.index)
        if freq is None:
            # Try to infer from consecutive points
            try:
                # Calculate most common time delta
                deltas = [train_data.index[i+1] - train_data.index[i] for i in range(len(train_data.index)-1)]
                if deltas:
                    most_common_delta = max(set(deltas), key=deltas.count)
                    # Use this delta to create a frequency string
                    if most_common_delta.days >= 1:
                        freq = f'{most_common_delta.days}D'
                    elif most_common_delta.days == 7:
                        freq = 'W'
                    else:
                        # Default to weekly if we can't determine
                        freq = 'W'
                else:
                    # Default to weekly if no deltas
                    freq = 'W'
            except Exception as e:
                logging.warning(f"ARIMA: Error inferring frequency: {e}, defaulting to weekly")
                freq = 'W'
        
        # Create forecast index - ensure it's after the training data
        last_date = train_data.index[-1]
        if isinstance(freq, str):
            # Use date_range with explicit frequency
            start_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(start=start_date, periods=test_len, freq=freq)
        else:
            # If freq is a timedelta, use it directly
            forecast_index = [last_date + (i+1)*freq for i in range(test_len)]
            
        # Generate forecast
        raw_fc = model_fit.forecast(steps=test_len)
        forecast = pd.Series(raw_fc, index=forecast_index)
        return forecast

    except Exception as e:
        logging.error(f"ARIMA Error(order={arima_order}, trend={trend_param}): {e}")
        # For debugging
        if 'Prediction must have `end` after `start`' in str(e):
            logging.info(f"ARIMA date range issue with train_data index: {train_data.index[0]} to {train_data.index[-1]}, test_len={test_len}")
        return None

# --- Orchestration ---

def find_best_model(item_data: pd.Series, params: Dict = DEFAULT_PARAMS, return_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Optional[List[Dict[str, Any]]]:
    """Evaluates multiple models and returns all models sorted by MAPE."""
    if item_data.empty or len(item_data) < 4: # Need enough data for train/test
        logging.warning(f"Skipping item {item_data.name}: Not enough data ({len(item_data)})")
        return None

    # Split data (70% train, 30% test)
    train_len = int(len(item_data) * 0.7)
    if train_len < 2: # Need at least 2 points for some models
         logging.warning(f"Skipping item {item_data.name}: Not enough training data ({train_len}) after split")
         return None
    train_data = item_data.iloc[:train_len]
    test_data = item_data.iloc[train_len:]
    test_len = len(test_data)
    
    # Store the original data for visualization if needed
    full_data = item_data.copy()

    if test_len == 0:
        logging.warning(f"Skipping item {item_data.name}: No test data after split")
        return None

    # Use a dictionary to track unique model configurations
    # Key format: model_name + str(params)
    unique_results = {}

    # 1. Moving Average
    for w in params.get("ma_windows", []):
        forecast = forecast_moving_average(train_data, test_len, w)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Moving Average_window_{w}"
            result_dict = {
                "model": "Moving Average", 
                "param_window": w, 
                "error": error_value,
                "error_metric": error_metric
            }
            if return_forecasts:
                # Just store the data we already have for visualization
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 2. Exponential Smoothing (Simple)
    for alpha in params.get("es_alphas", []):
        forecast = forecast_exponential_smoothing(train_data, test_len, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Exponential Smoothing_alpha_{alpha}"
            result_dict = {
                "model": "Exponential Smoothing", 
                "param_alpha": alpha, 
                "error": error_value,
                "error_metric": error_metric
            }
            if return_forecasts:
                # Just store the data we already have for visualization
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 3. Linear Regression
    forecast = forecast_linear_regression(train_data, test_len)
    if forecast is not None:
        error_value = calculate_error_metric(test_data, forecast, error_metric)
        model_key = "Linear Regression"
        result_dict = {
            "model": "Linear Regression", 
            "param": "N/A", 
            "error": error_value,
            "error_metric": error_metric
        }
        if return_forecasts:
            # Just store the data we already have for visualization
            result_dict["train_data"] = train_data
            result_dict["test_data"] = test_data
            result_dict["forecast"] = forecast
            result_dict["full_data"] = full_data
        unique_results[model_key] = result_dict

    # 4. Multiple Linear Regression (with Seasonality)
    for seas in params.get("mlr_seasonalities", []):
        forecast = forecast_multiple_linear_regression(train_data, test_len, seas)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"MLR_seasonality_{seas}"
            result_dict = {
                "model": "MLR", 
                "param_seasonality": seas, 
                "error": error_value,
                "error_metric": error_metric
            }
            if return_forecasts:
                # Just store the data we already have for visualization
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 5. Croston
    for alpha in params.get("croston_alphas", []): # Use dedicated param key
        forecast = forecast_croston(train_data, test_len, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Croston_alpha_{alpha}"
            result_dict = {
                "model": "Croston", 
                "param_alpha": alpha, 
                "error": error_value,
                "error_metric": error_metric
            }
            if return_forecasts:
                # Just store the data we already have for visualization
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 6. ARIMA
    for arima_order in params.get("arima_orders", []):
        for trend in params.get("arima_trends", ['n']): # Iterate trends, default 'n'
            # Handle valid trend combinations based on differencing order
            p, d, q = arima_order
            # No need to skip - we'll convert 'c' to 'ct' in the forecast_arima function
                
            forecast = forecast_arima(train_data, test_len, arima_order, trend_param=trend)
            if forecast is not None:
                error_value = calculate_error_metric(test_data, forecast, error_metric)
                model_key = f"ARIMA_order_{arima_order}_trend_{trend}"
                result_dict = {
                    "model": "ARIMA",
                    "param_order": str(arima_order), # Store as string for output
                    "param_trend": trend,
                    "error": error_value,
                    "error_metric": error_metric
                }
                if return_forecasts:
                    # Just store the data we already have for visualization
                    result_dict["train_data"] = train_data
                    result_dict["test_data"] = test_data
                    result_dict["forecast"] = forecast
                    result_dict["full_data"] = full_data
                unique_results[model_key] = result_dict

    # Convert dictionary to list and sort by MAPE
    results = list(unique_results.values())
    
    if not results:
        logging.warning(f"No successful models found for item {item_data.name}")
        return None

    # Sort results by error metric (ascending)
    results.sort(key=lambda x: x['error'])
    best_result = results[0] if results else None
    
    if best_result:
        logging.info(f"Best model for {item_data.name}: {best_result['model']} with params { {k: v for k, v in best_result.items() if k.startswith('param_')} } -> {best_result['error_metric'].upper()}: {best_result['error']:.4f}")
        logging.info(f"Found {len(results)} unique model configurations")

    return results

def run_simplified_forecast(file_path: str, sheet_name: Optional[str] = 0, skiprows: int = 0, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = 'Forecast Item', output_path: str = "simplified_best_models.xlsx", model_params: Dict = DEFAULT_PARAMS, fill_missing_weeks_flag: bool = True, skip_leading_zeros: bool = False, return_best_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Loads data, runs model selection for each item, and saves the best models.
    If fill_missing_weeks_flag is True, missing weeks are filled with zeros. If False, missing weeks are omitted.
    """
    all_data = load_weekly_data(file_path, sheet_name, skiprows, date_col, value_col, item_col, fill_missing_weeks_flag=fill_missing_weeks_flag, skip_leading_zeros=skip_leading_zeros)
    print(f"All data shape: {all_data.shape}")
    best_models = []
    best_forecasts = {}
    # Ensure item_col exists before grouping
    effective_item_col = item_col if item_col and item_col in all_data.columns else 'Single_Item'
    print(f"Effective item column: {effective_item_col}")
    for item_name, item_data in all_data.groupby(effective_item_col):
        logging.info(f"--- Processing Item: {item_name} ---")
        model_results = find_best_model(item_data['Value'], model_params, return_forecasts=return_best_forecasts, error_metric=error_metric)
        if model_results:
            # Add item name to each model result
            for model_info in model_results:
                model_info['Forecast Item'] = item_name
                best_models.append(model_info)
                
            # Store best model forecast data for visualization if requested
            if return_best_forecasts and len(model_results) > 0:
                best_model = model_results[0]  # First model is the best (lowest MAPE)
                if 'forecast' in best_model:
                    best_forecasts[item_name] = {
                        'model': best_model['model'],
                        'error': best_model['error'],
                        'error_metric': best_model['error_metric'],
                        'train_data': best_model['train_data'],
                        'test_data': best_model['test_data'],
                        'forecast': best_model['forecast'],
                        'full_data': best_model['full_data'],
                        'params': {k: v for k, v in best_model.items() if k.startswith('param_')}
                    }

    if not best_models:
        logging.warning("No best models found for any item.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(best_models)

    # Reorder columns for clarity
    cols = ['Forecast Item', 'model', 'error', 'error_metric'] + [col for col in summary_df.columns if col.startswith('param_')]
    summary_df = summary_df[cols]
    summary_df = summary_df.sort_values(by=['Forecast Item', 'error']).reset_index(drop=True)

    # Clean up parameter column names
    summary_df.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in summary_df.columns]

    try:
        summary_df.to_excel(output_path, index=False)
        logging.info(f"Simplified best models summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")

    if return_best_forecasts:
        return summary_df, best_forecasts
    else:
        return summary_df, None

# --- Main Execution Example ---
if __name__ == "__main__":
    # Example usage:
    # Replace with your actual file path and column names
    # Assuming your Excel has 'Date', 'Value', and optionally 'Forecast Item'
    # If no 'Forecast Item' column, it will treat the whole dataset as one item.
    try:
        results = run_simplified_forecast(
            file_path="Outliers.xlsx", # REQUIRED: Update with your weekly data file
            skiprows=2,                # Optional: Update if needed
            date_col='Date',           # REQUIRED: Update with your date column name
            value_col='Actual.1',      # REQUIRED: Update with your value column name
            item_col='Forecast Item'   # Optional: Update with your item column name (or set to None)
        )
        if not results.empty:
            print("\n--- Best Model Summary ---")
            print(results)
            print("-" * 30)
        else:
            print("No results generated.")

    except FileNotFoundError:
        print("\nERROR: Input file not found. Please update 'file_path' in the script.")
    except KeyError:
        print("\nERROR: Column name mismatch. Please update 'date_col', 'value_col', or 'item_col' in the script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
