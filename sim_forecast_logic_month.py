# -*- coding: utf-8 -*-
"""
Simplified Demand Forecasting Logic

Focuses on monthly data, 70/30 train/test split, and selects the best model
based on monthly MAPE on the test set.
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
warnings.filterwarnings("ignore", category=UserWarning) # To ignore some statsmodels warnings
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

# --- Default Parameters (can be overridden) ---
# Default error metric to use for model selection
DEFAULT_ERROR_METRIC = 'smape'  # Options: 'mape', 'smape', 'rmse'

DEFAULT_PARAMS = {
    "ma_windows": list(range(2, 19)), # Moving average windows: 2 to 18 months
    "es_alphas": np.round(np.arange(0.01, 1.0, 0.01), 2).tolist(), # Smoothing levels for ES
    "croston_alphas": np.round(np.arange(0.01, 1.0, 0.01), 2).tolist(), # Smoothing levels for Croston
    "mlr_seasonalities": list(range(2, 12)), # Seasonal cycles for MLR: quarterly, semi-annually, annually
    "arima_orders": [
        (p, d, q) for p in range(3) for d in range(2) for q in range(3) # (0-2, 0-1, 0-2)
    ], # Expanded ARIMA orders
    "arima_trends": ['n', 'c', 't', 'ct'], # 'none', 'constant', 'trend', 'constant+trend'
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
    first_nonzero_idx = series[series != 0].index.min() if (series != 0).any() else None
    if first_nonzero_idx is None:
        return series
    return series.loc[first_nonzero_idx:]

def fill_missing_months(
    df: pd.DataFrame,
    date_col: str = "Date",
    item_col: Optional[str] = "Forecast Item"
) -> pd.DataFrame:
    """
    Pad every Forecast Item with the full monthly range between the
    global min-date and max-date. Missing months are inserted with Value=0.
    Assumes dates are generally month-start or month-end.
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col not in df.columns:
            raise KeyError(
                f"{date_col!r} not found in columns and the index "
                "is not a DatetimeIndex."
            )
        df = df.set_index(date_col) # Date column becomes index

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    date_idx = df.index

    # Infer monthly frequency (fallback to Month Start 'MS')
    # Normalize to month start for consistent frequency if possible
    # This helps pd.infer_freq if dates are, e.g., all 1st of month or all end of month
    try:
        # Attempt to make dates month-start to help inference, if not already
        # This is a heuristic. If data is truly mid-month, infer_freq might still struggle.
        normalized_dates = date_idx.to_period('M').to_timestamp('MS')
        inferred_freq = pd.infer_freq(normalized_dates.sort_values())
    except AttributeError: # Handle cases where index might not have to_period (e.g. non-datetime)
        inferred_freq = None
        
    freq = inferred_freq or "MS" # Default to Month Start if inference fails or not clear

    full_range = pd.date_range(
        start=date_idx.min(),
        end=date_idx.max(),
        freq=freq,
        name="Date",
    )

    is_multi = (
        item_col
        and item_col in df.columns
        and df[item_col].nunique() > 1
    )

    if not is_multi:
        out = df.reindex(full_range)
        out["Value"] = out["Value"].fillna(0)
        # If original df had item_col, ensure it's preserved
        if item_col and item_col in df.columns and item_col not in out.columns:
             out[item_col] = df[item_col].iloc[0] if not df.empty else "Single_Item_Padded"
        return out

    pieces = []
    for item, grp in df.groupby(item_col):
        padded = grp.reindex(full_range)
        padded["Value"] = padded["Value"].fillna(0)
        padded[item_col] = item
        pieces.append(padded)

    out = pd.concat(pieces)
    out = out.sort_index()
    return out

def load_monthly_data(file_path: str, sheet_name: Optional[str] = 0, skiprows: int = 0, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = 'Forecast Item', fill_missing_months_flag: bool = True, skip_leading_zeros: bool = False) -> pd.DataFrame:
    """
    Loads and prepares monthly data from an Excel file.
    If fill_missing_months_flag is True (default), fills missing monthly dates with zeros.
    If False, missing months are omitted.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
        logging.info(f"Loaded monthly data from {file_path}")

        df = df.rename(columns={date_col: 'Date', value_col: 'Value'})
        if item_col and item_col in df.columns:
             df = df[[item_col, 'Date', 'Value']]
        else:
            df = df[['Date', 'Value']]
            # Create a default item column if it doesn't exist
            # df[item_col if item_col else 'Forecast Item'] = 'Single_Item' # Use provided or default name
            # Let's use a fixed default name if item_col is None for simplicity in grouping later
            df['Forecast Item_temp'] = 'Single_Item' # Assign a default item name
            item_col = 'Forecast Item_temp' # Use this temporary name


        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])

        # Ensure a monthly frequency on the index if possible
        # This helps standardize before filling missing months
        try:
            # Normalize to month start, then try to set freq
            df.index = df.index.to_period('M').to_timestamp('MS')
            df = df.asfreq('MS') # This might introduce NaNs if original data wasn't perfectly MS
        except Exception as e:
            logging.warning(f"Could not forcefully set monthly frequency on index: {e}. Proceeding with inferred/existing.")


        if fill_missing_months_flag:
            df = fill_missing_months(df, date_col='Date', item_col=item_col)
        
        if skip_leading_zeros and not df.empty:
            if item_col and item_col in df.columns and df[item_col].nunique() > 1:
                groups = []
                for name, group in df.groupby(item_col):
                    processed_series = remove_leading_zeros(group['Value'])
                    processed_group = group.loc[processed_series.index]
                    groups.append(processed_group)
                if groups:
                    df = pd.concat(groups)
            else:
                processed_series = remove_leading_zeros(df['Value'])
                df = df.loc[processed_series.index]
        
        # If we used a temporary item column, rename it to a standard one if the original was None
        if 'Forecast Item_temp' in df.columns and item_col == 'Forecast Item_temp':
            df = df.rename(columns={'Forecast Item_temp': 'Forecast Item'})


        return df

    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise
    except KeyError as e:
        logging.error(f"Error: Column not found - {e}. Check column names (date_col, value_col, item_col).")
        raise
    except Exception as e:
        logging.error(f"Error loading monthly data: {e}")
        raise

def split_data(df: pd.DataFrame, train_split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and testing sets."""
    split_point = int(len(df) * train_split_ratio)
    train_data = df.iloc[:split_point]
    test_data = df.iloc[split_point:]
    logging.info(f"Data split: Train {len(train_data)} rows, Test {len(test_data)} rows")
    return train_data, test_data

def calculate_error_metric(actual: pd.Series, forecast: pd.Series, metric: Literal['mape', 'smape', 'rmse'] = 'smape') -> float:
    """Calculates specified error metric ensuring alignment and handling zeros."""
    actual = actual.copy()
    forecast = forecast.copy()
    forecast = forecast.reindex(actual.index)
    mask = actual.notna() & forecast.notna()
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return np.inf
    
    if metric == 'mape':
        mask_zero = actual != 0
        if not mask_zero.all():
            actual = actual[mask_zero]
            forecast = forecast[mask_zero]
            if len(actual) == 0: return np.inf
        return np.abs((actual - forecast) / actual).mean() * 100
    elif metric == 'smape':
        numerator = np.abs(forecast - actual)
        denominator = np.abs(forecast) + np.abs(actual)
        mask_zero = denominator != 0
        if not mask_zero.all():
            numerator = numerator[mask_zero]
            denominator = denominator[mask_zero]
            if len(numerator) == 0: return np.inf
        return (200 * numerator / denominator).mean()
    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(actual, forecast))
    else: # Default to SMAPE
        return calculate_error_metric(actual, forecast, 'smape')

# --- Model Implementations (Adjusted for Monthly) ---

def forecast_moving_average(train_data: pd.Series, test_len: int, window: int) -> Optional[pd.Series]:
    """Generates forecasts using a simple moving average."""
    if len(train_data) < window:
        logging.warning(f"MA: Not enough training data ({len(train_data)}) for window {window}")
        return None
    last_train_avg = train_data.iloc[-window:].mean()
    # Ensure a monthly frequency for the forecast index
    current_freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
    if isinstance(current_freq, str):
        current_freq = pd.tseries.frequencies.to_offset(current_freq)

    forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1) # Ensure it starts next month
    
    # If original data was, e.g. end of month, try to preserve that.
    # A simple heuristic: if original freq was 'M', use 'M'. Otherwise 'MS'.
    forecast_freq = 'M' if hasattr(current_freq, 'rule_code') and current_freq.rule_code == 'M' else 'MS'
        
    forecast_index = pd.date_range(start=forecast_start_date.normalize() if forecast_freq=='MS' else (forecast_start_date + pd.offsets.MonthEnd(0)), 
                                   periods=test_len, freq=forecast_freq)

    return pd.Series([last_train_avg] * test_len, index=forecast_index)


def forecast_exponential_smoothing(train_data: pd.Series, test_len: int, alpha: float) -> Optional[pd.Series]:
    """Generates forecasts using simple exponential smoothing."""
    try:
        # Ensure train_data has a frequency for SimpleExpSmoothing
        if train_data.index.freq is None:
            inferred_freq = pd.infer_freq(train_data.index)
            if inferred_freq:
                train_data = train_data.asfreq(inferred_freq)
            else: # Fallback if still no freq
                train_data = train_data.asfreq('MS') # Default to Month Start
                logging.warning("ES: Training data frequency not found, defaulted to 'MS'.")

        model = sm.tsa.SimpleExpSmoothing(train_data, initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
        forecast_values = model.forecast(steps=test_len)
        
        current_freq_offset = train_data.index.freq or pd.tseries.frequencies.to_offset('MS')
        forecast_start_date = train_data.index[-1] + current_freq_offset
        
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=current_freq_offset)
        return pd.Series(forecast_values, index=forecast_index)
    except Exception as e:
        logging.error(f"ES Error (alpha={alpha}): {e}")
        return None

def forecast_linear_regression(train_data: pd.Series, test_len: int) -> Optional[pd.Series]:
    """Generates forecasts using linear regression (trend)."""
    try:
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values
        model = LinearRegression().fit(X_train, y_train)
        
        logging.info(f"Linear Regression Coefficients: {model.coef_}")
        logging.info(f"Linear Regression Intercept: {model.intercept_}")

        X_test = np.arange(len(train_data), len(train_data) + test_len).reshape(-1, 1)
        forecast_values = model.predict(X_test)
        
        current_freq_offset = train_data.index.freq or pd.tseries.frequencies.to_offset('MS')
        forecast_start_date = train_data.index[-1] + current_freq_offset
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=current_freq_offset)
        
        return pd.Series(forecast_values, index=forecast_index)
    except Exception as e:
        logging.error(f"LR Error: {e}")
        return None

def forecast_multiple_linear_regression(train_data: pd.Series, test_len: int, seasonality: int) -> Optional[pd.Series]:
    """
    Forecast next `test_len` periods using multiple linear regression
    with monthly seasonality and linear trend.
    `seasonality` is the number of months in a cycle (e.g., 12 for yearly).
    """
    # The test_len+2 and slicing preds[2:] seems specific, will keep for direct conversion
    # Consider reviewing this if it causes issues with monthly data or if its purpose is unknown
    effective_test_len = test_len + 2 

    # Ensure train_data has a DatetimeIndex and monthly frequency
    if not isinstance(train_data.index, pd.DatetimeIndex):
        train_data.index = pd.to_datetime(train_data.index)
    
    original_freq = train_data.index.freq or pd.infer_freq(train_data.index)
    if not original_freq:
        train_data = train_data.asfreq('MS') # Default to Month Start
        logging.warning("MLR: Training data frequency not found or irregular, defaulted to 'MS'.")
        original_freq = train_data.index.freq
    elif isinstance(original_freq, str):
         train_data = train_data.asfreq(original_freq) # Ensure it's set
         original_freq = train_data.index.freq


    # Trend feature: days since the start of the series
    numeric_date_train = (train_data.index - train_data.index.min()).days
    
    # Seasonal feature: month number within the specified seasonal cycle
    # Calculate months from start to create a 0-indexed seasonal period
    month_sequence_train = (train_data.index.year - train_data.index.min().year) * 12 + \
                           (train_data.index.month - train_data.index.min().month)
    seasonal_period_train = month_sequence_train % seasonality
    
    X_train_seasonal = pd.get_dummies(seasonal_period_train, prefix=f'month_cycle_s{seasonality}')
    # Ensure all possible seasonal dummy columns exist
    for i in range(seasonality):
        col_name = f'month_cycle_s{seasonality}_{i}'
        if col_name not in X_train_seasonal.columns:
            X_train_seasonal[col_name] = 0
    X_train_seasonal = X_train_seasonal.sort_index(axis=1) # Consistent column order

    X_train = X_train_seasonal
    X_train['trend'] = numeric_date_train
    y_train = train_data.values
    
    model = LinearRegression().fit(X_train, y_train)
    
    last_date_train = train_data.index.max()
    # Use the determined frequency of the training data for future dates
    future_index = pd.date_range(start=last_date_train + original_freq, periods=effective_test_len, freq=original_freq)
    
    numeric_date_test = (future_index - train_data.index.min()).days
    month_sequence_test = (future_index.year - train_data.index.min().year) * 12 + \
                          (future_index.month - train_data.index.min().month)
    seasonal_period_test = month_sequence_test % seasonality
                          
    X_test_seasonal = pd.get_dummies(seasonal_period_test, prefix=f'month_cycle_s{seasonality}')
    for i in range(seasonality): # Ensure all columns
        col_name = f'month_cycle_s{seasonality}_{i}'
        if col_name not in X_test_seasonal.columns:
            X_test_seasonal[col_name] = 0
    X_test_seasonal = X_test_seasonal.sort_index(axis=1) # Consistent column order
    
    X_test = X_test_seasonal
    X_test['trend'] = numeric_date_test
    
    preds = model.predict(X_test)
    preds = np.maximum(0, preds) # Enforce non-negativity
    
    # Apply the original slicing logic
    return pd.Series(preds[2:], index=future_index[:-2], name='forecast')


def forecast_croston(train_data: pd.Series, test_len: int, alpha: float = 0.1) -> Optional[pd.Series]:
    """Generates forecasts using Croston's method for intermittent demand (monthly)."""
    if len(train_data) < 2:
        logging.warning(f"Croston: Not enough data ({len(train_data)})")
        return None

    demand = train_data.values
    n = len(demand)
    Z, X, fitted = np.zeros(n), np.zeros(n), np.zeros(n)
    first_idx = next((i for i, v in enumerate(demand) if v != 0), None)

    if first_idx is None:
        logging.warning("Croston: No non-zero demands in training data")
        # Return a series of zeros for the forecast period
        current_freq_offset = train_data.index.freq or pd.tseries.frequencies.to_offset('MS')
        forecast_start_date = train_data.index[-1] + current_freq_offset
        idx = pd.date_range(start=forecast_start_date, periods=test_len, freq=current_freq_offset)
        return pd.Series([0] * test_len, index=idx)


    Z[first_idx], X[first_idx], q = demand[first_idx], 1, 1
    for i in range(first_idx + 1, n):
        if demand[i] != 0:
            Z[i] = alpha * demand[i] + (1 - alpha) * Z[i - 1]
            X[i] = alpha * q + (1 - alpha) * X[i - 1]
            q = 1
        else:
            Z[i], X[i] = Z[i - 1], X[i - 1]
            q += 1
        fitted[i] = Z[i - 1] / X[i - 1] if X[i - 1] > 0 else 0
    
    forecast_value = Z[-1] / X[-1] if X[-1] > 0 else 0
    
    current_freq_offset = train_data.index.freq or pd.infer_freq(train_data.index) or pd.tseries.frequencies.to_offset('MS')
    if isinstance(current_freq_offset, str): # Ensure it's an offset
        current_freq_offset = pd.tseries.frequencies.to_offset(current_freq_offset)

    forecast_start_date = train_data.index[-1] + current_freq_offset
    idx = pd.date_range(start=forecast_start_date, periods=test_len, freq=current_freq_offset)
    return pd.Series([forecast_value] * test_len, index=idx)

def forecast_arima(
    train_data: pd.Series,
    test_len: int,
    arima_order: Tuple[int, int, int],
    trend_param: str = "n"
) -> Optional[pd.Series]:
    """Generates forecasts using an ARIMA model (monthly)."""
    p, d, q = arima_order
    # Statsmodels convention: for d > 0, trend 'c' (constant) is often not directly used,
    # 't' (trend) or 'ct' (constant+trend) might be more appropriate or 'n' (none).
    # If d > 0, 'c' can be absorbed by differencing, so 'ct' becomes effectively a trend, and 'c' might be redundant.
    # No explicit change here, relying on statsmodels to handle it or user to choose appropriate trend.
    # The original warning for d>0 and trend='c' to 'ct' can be kept if it's preferred behaviour.
    if d > 0 and trend_param == 'c':
         logging.warning(f"ARIMA: For order {arima_order} with d={d}, trend 'c' might be implicitly handled or better as 't' or 'ct'. Using '{trend_param}'.")
         # Original code changed 'c' to 't'. Let's keep it if that was intended:
         # trend_param = 't' 


    min_data_needed = max(p, q) + d + 1 # A rough estimate
    if len(train_data) < min_data_needed:
        logging.warning(f"ARIMA: Not enough training data ({len(train_data)}) for order={arima_order}, trend={trend_param}. Need at least {min_data_needed}.")
        return None
    if test_len <= 0: return None

    try:
        # Ensure train_data has a frequency for ARIMA
        if train_data.index.freq is None:
            inferred_freq = pd.infer_freq(train_data.index)
            if inferred_freq:
                train_data = train_data.asfreq(inferred_freq)
            else: # Fallback if still no freq
                train_data = train_data.asfreq('MS') # Default to Month Start for monthly data
                logging.warning("ARIMA: Training data frequency not found, defaulted to 'MS'.")
        
        model = ARIMA(train_data, order=arima_order, trend=trend_param, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        
        forecast_values = model_fit.forecast(steps=test_len)
        
        # Use the frequency from the (potentially modified) training data
        current_freq_offset = train_data.index.freq
        forecast_start_date = train_data.index[-1] + current_freq_offset
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=current_freq_offset)
        
        return pd.Series(forecast_values, index=forecast_index)
    except Exception as e:
        logging.error(f"ARIMA Error(order={arima_order}, trend={trend_param}): {e}")
        return None

# --- Orchestration ---

def find_best_model(item_data: pd.Series, params: Dict = DEFAULT_PARAMS, return_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Optional[List[Dict[str, Any]]]:
    """Evaluates multiple models and returns all models sorted by the chosen error metric."""
    if item_data.empty or len(item_data) < 4:
        logging.warning(f"Skipping item {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}: Not enough data ({len(item_data)})")
        return None

    if item_data.index.freq is None:
        inferred = pd.infer_freq(item_data.index)
        if inferred:
            item_data = item_data.asfreq(inferred)
        else:
            item_data = item_data.asfreq('MS') # Default to Month Start
            logging.warning(f"Item {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}: Frequency not found, defaulted to 'MS'.")


    train_len = int(len(item_data) * 0.7)
    if train_len < 2:
         logging.warning(f"Skipping item {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}: Not enough training data ({train_len}) after split")
         return None
    train_data = item_data.iloc[:train_len]
    test_data = item_data.iloc[train_len:]
    test_len = len(test_data)
    
    full_data = item_data.copy()
    if test_len == 0:
        logging.warning(f"Skipping item {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}: No test data after split")
        return None

    unique_results = {}

    # 1. Moving Average
    for w in params.get("ma_windows", []):
        forecast = forecast_moving_average(train_data, test_len, w)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Moving Average_window_{w}"
            result_dict = {"model": "Moving Average", "param_window": w, "error": error_value, "error_metric": error_metric}
            if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
            unique_results[model_key] = result_dict

    # 2. Exponential Smoothing
    for alpha in params.get("es_alphas", []):
        forecast = forecast_exponential_smoothing(train_data, test_len, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Exponential Smoothing_alpha_{alpha}"
            result_dict = {"model": "Exponential Smoothing", "param_alpha": alpha, "error": error_value, "error_metric": error_metric}
            if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
            unique_results[model_key] = result_dict
    
    # 3. Linear Regression
    forecast = forecast_linear_regression(train_data, test_len)
    if forecast is not None:
        error_value = calculate_error_metric(test_data, forecast, error_metric)
        model_key = "Linear Regression"
        result_dict = {"model": "Linear Regression", "param_trend_type": "linear", "error": error_value, "error_metric": error_metric}
        if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
        unique_results[model_key] = result_dict

    # 4. Multiple Linear Regression (with Seasonality)
    for seas in params.get("mlr_seasonalities", []):
        if len(train_data) < seas + 1 : # Need enough data for seasonality
            logging.warning(f"MLR: Not enough training data for seasonality {seas}")
            continue
        forecast = forecast_multiple_linear_regression(train_data, test_len, seas)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"MLR_seasonality_{seas}"
            result_dict = {"model": "MLR", "param_seasonality_cycle": seas, "error": error_value, "error_metric": error_metric}
            if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
            unique_results[model_key] = result_dict

    # 5. Croston
    for alpha in params.get("croston_alphas", []):
        forecast = forecast_croston(train_data, test_len, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data, forecast, error_metric)
            model_key = f"Croston_alpha_{alpha}"
            result_dict = {"model": "Croston", "param_alpha": alpha, "error": error_value, "error_metric": error_metric}
            if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
            unique_results[model_key] = result_dict

    # 6. ARIMA
    for arima_order in params.get("arima_orders", []):
        for trend in params.get("arima_trends", ['n']):
            forecast = forecast_arima(train_data, test_len, arima_order, trend_param=trend)
            if forecast is not None:
                error_value = calculate_error_metric(test_data, forecast, error_metric)
                model_key = f"ARIMA_order_{arima_order}_trend_{trend}"
                result_dict = {"model": "ARIMA", "param_order": str(arima_order), "param_trend": trend, "error": error_value, "error_metric": error_metric}
                if return_forecasts: result_dict.update({"train_data": train_data, "test_data": test_data, "forecast": forecast, "full_data": full_data})
                unique_results[model_key] = result_dict

    results = list(unique_results.values())
    if not results:
        logging.warning(f"No successful models found for item {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}")
        return None

    results.sort(key=lambda x: x['error'])
    best_result = results[0]
    param_info = {k: v for k, v in best_result.items() if k.startswith('param_')}
    logging.info(f"Best model for {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}: {best_result['model']} with params {param_info} -> {best_result['error_metric'].upper()}: {best_result['error']:.4f}")
    logging.info(f"Found {len(results)} unique model configurations for {item_data.name if hasattr(item_data, 'name') else 'Unnamed Series'}")

    return results


def run_monthly_forecast(file_path: str, sheet_name: Optional[str] = 0, skiprows: int = 0, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = 'Forecast Item', output_path: str = "monthly_best_models.xlsx", model_params: Dict = DEFAULT_PARAMS, fill_missing_months_flag: bool = True, skip_leading_zeros: bool = False, return_best_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Loads monthly data, runs model selection for each item, and saves the best models.
    """
    all_data = load_monthly_data(file_path, sheet_name, skiprows, date_col, value_col, item_col, fill_missing_months_flag=fill_missing_months_flag, skip_leading_zeros=skip_leading_zeros)
    
    if all_data.empty:
        logging.error("No data loaded. Exiting.")
        return pd.DataFrame(), None
        
    logging.info(f"All data shape after loading: {all_data.shape}")
    
    best_models_records = []
    best_forecasts_dict = {}
    
    # Determine the actual item column name used (could be original or default 'Forecast Item')
    # This logic needs to align with how load_monthly_data handles item_col when it's None
    if item_col and item_col in all_data.columns:
        grouping_col = item_col
    elif 'Forecast Item' in all_data.columns: # Default name if load_monthly_data created it
        grouping_col = 'Forecast Item'
    else: # Should not happen if load_monthly_data ensures an item column
        logging.error("Item column for grouping not found in loaded data.")
        return pd.DataFrame(), None


    for item_name, item_data_group in all_data.groupby(grouping_col):
        logging.info(f"--- Processing Item: {item_name} ---")
        # Ensure 'Value' column exists in the group
        if 'Value' not in item_data_group.columns:
            logging.error(f"Column 'Value' not found for item {item_name}. Skipping.")
            continue
        
        item_series = item_data_group['Value'].copy()
        item_series.name = str(item_name) # Set series name for logging inside find_best_model

        model_results_for_item = find_best_model(item_series, model_params, return_forecasts=return_best_forecasts, error_metric=error_metric)
        
        if model_results_for_item:
            for model_info in model_results_for_item:
                model_info[grouping_col] = item_name # Use the actual grouping column name
                best_models_records.append(model_info)
                
            if return_best_forecasts and model_results_for_item:
                best_model_for_item = model_results_for_item[0]
                if 'forecast' in best_model_for_item:
                    best_forecasts_dict[item_name] = {
                        'model': best_model_for_item['model'],
                        'error': best_model_for_item['error'],
                        'error_metric': best_model_for_item['error_metric'],
                        'train_data': best_model_for_item['train_data'],
                        'test_data': best_model_for_item['test_data'],
                        'forecast': best_model_for_item['forecast'],
                        'full_data': best_model_for_item['full_data'],
                        'params': {k: v for k, v in best_model_for_item.items() if k.startswith('param_')}
                    }

    if not best_models_records:
        logging.warning("No best models found for any item.")
        return pd.DataFrame(), None if return_best_forecasts else None

    summary_df = pd.DataFrame(best_models_records)
    
    # Define preferred column order, ensuring grouping_col is first
    cols_ordered = [grouping_col, 'model', 'error', 'error_metric']
    param_cols = sorted([col for col in summary_df.columns if col.startswith('param_')])
    cols_ordered.extend(param_cols)
    # Add any other columns that might have been missed (e.g., if return_forecasts was True but not primary interest for summary)
    other_cols = [col for col in summary_df.columns if col not in cols_ordered]
    cols_ordered.extend(other_cols)
    
    # Filter to existing columns only to avoid KeyError
    cols_ordered = [col for col in cols_ordered if col in summary_df.columns]

    summary_df = summary_df[cols_ordered]
    summary_df = summary_df.sort_values(by=[grouping_col, 'error']).reset_index(drop=True)
    summary_df.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in summary_df.columns]

    try:
        summary_df.to_excel(output_path, index=False)
        logging.info(f"Monthly best models summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")

    return summary_df, best_forecasts_dict if return_best_forecasts else None