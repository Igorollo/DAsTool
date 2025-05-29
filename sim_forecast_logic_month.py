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
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

# --- Default Parameters (can be overridden) ---
# Default error metric to use for model selection
DEFAULT_ERROR_METRIC = 'smape'  # Options: 'mape', 'smape', 'rmse'

DEFAULT_PARAMS = {
    "ma_windows": list(range(2, 27)), # Expanded: 2 to 26 months
    "es_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(), # Kept standard range
    "croston_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(), # Added for clarity, same as ES
    "mlr_seasonalities": list(range(4, 27)), # Expanded: 4 to 26 months (e.g., 12 for annual)
    "arima_orders": [
        (p, 0, q) for p in range(3) for q in range(3)
        ] + [(p, 1, q) for p in range(3) for q in range(3)
    ],
    "arima_trends": ['n', 'c', 'ct'],
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

    Works whether the frame already has a DatetimeIndex *or* still has a
    plain `date_col` column.
    """
    if df.empty:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        month_index = df.index
    else:
        if date_col not in df.columns:
            raise KeyError(
                f"{date_col!r} not found in columns and the index "
                "is not a DatetimeIndex."
            )
        df = df.set_index(date_col)
        month_index = df.index

    freq = pd.infer_freq(month_index.sort_values()) or "MS" # Default to Month Start

    full_range = pd.date_range(
        start=month_index.min(),
        end=month_index.max(),
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
    If False, missing months are omitted (no zeros inserted).
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
        logging.info(f"Loaded monthly data from {file_path}")

        df = df.rename(columns={date_col: 'Date', value_col: 'Value'})
        if item_col and item_col in df.columns:
             df = df[[item_col, 'Date', 'Value']]
        else:
            df = df[['Date', 'Value']]
            df[item_col] = 'Single_Item'

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])

        if fill_missing_months_flag:
            df = fill_missing_months(df, date_col='Date', item_col=item_col)
        print("FILLING COMPLETE")

        if skip_leading_zeros and not df.empty:
            if item_col and item_col in df.columns and len(df[item_col].unique()) > 1:
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
        logging.warning("No valid points for error calculation after alignment.")
        return np.inf

    if metric == 'mape':
        mask_zero = actual != 0
        if not mask_zero.all():
            logging.warning(f"Zeros found in actual values during MAPE calculation. These points ({sum(~mask_zero)}) will be excluded.")
            actual = actual[mask_zero]
            forecast = forecast[mask_zero]

            if len(actual) == 0:
                logging.warning("No valid points for MAPE calculation after zero handling.")
                return np.inf

        abs_percentage_errors = np.abs((actual - forecast) / actual) * 100
        return abs_percentage_errors.mean()

    elif metric == 'smape':
        numerator = np.abs(forecast - actual)
        denominator = np.abs(forecast) + np.abs(actual)

        mask_zero = denominator != 0
        if not mask_zero.all():
            logging.warning(f"Zero denominators found during SMAPE calculation. These points ({sum(~mask_zero)}) will be excluded.")
            numerator = numerator[mask_zero]
            denominator = denominator[mask_zero]

            if len(numerator) == 0:
                logging.warning("No valid points for SMAPE calculation after zero handling.")
                return np.inf

        smape_values = 200 * numerator / denominator
        return smape_values.mean()

    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(actual, forecast))

    else:
        logging.warning(f"Unknown error metric: {metric}. Using SMAPE instead.")
        return calculate_error_metric(actual, forecast, 'smape')

# --- Model Implementations (Simplified) ---

def forecast_moving_average(train_data: pd.Series, test_len: int, window: int) -> Optional[pd.Series]:
    """Generates forecasts using a simple moving average."""
    if len(train_data) < window:
        logging.warning(f"MA: Not enough training data ({len(train_data)}) for window {window}")
        return None
    last_train_avg = train_data.iloc[-window:].mean()
    
    freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
    forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
    forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
    
    forecast = pd.Series([last_train_avg] * test_len, index=forecast_index)
    return forecast

def forecast_exponential_smoothing(train_data: pd.Series, test_len: int, alpha: float) -> Optional[pd.Series]:
    """Generates forecasts using simple exponential smoothing."""
    try:
        model = sm.tsa.SimpleExpSmoothing(train_data, initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
        forecast_values = model.forecast(steps=test_len)
        
        freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
        
        forecast = pd.Series(forecast_values, index=forecast_index)
        return forecast
    except Exception as e:
        logging.error(f"ES Error (alpha={alpha}): {e}")
        return None

def forecast_linear_regression(train_data: pd.Series, test_len: int) -> Optional[pd.Series]:
    """Generates forecasts using linear regression (trend)."""
    try:
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # The print statements for coefficients might need context for monthly data.
        # For instance, model.coef_ is the trend per month.
        # The original print statements were likely placeholders or specific to a weekly context.
        # print("Linear Regression Coefficients (per period):")
        # print(model.coef_)
        # print("Linear Regression Intercept:")
        # print(model.intercept_)

        X_test = np.arange(len(train_data), len(train_data) + test_len).reshape(-1, 1)
        forecast_values = model.predict(X_test)

        freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
        
        forecast = pd.Series(forecast_values, index=forecast_index)
        return forecast
    except Exception as e:
        logging.error(f"LR Error: {e}")
        return None

def forecast_multiple_linear_regression(train_data: pd.Series, test_len: int, seasonality: int) -> pd.Series:
    """
    Forecast next `test_len` periods using multiple linear regression
    with monthly seasonality and linear trend, ignoring decay and evaluation.
    """
    test_len_adj = test_len # Keep original test_len for forecast series index
    # The original code added 2 to test_len and then sliced. Replicating if necessary,
    # but for monthly data, direct forecasting is often cleaner.
    # For now, let's assume test_len is the exact number of future periods needed.

    current_freq = train_data.index.freq or pd.infer_freq(train_data.index)
    if not current_freq:
        logging.warning("MLR: Could not infer frequency for train_data. Assuming 'MS'.")
        # Ensure train_data has a frequency before proceeding
        train_data = train_data.asfreq('MS') # Or handle error
        current_freq = 'MS'
    
    # Ensure DateTimeIndex with freq
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
         # Attempt to set frequency if not already set, common for monthly data from various sources
        train_data.index = pd.DatetimeIndex(train_data.index, freq=current_freq)
        if train_data.index.freq is None: # Still none after attempt
            raise ValueError("train_data must have a regular DateTimeIndex with a defined freq.")


    # Numeric time trend (e.g., months since start of data)
    time_trend_train = np.arange(len(train_data))

    # Seasonality feature: month index within the cycle of 'seasonality' months
    month_in_cycle_train = time_trend_train % seasonality

    X_train_df = pd.DataFrame({
        'time_trend': time_trend_train
    })

    # One-hot encode seasonality
    seasonal_dummies_train = pd.get_dummies(month_in_cycle_train, prefix=f'cycle_month', drop_first=False)
    # Ensure all possible seasonal columns are present (0 to seasonality-1)
    for i in range(seasonality):
        col_name = f'cycle_month_{i}'
        if col_name not in seasonal_dummies_train.columns:
            seasonal_dummies_train[col_name] = 0
    seasonal_dummies_train = seasonal_dummies_train[[f'cycle_month_{i}' for i in range(seasonality)]]


    X_train = pd.concat([X_train_df, seasonal_dummies_train], axis=1)
    y_train = train_data.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    last_date_train = train_data.index.max()
    # Use the determined/assigned frequency of the training data
    forecast_index = pd.date_range(start=last_date_train + pd.DateOffset(months=1), periods=test_len_adj, freq=train_data.index.freq)

    time_trend_test = np.arange(len(train_data), len(train_data) + test_len_adj)
    month_in_cycle_test = time_trend_test % seasonality
    
    X_test_df = pd.DataFrame({
        'time_trend': time_trend_test
    })
    seasonal_dummies_test = pd.get_dummies(month_in_cycle_test, prefix=f'cycle_month', drop_first=False)
    for i in range(seasonality):
        col_name = f'cycle_month_{i}'
        if col_name not in seasonal_dummies_test.columns:
            seasonal_dummies_test[col_name] = 0 # Add missing columns with 0
    # Ensure consistent column order with training
    seasonal_dummies_test = seasonal_dummies_test[[f'cycle_month_{i}' for i in range(seasonality)]]


    X_test = pd.concat([X_test_df, seasonal_dummies_test], axis=1)
    
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)

    return pd.Series(preds, index=forecast_index, name='forecast')


def forecast_croston(train_data: pd.Series, test_len: int, alpha: float = 0.1) -> Optional[pd.Series]:
    """Generates forecasts using Croston's method for intermittent demand."""
    if len(train_data) < 2:
        logging.warning(f"Croston: Not enough data ({len(train_data)})")
        return None

    demand = train_data.values
    n = len(demand)

    Z = np.zeros(n)
    X = np.zeros(n)
    fitted = np.zeros(n)

    first_idx = next((i for i, v in enumerate(demand) if v != 0), None)
    if first_idx is None:
        logging.warning("Croston: No non-zero demands in training data")
        # Return a series of zeros for the forecast period if no demand
        freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
        idx = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
        return pd.Series([0] * test_len, index=idx)


    Z[first_idx] = demand[first_idx]
    X[first_idx] = 1
    q = 1

    for i in range(first_idx + 1, n):
        if demand[i] != 0:
            Z[i] = alpha * demand[i] + (1 - alpha) * Z[i - 1]
            X[i] = alpha * q + (1 - alpha) * X[i - 1]
            q = 1
        else:
            Z[i] = Z[i - 1]
            X[i] = X[i - 1]
            q += 1
        if X[i - 1] > 0:
             fitted[i] = Z[i - 1] / X[i - 1]
        else:
             fitted[i] = 0

    if X[-1] > 0:
        forecast_value = Z[-1] / X[-1]
    else:
        forecast_value = 0

    freq = train_data.index.freq or pd.infer_freq(train_data.index)
    if freq is None:
        logging.warning("Croston: Cannot determine frequency from training data index. Assuming 'MS'.")
        freq = 'MS' # Default to Month Start

    forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
    idx = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)

    return pd.Series([forecast_value] * test_len, index=idx)

def forecast_arima(
    train_data: pd.Series,
    test_len: int,
    arima_order: Tuple[int, int, int],
    trend_param: str = "n"
) -> Optional[pd.Series]:
    """Generates forecasts using an ARIMA model."""
    p, d, q = arima_order
    if d > 0 and trend_param == 'c':
        logging.warning(f"ARIMA: Changing trend from 'c' to 't' for order {arima_order} with d={d} to allow constant with differencing.")
        # 't' is trend, 'ct' is constant + trend. For d > 0, 'c' becomes part of 'ct' effectively.
        # statsmodels uses 't' for trend component when d > 0.
        # If a constant is desired with differencing, it's usually part of the 'ct' or implicitly handled.
        # The original 'ct' for d>0 logic was a bit mixed up.
        # For d > 0, a constant is achieved by `trend='c'` (if model supports it and it means mean of differenced series)
        # or by the intercept in `trend='t'` or `trend='ct'`.
        # Let's simplify: if 'c' is passed for d>0, statsmodels usually handles it.
        # No, statsmodels documentation for ARIMA:
        # trend : {'n', 'c', 't', 'ct'}
        # If d > 0 and trend is 'c', it will raise an error. It should be 'n' or 't' or 'ct'.
        # If user wants a constant with d>0, 'ct' (or 't' if just trend drift) is more appropriate than just 'c'.
        # Let's change 'c' to 'n' if d>0, assuming they wanted a stationary model post-differencing,
        # or let it error out / rely on statsmodels default handling.
        # The original code changed 'c' to 't'.
        # A common choice if a constant is desired with differencing is that the *differenced* series has a mean.
        # This is what `trend='c'` implies for d=0.
        # If d>0, and `trend='c'` is used, it means the d-th differenced series has a non-zero mean.
        # If d > 0, `trend='c'` implies an intercept. `trend='t'` implies a trend. `trend='ct'` implies both.
        # The warning about changing 'c' to 'ct' (or 't' as in original code) is reasonable.
        # Let's stick to the original change to 't' for now.
        trend_param = 't' # As per original code's intent for d>0 with a constant-like term.
                          # A more robust solution might be to allow specific 'ct' if user intends it.

    min_data_needed = sum(arima_order) + d + 1 # A slightly more robust check
    if len(train_data) < min_data_needed:
        logging.warning(
            f"ARIMA: Not enough training data ({len(train_data)}) for order={arima_order}, trend={trend_param}. Need at least {min_data_needed}"
        )
        return None

    if test_len <= 0:
        logging.info("ARIMA: test_len is 0, no forecast to generate.")
        return None

    try:
        # Ensure train_data has a frequency
        current_freq = train_data.index.freq or pd.infer_freq(train_data.index)
        if current_freq is None:
            logging.warning("ARIMA: Frequency not found for train_data. Attempting to set to 'MS'.")
            # Create a new series with inferred or default frequency
            # This is crucial for ARIMA
            train_data_with_freq = train_data.asfreq('MS')
            if train_data_with_freq.isnull().all(): # check if asfreq failed
                 logging.error("ARIMA: Failed to set frequency 'MS', data might be unsuitable.")
                 return None
            train_data = train_data_with_freq # use the new series

        elif not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
             train_data = train_data.asfreq(current_freq) # ensure freq attribute is set


        model = ARIMA(train_data, order=arima_order, trend=trend_param)
        model_fit = model.fit()

        # Forecast index generation handled by statsmodels `forecast` method if index is regular
        # However, explicitly creating it ensures alignment with other models if needed
        raw_fc = model_fit.forecast(steps=test_len)

        # If raw_fc index is not DatetimeIndex (e.g. RangeIndex if original index was not well-defined)
        # or if we want to ensure consistency:
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1) # Assuming monthly data
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=train_data.index.freq or 'MS')

        forecast = pd.Series(raw_fc, index=forecast_index)
        return forecast

    except Exception as e:
        logging.error(f"ARIMA Error(order={arima_order}, trend={trend_param}): {e}")
        if 'Prediction must have `end` after `start`' in str(e): #pragma: no cover
            logging.info(f"ARIMA date range issue with train_data index: {train_data.index[0]} to {train_data.index[-1]}, test_len={test_len}")
        return None

# --- Orchestration ---

def find_best_model(item_data: pd.Series, params: Dict = DEFAULT_PARAMS, return_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Optional[List[Dict[str, Any]]]:
    """Evaluates multiple models and returns all models sorted by specified error metric."""
    if item_data.empty or len(item_data) < 4:
        logging.warning(f"Skipping item {item_data.name}: Not enough data ({len(item_data)})")
        return None

    if item_data.index.freq is None:
        inferred = pd.infer_freq(item_data.index)
        if inferred:
            item_data = item_data.asfreq(inferred)
        else:
            logging.warning(f"Could not infer frequency for {item_data.name}. Assuming 'MS'.")
            item_data = item_data.asfreq('MS') # Default to Month Start
            if item_data.isnull().all(): # Check if asfreq made all NaNs
                logging.error(f"Failed to process {item_data.name} after assuming 'MS' frequency.")
                return None


    train_len = int(len(item_data) * 0.7)
    if train_len < 2:
         logging.warning(f"Skipping item {item_data.name}: Not enough training data ({train_len}) after split")
         return None
    train_data = item_data.iloc[:train_len]
    test_data = item_data.iloc[train_len:]
    test_len = len(test_data)

    full_data = item_data.copy()

    if test_len == 0:
        logging.warning(f"Skipping item {item_data.name}: No test data after split")
        return None

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
            result_dict["train_data"] = train_data
            result_dict["test_data"] = test_data
            result_dict["forecast"] = forecast
            result_dict["full_data"] = full_data
        unique_results[model_key] = result_dict

    # 4. Multiple Linear Regression (with Seasonality)
    for seas in params.get("mlr_seasonalities", []):
        # Ensure seasonality is not too large for the training data length
        if seas >= len(train_data) / 2 and seas > 1: # Heuristic: need at least 2 full cycles for reliable seasonality
            logging.warning(f"MLR: Seasonality {seas} too large for train data length {len(train_data)}. Skipping.")
            continue
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
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 5. Croston
    for alpha in params.get("croston_alphas", []):
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
                result_dict["train_data"] = train_data
                result_dict["test_data"] = test_data
                result_dict["forecast"] = forecast
                result_dict["full_data"] = full_data
            unique_results[model_key] = result_dict

    # 6. ARIMA
    for arima_order in params.get("arima_orders", []):
        for trend in params.get("arima_trends", ['n']):
            forecast = forecast_arima(train_data, test_len, arima_order, trend_param=trend)
            if forecast is not None:
                error_value = calculate_error_metric(test_data, forecast, error_metric)
                model_key = f"ARIMA_order_{arima_order}_trend_{trend}"
                result_dict = {
                    "model": "ARIMA",
                    "param_order": str(arima_order),
                    "param_trend": trend,
                    "error": error_value,
                    "error_metric": error_metric
                }
                if return_forecasts:
                    result_dict["train_data"] = train_data
                    result_dict["test_data"] = test_data
                    result_dict["forecast"] = forecast
                    result_dict["full_data"] = full_data
                unique_results[model_key] = result_dict

    results = list(unique_results.values())

    if not results:
        logging.warning(f"No successful models found for item {item_data.name}")
        return None

    results.sort(key=lambda x: x['error'])
    best_result = results[0] if results else None

    if best_result:
        logging.info(f"Best model for {item_data.name}: {best_result['model']} with params { {k: v for k, v in best_result.items() if k.startswith('param_')} } -> {best_result['error_metric'].upper()}: {best_result['error']:.4f}")
        logging.info(f"Found {len(results)} unique model configurations for {item_data.name}")

    return results

def run_simplified_forecast(file_path: str, sheet_name: Optional[str] = 0, skiprows: int = 0, date_col: str = 'Date', value_col: str = 'Value', item_col: Optional[str] = 'Forecast Item', output_path: str = "simplified_best_models_monthly.xlsx", model_params: Dict = DEFAULT_PARAMS, fill_missing_months_flag: bool = True, skip_leading_zeros: bool = False, return_best_forecasts: bool = False, error_metric: str = DEFAULT_ERROR_METRIC) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Loads data, runs model selection for each item, and saves the best models.
    If fill_missing_months_flag is True, missing months are filled with zeros. If False, missing months are omitted.
    """
    all_data = load_monthly_data(file_path, sheet_name, skiprows, date_col, value_col, item_col, fill_missing_months_flag=fill_missing_months_flag, skip_leading_zeros=skip_leading_zeros)
    print(f"All data shape: {all_data.shape}")
    best_models_accumulator = [] # Changed name to avoid conflict with variable in loop
    best_forecasts_dict = {} # Changed name

    effective_item_col = item_col if item_col and item_col in all_data.columns else 'Single_Item'
    print(f"Effective item column: {effective_item_col}")

    for item_name, item_data_group in all_data.groupby(effective_item_col):
        logging.info(f"--- Processing Item: {item_name} ---")
        # Pass only the 'Value' series to find_best_model
        model_results_for_item = find_best_model(item_data_group['Value'], model_params, return_forecasts=return_best_forecasts, error_metric=error_metric)

        if model_results_for_item:
            for model_info in model_results_for_item:
                model_info_copy = model_info.copy() # Work on a copy
                model_info_copy['Forecast Item'] = item_name
                best_models_accumulator.append(model_info_copy)

            if return_best_forecasts and len(model_results_for_item) > 0:
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

    if not best_models_accumulator:
        logging.warning("No best models found for any item.")
        return pd.DataFrame(), None if return_best_forecasts else pd.DataFrame()


    summary_df = pd.DataFrame(best_models_accumulator)

    cols_order = ['Forecast Item', 'model', 'error', 'error_metric']
    param_cols = sorted([col for col in summary_df.columns if col.startswith('param_') and col not in cols_order])
    final_cols = cols_order + param_cols
    # Ensure all expected columns are present, add if missing (e.g., if some models failed for all items)
    for col in final_cols:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA


    summary_df = summary_df[final_cols] # Reorder
    summary_df = summary_df.sort_values(by=['Forecast Item', 'error']).reset_index(drop=True)

    summary_df.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in summary_df.columns]

    try:
        summary_df.to_excel(output_path, index=False)
        logging.info(f"Simplified best models summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")

    if return_best_forecasts:
        return summary_df, best_forecasts_dict
    else:
        return summary_df, None

# --- Main Execution Example ---
if __name__ == "__main__":
    try:
        # Note: Update file_path, date_col, value_col, item_col as needed for your monthly data.
        summary_df_results, forecasts_data = run_simplified_forecast(
            file_path="monthly_data_example.xlsx",  # REQUIRED: Update with your MONTHLY data file
            # skiprows=0,                             # Optional: Update if needed
            # date_col='MonthDate',                   # REQUIRED: Update with your date column name for monthly data
            # value_col='SalesVolume',                # REQUIRED: Update with your value column name
            # item_col='ProductCategory',             # Optional: Update with your item column name
            fill_missing_months_flag=True,          # Process monthly data
            skip_leading_zeros=True,
            return_best_forecasts=True,             # If you want to inspect forecast objects
            error_metric='smape'                    # Options: 'mape', 'smape', 'rmse'
        )
        if not summary_df_results.empty:
            print("\n--- Best Model Summary (Monthly) ---")
            print(summary_df_results.head())
            print("-" * 30)

            if forecasts_data:
                print(f"\n--- Forecast data for {len(forecasts_data)} items retrieved ---")
                # Example: Print forecast for the first item if available
                # first_item_name = next(iter(forecasts_data))
                # print(f"\nSample forecast for item: {first_item_name}")
                # print(forecasts_data[first_item_name]['forecast'].head())

        else:
            print("No results generated.")

    except FileNotFoundError:
        print("\nERROR: Input file not found. Please update 'file_path' in the script.")
    except KeyError as e: #pragma: no cover
        print(f"\nERROR: Column name mismatch: {e}. Please update 'date_col', 'value_col', or 'item_col'.")
    except Exception as e: #pragma: no cover
        print(f"\nAn unexpected error occurred: {e}")