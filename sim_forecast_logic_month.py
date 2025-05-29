# -*- coding: utf-8 -*-
"""
Simplified Demand Forecasting Logic

Focuses on monthly data, 70/30 train/test split, and selects the best model
based on monthly MAPE on the test set.
Includes rolling forecast accuracy for the last 3 months.
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
DEFAULT_ERROR_METRIC = 'smape'  # Options: 'mape', 'smape', 'rmse'

DEFAULT_PARAMS = {
    "ma_windows": list(range(2, 27)),
    "es_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(),
    "croston_alphas": np.round(np.arange(0.01, 1.0, 0.01), 1).tolist(),
    "mlr_seasonalities": list(range(4, 13)), # Adjusted for typical monthly (e.g. 12 for annual)
    "arima_orders": [
        (p, 0, q) for p in range(3) for q in range(3)
        ] + [(p, 1, q) for p in range(3) for q in range(3)
    ],
    "arima_trends": ['n', 'c', 'ct'],
}

# --- Helper Functions --- (Assuming these are from the previous version and correct for monthly)

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

    freq = pd.infer_freq(month_index.sort_values()) or "MS" 

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
        logging.info("Missing months filling (if any) complete.")

        if skip_leading_zeros and not df.empty:
            processed_groups = []
            if item_col and item_col in df.columns and df[item_col].nunique() > 1:
                for _, group in df.groupby(item_col):
                    processed_series = remove_leading_zeros(group['Value'])
                    processed_groups.append(group.loc[processed_series.index])
                if processed_groups:
                    df = pd.concat(processed_groups)
                else: # Handle case where all groups become empty
                    df = pd.DataFrame(columns=df.columns).set_index(df.index.name) if isinstance(df.index, pd.DatetimeIndex) else pd.DataFrame(columns=df.columns)


            elif not df.empty : # Single item or item_col not used effectively
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

def calculate_error_metric(actual: pd.Series, forecast: pd.Series, metric: Literal['mape', 'smape', 'rmse'] = 'smape') -> float:
    """Calculates specified error metric ensuring alignment and handling zeros."""
    actual = actual.copy()
    forecast = forecast.copy()

    forecast = forecast.reindex(actual.index)

    mask = actual.notna() & forecast.notna()
    actual = actual[mask]
    forecast = forecast[mask]

    if len(actual) == 0:
        #logging.warning("No valid points for error calculation after alignment.")
        return np.inf

    if metric == 'mape':
        mask_zero = actual != 0
        if not mask_zero.all():
            #logging.warning(f"Zeros found in actual values during MAPE calculation. These points ({sum(~mask_zero)}) will be excluded.")
            actual_non_zero = actual[mask_zero]
            forecast_non_zero = forecast[mask_zero]

            if len(actual_non_zero) == 0:
                #logging.warning("No valid points for MAPE calculation after zero handling.")
                return np.inf
            actual, forecast = actual_non_zero, forecast_non_zero

        abs_percentage_errors = np.abs((actual - forecast) / actual) * 100
        return abs_percentage_errors.mean()

    elif metric == 'smape':
        numerator = np.abs(forecast - actual)
        denominator = np.abs(forecast) + np.abs(actual)
        
        # For SMAPE, if a actual and forecast are both 0 for a period, SMAPE for that period is 0.
        # If one is 0 and other is not, SMAPE is 200.
        # The formula handles this naturally if we don't exclude zero denominators where num is also zero.
        # However, to avoid NaN from 0/0, we can filter where denominator is 0.
        # If num and den are both 0, the point contributes 0 to sum of smape_values if not included.
        
        smape_values = np.zeros_like(numerator)
        non_zero_den_mask = denominator != 0
        
        smape_values[non_zero_den_mask] = 200 * numerator[non_zero_den_mask] / denominator[non_zero_den_mask]
        # For cases where actual=0 and forecast=0, num=0, den=0. Result should be 0.
        # The above handles it. If num!=0 and den=0 (not possible as den = |F|+|A|), then it's an issue.
        # The only way den=0 is if F=0 and A=0.

        return smape_values.mean()


    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(actual, forecast))

    else: # Should not happen with Literal
        #logging.warning(f"Unknown error metric: {metric}. Using SMAPE instead.")
        return calculate_error_metric(actual, forecast, 'smape')

# --- Model Implementations (Simplified - Unchanged as per request) ---
def forecast_moving_average(train_data: pd.Series, test_len: int, window: int) -> Optional[pd.Series]:
    if len(train_data) < window:
        # logging.warning(f"MA: Not enough training data ({len(train_data)}) for window {window}")
        return None
    last_train_avg = train_data.iloc[-window:].mean()
    freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
    try:
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
    except IndexError: # train_data is empty
        return None
    forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
    forecast = pd.Series([last_train_avg] * test_len, index=forecast_index)
    return forecast

def forecast_exponential_smoothing(train_data: pd.Series, test_len: int, alpha: float) -> Optional[pd.Series]:
    if train_data.empty: return None
    try:
        model = sm.tsa.SimpleExpSmoothing(train_data, initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
        forecast_values = model.forecast(steps=test_len)
        freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
        forecast = pd.Series(forecast_values, index=forecast_index)
        return forecast
    except Exception: # as e:
        # logging.error(f"ES Error (alpha={alpha}): {e}")
        return None

def forecast_linear_regression(train_data: pd.Series, test_len: int) -> Optional[pd.Series]:
    if len(train_data) < 2: return None # Need at least 2 points for regression
    try:
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values
        model = LinearRegression()
        model.fit(X_train, y_train)
        X_test = np.arange(len(train_data), len(train_data) + test_len).reshape(-1, 1)
        forecast_values = model.predict(X_test)
        freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)
        forecast = pd.Series(forecast_values, index=forecast_index)
        return forecast
    except Exception: # as e:
        # logging.error(f"LR Error: {e}")
        return None

def forecast_multiple_linear_regression(train_data: pd.Series, test_len: int, seasonality: int) -> Optional[pd.Series]:
    if len(train_data) < seasonality or len(train_data) < 2 : # Basic check
        # logging.warning(f"MLR: Not enough data for seasonality {seasonality} or basic regression.")
        return None
    try:
        current_freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
        if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
            train_data = train_data.asfreq(current_freq)
            if train_data.index.freq is None: # Still none
                # logging.error("MLR: Could not establish frequency for train_data.")
                return None
        
        time_trend_train = np.arange(len(train_data))
        month_in_cycle_train = time_trend_train % seasonality
        X_train_df = pd.DataFrame({'time_trend': time_trend_train})
        seasonal_dummies_train = pd.get_dummies(month_in_cycle_train, prefix=f'cycle_month', drop_first=False)
        for i in range(seasonality):
            col_name = f'cycle_month_{i}'
            if col_name not in seasonal_dummies_train.columns:
                seasonal_dummies_train[col_name] = 0
        seasonal_dummies_train = seasonal_dummies_train.reindex(columns=[f'cycle_month_{i}' for i in range(seasonality)], fill_value=0)
        X_train = pd.concat([X_train_df, seasonal_dummies_train], axis=1)
        y_train = train_data.values

        model = LinearRegression()
        model.fit(X_train, y_train)

        last_date_train = train_data.index.max()
        forecast_index = pd.date_range(start=last_date_train + pd.DateOffset(months=1), periods=test_len, freq=train_data.index.freq)
        time_trend_test = np.arange(len(train_data), len(train_data) + test_len)
        month_in_cycle_test = time_trend_test % seasonality
        X_test_df = pd.DataFrame({'time_trend': time_trend_test})
        seasonal_dummies_test = pd.get_dummies(month_in_cycle_test, prefix=f'cycle_month', drop_first=False)
        for i in range(seasonality):
            col_name = f'cycle_month_{i}'
            if col_name not in seasonal_dummies_test.columns:
                seasonal_dummies_test[col_name] = 0
        seasonal_dummies_test = seasonal_dummies_test.reindex(columns=[f'cycle_month_{i}' for i in range(seasonality)], fill_value=0)
        X_test = pd.concat([X_test_df, seasonal_dummies_test], axis=1)
        
        preds = model.predict(X_test)
        preds = np.maximum(0, preds) # Ensure non-negativity
        return pd.Series(preds, index=forecast_index, name='forecast')
    except Exception: # as e:
        # logging.error(f"MLR Error (seasonality={seasonality}): {e}")
        return None

def forecast_croston(train_data: pd.Series, test_len: int, alpha: float = 0.1) -> Optional[pd.Series]:
    if len(train_data) < 2: # logging.warning(f"Croston: Not enough data ({len(train_data)})"); 
        return None
    
    demand = train_data.values; n = len(demand)
    Z = np.zeros(n); X = np.zeros(n) # fitted = np.zeros(n) # Not used for forecast
    first_idx = next((i for i, v in enumerate(demand) if v != 0), None)

    freq = train_data.index.freq or pd.infer_freq(train_data.index) or 'MS'
    try:
        forecast_start_date = train_data.index[-1] + pd.DateOffset(months=1)
    except IndexError: return None # Empty train_data
    idx = pd.date_range(start=forecast_start_date, periods=test_len, freq=freq)

    if first_idx is None: # All zeros
        # logging.warning("Croston: No non-zero demands in training data")
        return pd.Series([0] * test_len, index=idx)

    Z[first_idx] = demand[first_idx]; X[first_idx] = 1.0; q = 1.0
    for i in range(first_idx + 1, n):
        if demand[i] != 0:
            Z[i] = alpha * demand[i] + (1 - alpha) * Z[i-1]
            X[i] = alpha * q + (1 - alpha) * X[i-1]
            q = 1.0
        else:
            Z[i] = Z[i-1]; X[i] = X[i-1]; q += 1.0
    forecast_value = Z[n-1] / X[n-1] if X[n-1] > 0 else 0
    return pd.Series([forecast_value] * test_len, index=idx)

def forecast_arima(train_data: pd.Series, test_len: int, arima_order: Tuple[int, int, int], trend_param: str = "n") -> Optional[pd.Series]:
    p, d, q = arima_order
    # Adjusted trend logic for ARIMA based on statsmodels requirements
    if d == 0 and trend_param not in ['n', 'c']: trend_param = 'c' # Default to const if no diff and trend is t/ct
    if d > 0 and trend_param == 'c': trend_param = 'n' # Constant is implicit or handled by 'ct' if d>0
    
    min_data_needed = max(sum(arima_order) + d + 1, p + q + d + 1, 7) # Heuristic minimums
    if len(train_data) < min_data_needed :
        # logging.warning(f"ARIMA: Not enough training data ({len(train_data)}) for order={arima_order}, trend={trend_param}. Need at least {min_data_needed}")
        return None
    if test_len <= 0: return None

    try:
        current_freq = train_data.index.freq or pd.infer_freq(train_data.index)
        train_data_for_arima = train_data.copy()
        if current_freq is None:
            # logging.warning("ARIMA: Frequency not found for train_data. Attempting to set to 'MS'.")
            train_data_for_arima = train_data_for_arima.asfreq('MS')
        elif not isinstance(train_data_for_arima.index, pd.DatetimeIndex) or train_data_for_arima.index.freq is None:
             train_data_for_arima = train_data_for_arima.asfreq(current_freq)
        
        if train_data_for_arima.isnull().any(): # Check if asfreq introduced NaNs that ARIMA can't handle
            # logging.warning("ARIMA: NaNs in series after frequency setting. Trying to fill with previous value.")
            train_data_for_arima = train_data_for_arima.fillna(method='ffill').fillna(method='bfill') # Fill any gaps
        if train_data_for_arima.isnull().any(): # Still NaNs
            # logging.error("ARIMA: Could not resolve NaNs in series.")
            return None


        model = ARIMA(train_data_for_arima, order=arima_order, trend=trend_param, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        
        raw_fc = model_fit.forecast(steps=test_len)
        
        forecast_start_date = train_data_for_arima.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start_date, periods=test_len, freq=train_data_for_arima.index.freq or 'MS')
        forecast = pd.Series(raw_fc, index=forecast_index)
        return forecast
    except Exception: # as e:
        # logging.error(f"ARIMA Error(order={arima_order}, trend={trend_param}): {e}")
        return None

# --- Orchestration ---

def find_best_model(item_data: pd.Series, params: Dict = DEFAULT_PARAMS, error_metric: str = DEFAULT_ERROR_METRIC) -> Optional[List[Dict[str, Any]]]:
    """
    Evaluates multiple models on an internal train/test split of item_data.
    Returns all models sorted by the specified error metric on the internal test set.
    This function DOES NOT perform the final 1-step forecast for accuracy calculation.
    """
    # Reduced minimum length for find_best_model internal split logic
    min_len_for_split = 7 # e.g. 70% of 7 is ~5 (train), ~2 (test). Smallest MA window is 2.
    if item_data.empty or len(item_data) < min_len_for_split:
        # logging.warning(f"find_best_model: Skipping item {item_data.name}: Not enough data ({len(item_data)}) for internal 70/30 split. Need {min_len_for_split}.")
        return None

    # Ensure frequency for internal operations
    if item_data.index.freq is None:
        inferred_freq = pd.infer_freq(item_data.index) or 'MS'
        item_data = item_data.asfreq(inferred_freq)
        if item_data.isnull().all(): # Check if asfreq made all NaNs
            # logging.error(f"find_best_model: Failed to process {item_data.name} after assuming '{inferred_freq}' frequency.")
            return None
            
    # Internal train/test split for model parameter selection
    # Ensure test_len is at least 1 for error calculation
    train_len = int(len(item_data) * 0.7)
    if train_len < 2 : train_len = len(item_data) -1 # ensure at least 1 for test if data is tiny
    if train_len <=0 : return None # not enough to even make a single test point

    test_len_internal = len(item_data) - train_len
    if test_len_internal == 0: # if data is extremely short, e.g. 3 points, 70% is 2, test is 1
        if len(item_data) > 1 : # if we have at least 2 points
             train_len = len(item_data) -1
             test_len_internal = 1
        else: # Cannot split if only 1 point
            # logging.warning(f"find_best_model: Not enough data for item {item_data.name} to create internal test set.")
            return None


    train_data_internal = item_data.iloc[:train_len]
    test_data_internal = item_data.iloc[train_len:]
    
    if train_data_internal.empty or test_data_internal.empty:
        # logging.warning(f"find_best_model: Empty train or test set for {item_data.name} after internal split.")
        return None

    unique_results = {}

    # Model evaluations (using train_data_internal, test_data_internal, test_len_internal)
    # 1. Moving Average
    for w in params.get("ma_windows", []):
        forecast = forecast_moving_average(train_data_internal, test_len_internal, w)
        if forecast is not None:
            error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
            unique_results[f"Moving Average_window_{w}"] = {"model": "Moving Average", "param_window": w, "error": error_value, "error_metric": error_metric}
    
    # 2. Exponential Smoothing
    for alpha in params.get("es_alphas", []):
        forecast = forecast_exponential_smoothing(train_data_internal, test_len_internal, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
            unique_results[f"ES_alpha_{alpha}"]= {"model": "Exponential Smoothing", "param_alpha": alpha, "error": error_value, "error_metric": error_metric}

    # 3. Linear Regression
    forecast = forecast_linear_regression(train_data_internal, test_len_internal)
    if forecast is not None:
        error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
        unique_results["LR"] = {"model": "Linear Regression", "param": "N/A", "error": error_value, "error_metric": error_metric}

    # 4. MLR
    for seas in params.get("mlr_seasonalities", []):
        if seas >= len(train_data_internal) / 2 and seas > 1 : continue # Heuristic
        forecast = forecast_multiple_linear_regression(train_data_internal, test_len_internal, seas)
        if forecast is not None:
            error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
            unique_results[f"MLR_seas_{seas}"] = {"model": "MLR", "param_seasonality": seas, "error": error_value, "error_metric": error_metric}
            
    # 5. Croston
    for alpha in params.get("croston_alphas", []):
        forecast = forecast_croston(train_data_internal, test_len_internal, alpha)
        if forecast is not None:
            error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
            unique_results[f"Croston_alpha_{alpha}"] = {"model": "Croston", "param_alpha": alpha, "error": error_value, "error_metric": error_metric}

    # 6. ARIMA
    for arima_order in params.get("arima_orders", []):
        for trend in params.get("arima_trends", ['n']):
            forecast = forecast_arima(train_data_internal, test_len_internal, arima_order, trend_param=trend)
            if forecast is not None:
                error_value = calculate_error_metric(test_data_internal, forecast, error_metric)
                unique_results[f"ARIMA_{arima_order}_t_{trend}"] = {"model": "ARIMA", "param_order": str(arima_order), "param_trend": trend, "error": error_value, "error_metric": error_metric}
    
    if not unique_results:
        # logging.warning(f"find_best_model: No successful models found for item {item_data.name} during internal evaluation.")
        return None

    results = list(unique_results.values())
    results.sort(key=lambda x: x['error'])
    
    # logging.info(f"find_best_model: Best internal model for {item_data.name}: {results[0]['model']} with params { {k: v for k, v in results[0].items() if k.startswith('param_')} } -> {results[0]['error_metric'].upper()}: {results[0]['error']:.4f}")
    return results


def run_simplified_forecast(
    file_path: str, 
    sheet_name: Optional[str] = 0, 
    skiprows: int = 0, 
    date_col: str = 'Date', 
    value_col: str = 'Value', 
    item_col: Optional[str] = 'Forecast Item', 
    output_path: str = "rolling_forecast_accuracy.xlsx", 
    model_params: Dict = DEFAULT_PARAMS, 
    fill_missing_months_flag: bool = True, 
    skip_leading_zeros: bool = False, 
    error_metric: str = DEFAULT_ERROR_METRIC
) -> pd.DataFrame:
    """
    Loads data, performs rolling forecast accuracy evaluation for the last 3 months for each item.
    Saves a summary of these accuracies.
    """
    all_data = load_monthly_data(
        file_path, sheet_name, skiprows, date_col, value_col, item_col, 
        fill_missing_months_flag=fill_missing_months_flag, 
        skip_leading_zeros=skip_leading_zeros
    )
    
    if all_data.empty:
        logging.warning("Loaded data is empty. No forecast evaluation will be performed.")
        return pd.DataFrame()

    overall_accuracy_results = []
    effective_item_col = item_col if item_col and item_col in all_data.columns else 'Single_Item'

    for item_name, item_data_group in all_data.groupby(effective_item_col):
        logging.info(f"--- Processing Item: {item_name} ---")
        original_item_series = item_data_group['Value'].copy()

        # Minimum length: 7 for find_best_model + 3 for rolling = 10
        min_total_len = 10 
        if len(original_item_series) < min_total_len:
            logging.warning(f"Skipping item {item_name}: Not enough data ({len(original_item_series)}) for 3-month rolling forecast evaluation. Need at least {min_total_len}.")
            continue

        # Loop for the last 3 months (0: last month, 1: month before last, 2: two months before last)
        for k_month_offset in range(3): # k_month_offset = 0, 1, 2
            
            # Determine the actual value and its date
            try:
                actual_value = original_item_series.iloc[-(k_month_offset + 1)]
                actual_date = original_item_series.index[-(k_month_offset + 1)]
            except IndexError:
                logging.warning(f"Item {item_name}: Not enough data to evaluate {k_month_offset + 1} months ago from end.")
                break # Stop for this item if not enough historical points

            # Determine training data for this iteration
            # series_for_training ends right before the 'actual_date'
            end_index_for_training = -(k_month_offset + 1)
            if end_index_for_training == -1: # means actual is the very last point
                 series_for_training = original_item_series.iloc[:-1]
            else: # actual is further back
                 series_for_training = original_item_series.iloc[:end_index_for_training]


            min_training_len_for_find_best = 7 # As used in find_best_model
            if len(series_for_training) < min_training_len_for_find_best:
                logging.warning(f"Item {item_name}, Target {actual_date}: Not enough training data ({len(series_for_training)}) for find_best_model. Skipping this period.")
                continue
            
            logging.info(f"Item {item_name}: Evaluating for target date {actual_date} (Training data length: {len(series_for_training)}).")

            # Find best model type and parameters using data up to `series_for_training`
            # `find_best_model` uses its own internal 70/30 split on `series_for_training`
            model_selection_results = find_best_model(
                series_for_training.copy(), # Pass a copy to avoid modifications
                params=model_params, 
                error_metric=error_metric
            )

            if not model_selection_results:
                logging.warning(f"Item {item_name}, Target {actual_date}: No best model identified by find_best_model.")
                overall_accuracy_results.append({
                    "Forecast Item": item_name, "Target Forecast Date": actual_date, "Months Ago": k_month_offset + 1,
                    "Best Model": "N/A", "Best Model Params": {}, "Forecast Value": np.nan, "Actual Value": actual_value,
                    "Forecast Accuracy (1 - |F-A|/|A|)": np.nan, "Internal Model Selection Error Metric": error_metric, "Internal Model Selection Error": np.nan
                })
                continue
            
            best_model_details_for_period = model_selection_results[0]
            
            # Retrain the identified best model on the *entire* `series_for_training` and forecast 1 step
            forecast_value_for_actual_date = np.nan # Initialize
            model_name = best_model_details_for_period['model']
            
            # Helper to get forecast_value_for_actual_date
            # This re-trains on `series_for_training` and forecasts 1 step
            forecast_series_final = None
            if model_name == "Moving Average":
                window = best_model_details_for_period.get('param_window')
                if window is not None: forecast_series_final = forecast_moving_average(series_for_training, 1, window)
            elif model_name == "Exponential Smoothing":
                alpha = best_model_details_for_period.get('param_alpha')
                if alpha is not None: forecast_series_final = forecast_exponential_smoothing(series_for_training, 1, alpha)
            elif model_name == "Linear Regression":
                forecast_series_final = forecast_linear_regression(series_for_training, 1)
            elif model_name == "MLR":
                seasonality = best_model_details_for_period.get('param_seasonality')
                if seasonality is not None: forecast_series_final = forecast_multiple_linear_regression(series_for_training, 1, seasonality)
            elif model_name == "Croston":
                alpha = best_model_details_for_period.get('param_alpha')
                if alpha is not None: forecast_series_final = forecast_croston(series_for_training, 1, alpha)
            elif model_name == "ARIMA":
                order_str = best_model_details_for_period.get('param_order')
                trend = best_model_details_for_period.get('param_trend')
                if order_str and trend:
                    try: order = eval(order_str)
                    except: order = (0,0,0); logging.error("ARIMA order eval failed")
                    forecast_series_final = forecast_arima(series_for_training, 1, order, trend)
            
            if forecast_series_final is not None and not forecast_series_final.empty:
                forecast_value_for_actual_date = forecast_series_final.iloc[0]

            # Calculate accuracy: 1 - (abs(F-A)/|A|)
            accuracy = np.nan
            if not np.isnan(forecast_value_for_actual_date): # Check if forecast was made
                if actual_value == 0:
                    if forecast_value_for_actual_date == 0:
                        accuracy = 1.0
                    else: # Actual is 0, Forecast is non-zero -> Undefined or 0% accurate by some conventions
                        accuracy = np.nan 
                        logging.debug(f"Item {item_name}, Target {actual_date}: Actual is 0, Forecast is non-zero. Accuracy is NaN.")
                else: # Actual value is not 0
                    accuracy = 1.0 - (abs(forecast_value_for_actual_date - actual_value) / abs(actual_value))
            
            logging.info(f"  Item {item_name}, Target {actual_date}: Model={model_name}, F={forecast_value_for_actual_date:.2f}, A={actual_value:.2f}, Acc={accuracy:.4f}")

            overall_accuracy_results.append({
                "Forecast Item": item_name,
                "Target Forecast Date": actual_date,
                "Months Ago": k_month_offset + 1, # 1=last month, 2=month before last, etc.
                "Best Model": model_name,
                "Best Model Params": {k.replace('param_', ''): v for k, v in best_model_details_for_period.items() if k.startswith('param_')},
                "Forecast Value": forecast_value_for_actual_date,
                "Actual Value": actual_value,
                "Forecast Accuracy (1 - |F-A|/|A|)": accuracy,
                "Internal Model Selection Error Metric": best_model_details_for_period['error_metric'],
                "Internal Model Selection Error": best_model_details_for_period['error']
            })

    if not overall_accuracy_results:
        logging.warning("No accuracy results were generated for any item.")
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame(overall_accuracy_results)
        # Reorder columns for clarity
        cols_order = [
            "Forecast Item", "Target Forecast Date", "Months Ago", "Best Model", 
            "Forecast Value", "Actual Value", "Forecast Accuracy (1 - |F-A|/|A|)",
            "Best Model Params", "Internal Model Selection Error Metric", "Internal Model Selection Error"
        ]
        # Add any missing columns (e.g., if all params were None)
        for col in cols_order:
            if col not in summary_df.columns:
                 summary_df[col] = np.nan if col != "Best Model Params" else pd.Series([{} for _ in range(len(summary_df))])

        summary_df = summary_df[cols_order]


    try:
        summary_df.to_excel(output_path, index=False)
        logging.info(f"Rolling forecast accuracy summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")

    return summary_df