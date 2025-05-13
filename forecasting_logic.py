# -*- coding: utf-8 -*-
"""
Refactored Demand Forecasting Model Comparison Tool

This script evaluates different time-series forecasting models (Moving Average,
Exponential Smoothing, Linear Regression, Multiple Linear Regression) to find
the best fit for demand forecasting based on historical weekly data. It uses
a rolling forecast validation approach and compares monthly aggregated forecasts
against actual monthly shipments using MAPE.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional

# --- Configuration ---

CONFIG = {
    # File Paths
    "raw_data_path": "Outliers.xlsx",  # Historical weekly data
    "actuals_path": "HDA Total Summary.xlsx", # Actual monthly shipments
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

# --- Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore specific pandas warnings if needed
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

# --- Helper Functions: Date Manipulation ---

def get_first_monday(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the first Monday of the month for the given datetime."""
    first_day = dt.replace(day=1)
    days_to_monday = (7 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_to_monday)

def get_last_monday(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the last Monday of the month for the given datetime."""
    # Go to the first day of the next month, then subtract days to get to the last day of the original month
    next_month_first_day = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day_of_month = next_month_first_day - timedelta(days=1)
    # Find the last Monday on or before the last day of the month
    days_from_monday = last_day_of_month.weekday() # Monday is 0, Sunday is 6
    last_monday = last_day_of_month - timedelta(days=days_from_monday)
    return last_monday

# --- Helper Functions: Data Loading and Preparation ---

def load_and_prepare_history(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads and preprocesses historical weekly demand data."""
    try:
        df = pd.read_excel(config["raw_data_path"], skiprows=config["raw_data_skiprows"])
        logging.info(f"Loaded historical data from {config['raw_data_path']}")
    except FileNotFoundError:
        logging.error(f"Error: Historical data file not found at {config['raw_data_path']}")
        raise
    except Exception as e:
        logging.error(f"Error loading historical data: {e}")
        raise

    df = df[["Forecast Item", "Date", config["history_col_name"]]]
    df = df.rename(columns={config["history_col_name"]: "history"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["history"] = df["history"].apply(lambda x: max(x, 0)).astype(float) # Ensure non-negative and float
    df = df.sort_values(by=["Forecast Item", "Date"])
    
    df.set_index('Date', inplace=True)
    return df

def load_and_prepare_actuals(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads and prepares actual monthly shipment data."""
    try:
        df = pd.read_excel(config["actuals_path"], skiprows=config["actuals_skiprows"]).T
        logging.info(f"Loaded actuals data from {config['actuals_path']}")
    except FileNotFoundError:
        logging.error(f"Error: Actuals data file not found at {config['actuals_path']}")
        raise
    except Exception as e:
        logging.error(f"Error loading actuals data: {e}")
        raise

    forecast_item_row = df.iloc[0]
    df = df.loc[:, forecast_item_row.notna()]
    df.columns = df.loc["Forecast Item"].tolist()
    df = df.iloc[5:] # Assuming header rows are fixed
    df.index = pd.to_datetime(df.index, format='%b %Y')
    df.index = df.index.to_period('M') # Use PeriodIndex for easy monthly matching
    df = df.astype(float) # Ensure numeric types
    logging.info("Prepared actual monthly shipment data.")
    return df

def generate_experiment_splits(weekly_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """Generates train/test split date ranges for rolling forecast validation."""
    lag = config["forecast_lag"]
    total_experiments = config["num_experiments"]
    min_weeks = config["min_weeks_in_month"]

    # Calculate valid months based on week counts for the *entire* dataset history
    # This assumes all items share a similar relevant date range for splitting logic
    temp_data = weekly_data.reset_index()
    temp_data['Month_Year'] = temp_data['Date'].dt.to_period('M')
    weeks_per_month = temp_data.groupby('Month_Year').size()
    valid_months = weeks_per_month[weeks_per_month >= min_weeks]

    if len(valid_months) < lag + total_experiments:
        raise ValueError(f"Not enough valid months ({len(valid_months)}) in history "
                         f"to perform {total_experiments} experiments with lag {lag}.")

    experiment_dates = {}
    num_months_to_offset = lag + total_experiments - 1

    for i in range(total_experiments):
        # Define the training months for this experiment
        train_months_period = valid_months.iloc[i:-(num_months_to_offset - i) or None]

        if train_months_period.empty:
            logging.warning(f"Skipping experiment {i+1} due to insufficient remaining months.")
            continue

        train_start_month = train_months_period.index[0]
        train_end_month = train_months_period.index[-1]

        # Convert month periods to actual start/end dates (using Mondays)
        train_start_date = get_first_monday(train_start_month.start_time)
        # Train end date is the start of the *last* Monday of the end month.
        # The slice later will be [start, end)
        train_end_date = get_last_monday(train_end_month.start_time) + timedelta(days=7) # End date is exclusive

        # Determine test period start/end month
        test_start_month = train_end_month + lag
        test_end_month = test_start_month # Assuming lag defines the single month to test

        test_start_date = get_first_monday(test_start_month.start_time)
        test_end_date = get_last_monday(test_end_month.start_time)

        experiment_key = f"Experiment_{i+1}"
        experiment_dates[experiment_key] = {
            "train_range": [train_start_date, train_end_date],
            "test_range": [test_start_date, test_end_date]
        }
        logging.debug(f"{experiment_key}: Train [{train_start_date.date()} - {train_end_date.date()}), "
                      f"Test [{test_start_date.date()} - {test_end_date.date()}]")

    if not experiment_dates:
         raise ValueError("Could not generate any valid experiment splits.")

    logging.info(f"Generated {len(experiment_dates)} experiment splits.")
    return experiment_dates

# --- Helper Functions: Forecasting & Evaluation ---

def adjust_forecast_for_partial_weeks(
    weekly_forecast_df: pd.DataFrame,
    test_start_date: pd.Timestamp,
    test_end_date: pd.Timestamp,
    forecast_col: str
) -> pd.DataFrame:
    """
    Adjusts the forecast amount for the first and last weeks of the test month
    to account for partial week coverage. Operates **in place**.
    """
    df = weekly_forecast_df # Operate directly, maybe return copy if preferred

    if df.empty:
        return df

    # Adjust first week
    first_forecast_date = df.index.min()
    days_in_first_week = 7 - (first_forecast_date - test_start_date).days
    if days_in_first_week < 7:
        adjustment_factor = days_in_first_week / 7.0
        df.loc[first_forecast_date, forecast_col] *= adjustment_factor
        # Original logic added proportion - this seems more correct for prorating
        # df.loc[first_forecast_date, forecast_col] += (df.loc[first_forecast_date, forecast_col] * (days_offset / 7.0))
        logging.debug(f"Adjusted first week forecast ({first_forecast_date.date()}) by factor {adjustment_factor:.2f}")


    # Adjust last week
    last_forecast_date = df.index.max()
    end_of_test_month = test_end_date.to_period('M').end_time
    days_in_last_week = (end_of_test_month - last_forecast_date).days + 1 # +1 because inclusive
    if days_in_last_week < 7:
        adjustment_factor = days_in_last_week / 7.0
        df.loc[last_forecast_date, forecast_col] *= adjustment_factor
        logging.debug(f"Adjusted last week forecast ({last_forecast_date.date()}) by factor {adjustment_factor:.2f}")

    return df


def calculate_monthly_mape(
    item_name: str,
    test_data_weekly: pd.DataFrame, # Should have DateTimeIndex and forecast_col
    monthly_actuals: pd.DataFrame, # Should have PeriodIndex ('M') and column matching item_name
    forecast_col: str,
    model_params: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Aggregates weekly forecasts to monthly, calculates MAPE against actuals,
    and returns a DataFrame with results for the month.
    """
    if test_data_weekly.empty:
        logging.warning(f"[{item_name}] Test data is empty for MAPE calculation with params {model_params}.")
        return None

    monthly_forecast = test_data_weekly[[forecast_col]].resample('M').sum()
    monthly_forecast.index = monthly_forecast.index.to_period('M')

    # Prepare actuals for the specific item and period
    test_period = monthly_forecast.index
    try:
        actuals_subset = monthly_actuals.loc[test_period, [item_name]]
        actuals_subset = actuals_subset.rename(columns={item_name: "actual_shipment"})
    except KeyError:
         logging.error(f"[{item_name}] Actual shipment data not found for item or period {test_period}.")
         return None
    except Exception as e:
        logging.error(f"[{item_name}] Error accessing actuals: {e}")
        return None

    # Merge forecast and actuals
    monthly_results = pd.merge(monthly_forecast, actuals_subset, left_index=True, right_index=True, how="left")

    if monthly_results.empty or monthly_results['actual_shipment'].isnull().all():
        logging.warning(f"[{item_name}] No matching actuals found for period {test_period} with params {model_params}.")
        return None

    # Calculate MAPE, handling zero actuals
    actual = monthly_results["actual_shipment"]
    forecast = monthly_results[forecast_col]

    # Ensure non-negative forecasts after aggregation/adjustment
    forecast = forecast.clip(lower=0)
    monthly_results[forecast_col] = forecast

    # Calculate Absolute Error
    monthly_results['abs_error'] = abs(forecast - actual)

    # Calculate MAPE where actual is not zero
    mask = actual != 0
    monthly_results['mape'] = np.nan
    monthly_results.loc[mask, 'mape'] = abs(forecast[mask] - actual[mask]) / actual[mask]
    monthly_results.loc[~mask, 'mape'] = 0.0 # Assign 0 MAPE where actual is 0 (common practice)

    # Add metadata
    monthly_results["Forecast Item"] = item_name
    for key, value in model_params.items():
        monthly_results[f"param_{key}"] = value

    return monthly_results

# --- Model Implementation Functions ---

def forecast_moving_average(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp],
    monthly_actuals: pd.DataFrame,
    item_name: str,
    window: int
) -> Optional[pd.DataFrame]:
    """Fits Moving Average, forecasts, adjusts, evaluates MAPE."""
    model_name = "Moving Average"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"
    params = {"window": window}

    if train_data.empty or len(train_data) < window:
        logging.warning(f"[{item_name} - {model_name}] Insufficient training data ({len(train_data)}) for window {window}.")
        return None

    try:
        rolling_mean = train_data['history'].rolling(window=window).mean()
        last_known_mean = rolling_mean.iloc[-1]
        if pd.isna(last_known_mean):
             logging.warning(f"[{item_name} - {model_name}] Last rolling mean is NaN for window {window}.")
             return None

        # Create forecast DataFrame for the test period
        test_start_date, test_end_date = test_range
        # Generate weekly dates (Mondays) within the test range
        test_dates = pd.date_range(start=test_start_date, end=test_end_date, freq='W-MON')
        if test_dates.empty:
            logging.warning(f"[{item_name} - {model_name}] No valid Mondays found in test range for window {window}.")
            return None

        test_forecast_df = pd.DataFrame(index=test_dates)
        test_forecast_df[forecast_col] = last_known_mean
        test_forecast_df[forecast_col] = test_forecast_df[forecast_col].clip(lower=0) # Ensure non-negative

        # Adjust for partial weeks
        test_forecast_df = adjust_forecast_for_partial_weeks(test_forecast_df, test_start_date, test_end_date, forecast_col)

        # Calculate MAPE
        monthly_results = calculate_monthly_mape(item_name, test_forecast_df, monthly_actuals, forecast_col, params)
        return monthly_results

    except Exception as e:
        logging.error(f"[{item_name} - {model_name}] Error during forecasting with window {window}: {e}", exc_info=True)
        return None


def forecast_exponential_smoothing(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp],
    monthly_actuals: pd.DataFrame,
    item_name: str,
    alpha: float,
    config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Fits Simple Exponential Smoothing, forecasts, adjusts, evaluates MAPE."""
    model_name = "Exp Smoothing"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"
    params = {"alpha": alpha}

    if train_data.empty:
        logging.warning(f"[{item_name} - {model_name}] Training data empty for alpha {alpha}.")
        return None

    try:
        # Ensure weekly frequency for statsmodels
        train_data_weekly = train_data['history'].asfreq('W-MON')
        if train_data_weekly.isnull().any():
            # Simple forward fill for internal gaps if any - consider imputation if needed
            train_data_weekly = train_data_weekly.ffill()
        if train_data_weekly.isnull().any(): # Still NaN after ffill (e.g., at start)
            logging.warning(f"[{item_name} - {model_name}] NaNs in training data after freq conversion for alpha {alpha}.")
            return None # Model fitting might fail

        model = sm.tsa.SimpleExpSmoothing(train_data_weekly, initialization_method='estimated')
        fit = model.fit(smoothing_level=alpha, optimized=False) # Use specified alpha

        # Forecast necessary steps
        test_start_date, test_end_date = test_range
        last_train_date = train_data_weekly.index.max()
        # Calculate steps needed from last train date to test end date
        # Add extra buffer (e.g. a few weeks) just in case of date alignment issues
        weeks_needed = (test_end_date - last_train_date).days // 7 + 5
        weeks_needed = max(1, weeks_needed) # Forecast at least 1 step

        forecast_values = fit.forecast(steps=weeks_needed)
        forecast_values = forecast_values.clip(lower=0) # Ensure non-negative

        # Create forecast index starting right after training data
        forecast_index = pd.date_range(start=last_train_date + timedelta(days=7), periods=weeks_needed, freq='W-MON')
        forecast_series = pd.Series(forecast_values, index=forecast_index)

        # Select the forecast within the actual test range
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]

        if test_forecast_series.empty:
            logging.warning(f"[{item_name} - {model_name}] Forecast did not cover test range for alpha {alpha}.")
            return None

        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])

        # Adjust for partial weeks
        test_forecast_df = adjust_forecast_for_partial_weeks(test_forecast_df, test_start_date, test_end_date, forecast_col)

        # Calculate MAPE
        monthly_results = calculate_monthly_mape(item_name, test_forecast_df, monthly_actuals, forecast_col, params)
        return monthly_results

    except ValueError as ve:
         logging.warning(f"[{item_name} - {model_name}] Value error during model fit (alpha={alpha}): {ve}")
         return None
    except Exception as e:
        logging.error(f"[{item_name} - {model_name}] Error during forecasting with alpha {alpha}: {e}", exc_info=True)
        return None

def forecast_linear_regression(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp],
    monthly_actuals: pd.DataFrame,
    item_name: str,
    config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Fits Linear Regression (trend), forecasts, adjusts, evaluates MAPE."""
    model_name = "Linear Regression"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"
    params = {"model": "trend_only"} # No tunable params here

    if train_data.empty:
        logging.warning(f"[{item_name} - {model_name}] Training data empty.")
        return None

    try:
        # Prepare data for LR
        train_data_lr = train_data.copy()
        train_data_lr['time_index'] = np.arange(len(train_data_lr))
        X_train = train_data_lr[['time_index']]
        y_train = train_data_lr['history']

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prepare future time indices
        test_start_date, test_end_date = test_range
        last_train_date = train_data.index.max()
        first_future_index = len(train_data) # Index starts right after training data

        # Generate future dates and corresponding indices
        future_dates = pd.date_range(start=last_train_date + timedelta(days=7), end=test_end_date + timedelta(days=6), freq='W-MON') # Ensure coverage
        future_indices = np.arange(first_future_index, first_future_index + len(future_dates))
        X_future = pd.DataFrame({'time_index': future_indices})

        # Predict
        forecast_values = model.predict(X_future)
        forecast_values = np.maximum(0, forecast_values) # Ensure non-negative

        # Create forecast series and select test range
        forecast_series = pd.Series(forecast_values, index=future_dates)
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]

        if test_forecast_series.empty:
            logging.warning(f"[{item_name} - {model_name}] Forecast did not cover test range.")
            return None

        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])

        # Adjust for partial weeks
        test_forecast_df = adjust_forecast_for_partial_weeks(test_forecast_df, test_start_date, test_end_date, forecast_col)

        # Calculate MAPE
        monthly_results = calculate_monthly_mape(item_name, test_forecast_df, monthly_actuals, forecast_col, params)
        return monthly_results

    except Exception as e:
        logging.error(f"[{item_name} - {model_name}] Error during forecasting: {e}", exc_info=True)
        return None

def prepare_mlr_features(
    item_data: pd.DataFrame,
    seasonality_max: int,
    decay_factors: List[float]
) -> pd.DataFrame:
    """Pre-calculates features needed for MLR model."""
    data = item_data.reset_index().copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Base Numerical Date
    data['Numerical_Date'] = (data['Date'] - data['Date'].min()).dt.days

    # Seasonality (Week Number based on max seasonality)
    # Use modulo for week number within the cycle
    data[f'Week_Number_{seasonality_max}'] = data['Numerical_Date'] // 7 % seasonality_max

    # One-Hot Encode Weeks up to max_seasonality
    for i in range(seasonality_max):
         data[f'Week_{i}'] = (data[f'Week_Number_{seasonality_max}'] == i).astype(int)

    # Decayed Trend Features
    for decay in decay_factors:
        # Handle the 'no decay' case (decay=1 in config -> 0% weight decay)
        if abs(decay - 1.0) < 1e-6: # Treat 1.0 as no decay -> use raw Numerical_Date
             col_name = f'Decayed_Numerical_Date_Factor_1.0'
             if col_name not in data: # Avoid recreation if called multiple times
                 data[col_name] = data['Numerical_Date']
        # Handle the specific decay factor (e.g., 0.05 in config -> 95% weight decay)
        # The formula (1 - config_decay) ** numerical_date is complex. Let's replicate the original intent:
        # Original: Numerical_Date * ((1 - decay_factor) ** Numerical_Date)
        # For decay_factor = 0.05 -> weight = 0.95 ** days -> trend gets suppressed heavily over time
        # For decay_factor = 1.0 -> weight = 0.0 ** days (problematic!) -> Let's assume it meant linear trend
        elif abs(decay - 0.05) < 1e-6 :
            col_name = f'Decayed_Numerical_Date_Factor_0.05'
            if col_name not in data:
                 # Calculate weight: (1 - decay) -> 0.95
                 # Use days / 7 for weekly decay? Original used 'Numerical_Date' (daily?)
                 # Let's stick to original daily decay for replication
                 weight = (1.0 - decay) ** data['Numerical_Date']
                 data[col_name] = data['Numerical_Date'] * weight
        else:
            logging.warning(f"Unhandled decay factor {decay} in MLR feature prep.")


    data = data.set_index('Date')
    logging.debug(f"[{data['Forecast Item'].iloc[0]}] Prepared MLR features.")
    return data


def forecast_multiple_linear_regression(
    item_data_with_features: pd.DataFrame,
    train_range: List[pd.Timestamp],
    test_range: List[pd.Timestamp],
    monthly_actuals: pd.DataFrame,
    item_name: str,
    seasonality: int,
    decay_factor: float,
    config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Fits MLR with seasonality and trend decay, forecasts, adjusts, evaluates MAPE."""
    model_name = "MLR"
    forecast_col = f"forecast_{model_name.lower()}"
    params = {"seasonality": seasonality, "decay_factor": decay_factor}
    train_weeks = config["mlr_train_weeks"]

    if item_data_with_features.empty:
        logging.warning(f"[{item_name} - {model_name}] Feature data empty for params {params}.")
        return None

    try:
        # Select train and test data based on DATES from the pre-featured data
        train_start_date, train_end_date = train_range
        test_start_date, test_end_date = test_range

        full_train_data = item_data_with_features.loc[train_start_date:train_end_date - timedelta(days=1)] # Make end date exclusive

        # Use only last N weeks for training if specified
        if len(full_train_data) > train_weeks:
            train_data_mlr = full_train_data.iloc[-train_weeks:]
        else:
            train_data_mlr = full_train_data

        if train_data_mlr.empty:
             logging.warning(f"[{item_name} - {model_name}] Insufficient training data after slicing for {train_weeks} weeks. Params {params}.")
             return None

        # Test data covers potential future dates needed for prediction
        potential_test_data = item_data_with_features.loc[train_end_date:]
        if potential_test_data.empty:
            logging.warning(f"[{item_name} - {model_name}] No data available after training end date {train_end_date.date()} for params {params}.")
            return None

        # Define features based on current seasonality and decay factor
        seasonal_features = [f'Week_{i}' for i in range(seasonality)]

        # Select correct trend feature column
        if abs(decay_factor - 1.0) < 1e-6:
            trend_feature = 'Decayed_Numerical_Date_Factor_1.0' # Corresponds to Numerical_Date
        elif abs(decay_factor - 0.05) < 1e-6:
             trend_feature = 'Decayed_Numerical_Date_Factor_0.05'
        else:
             logging.error(f"[{item_name} - {model_name}] Invalid decay factor {decay_factor} specified for feature selection.")
             return None

        features = seasonal_features + [trend_feature]

        # Ensure all features exist
        if not all(f in train_data_mlr.columns for f in features):
            missing = [f for f in features if f not in train_data_mlr.columns]
            logging.error(f"[{item_name} - {model_name}] Missing required features: {missing} for params {params}")
            return None

        X_train = train_data_mlr[features]
        y_train = train_data_mlr['history']
        X_test = potential_test_data[features]

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        forecast_values = model.predict(X_test)
        forecast_values = np.maximum(0, forecast_values) # Ensure non-negative

        # Create forecast series and select actual test range
        forecast_series = pd.Series(forecast_values, index=potential_test_data.index)
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]

        if test_forecast_series.empty:
            logging.warning(f"[{item_name} - {model_name}] Forecast did not cover test range for params {params}.")
            return None

        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])

        # Adjust for partial weeks
        test_forecast_df = adjust_forecast_for_partial_weeks(test_forecast_df, test_start_date, test_end_date, forecast_col)

        # Calculate MAPE
        # Map config decay factor back to original representation if needed for output consistency
        output_decay_param = 0 if abs(decay_factor - 0.05) < 1e-6 else 1
        output_params = {"seasonality": seasonality, "trend_decay_label": output_decay_param}
        monthly_results = calculate_monthly_mape(item_name, test_forecast_df, monthly_actuals, forecast_col, output_params)
        return monthly_results

    except Exception as e:
        logging.error(f"[{item_name} - {model_name}] Error during forecasting with params {params}: {e}", exc_info=True)
        return None

# --- Orchestration ---

def run_backtest(config: Dict[str, Any]) -> pd.DataFrame:
    """Runs the backtesting process for all items and models."""
    logging.info("Starting backtesting process...")

    # Load data
    weekly_history = load_and_prepare_history(config)
    monthly_actuals = load_and_prepare_actuals(config)

    # Generate experiment splits (using combined history for date range calculation)
    experiment_splits = generate_experiment_splits(weekly_history, config)

    all_results = []
    forecast_items = weekly_history['Forecast Item'].unique()
    logging.info(f"Found {len(forecast_items)} items to process: {', '.join(forecast_items)}")

    for item_name in forecast_items:
        item_data = weekly_history[weekly_history['Forecast Item'] == item_name].copy()
        item_actuals = monthly_actuals # Already prepared

        if item_data.empty:
            logging.warning(f"Skipping item '{item_name}' due to no historical data.")
            continue

        logging.info(f"--- Processing Item: {item_name} ---")

        # Pre-calculate MLR features for this item once
        item_data_mlr_features = prepare_mlr_features(
            item_data,
            config["mlr_seasonality_max"],
            config["mlr_trend_decay_factors"]
        )

        for exp_name, split_dates in experiment_splits.items():
            train_range = split_dates["train_range"]
            test_range = split_dates["test_range"]
            logging.info(f"[{item_name} - {exp_name}] Train: {train_range[0].date()} to <{train_range[1].date()}, Test: {test_range[0].date()} to {test_range[1].date()}")

            # Get training data for this split
            train_data = item_data.loc[train_range[0]:train_range[1] - timedelta(days=1)] # Exclusive end date

            if train_data.empty:
                logging.warning(f"[{item_name} - {exp_name}] No training data in the specified range. Skipping split.")
                continue

            # --- Run Models ---
            # Moving Average
            for window in range(config["ma_window_min"], config["ma_window_max"] + 1):
                result = forecast_moving_average(train_data, test_range, item_actuals, item_name, window)
                if result is not None:
                    result["experiment"] = exp_name
                    result["model"] = "Moving Average"
                    all_results.append(result)

            # Exponential Smoothing
            for alpha in config["es_alphas"]:
                 result = forecast_exponential_smoothing(train_data, test_range, item_actuals, item_name, alpha, config)
                 if result is not None:
                    result["experiment"] = exp_name
                    result["model"] = "Exp Smoothing"
                    all_results.append(result)

            # Linear Regression
            result = forecast_linear_regression(train_data, test_range, item_actuals, item_name, config)
            if result is not None:
                result["experiment"] = exp_name
                result["model"] = "Linear Regression"
                all_results.append(result)

            # Multiple Linear Regression
            # Use pre-calculated features, slicing happens inside the function
            for decay in config["mlr_trend_decay_factors"]:
                 for seas in range(config["mlr_seasonality_min"], config["mlr_seasonality_max"] + 1):
                    result = forecast_multiple_linear_regression(
                        item_data_mlr_features, train_range, test_range, item_actuals, item_name, seas, decay, config
                        )
                    if result is not None:
                        result["experiment"] = exp_name
                        result["model"] = "MLR"
                        all_results.append(result)

    logging.info("Backtesting process completed.")
    if not all_results:
        logging.warning("No results were generated during the backtest.")
        return pd.DataFrame()

    # Combine all monthly results
    final_results_df = pd.concat(all_results).reset_index().rename(columns={"index": "Month"})
    final_results_df['Month'] = final_results_df['Month'].dt.to_timestamp() # Convert Period back to Timestamp if needed
    return final_results_df


def summarize_results(results_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Summarizes the backtest results to find the best parameters per model."""
    if results_df.empty:
        logging.warning("Cannot summarize empty results DataFrame.")
        return pd.DataFrame()

    logging.info("Summarizing results...")

    # Identify parameter columns dynamically
    param_cols = [col for col in results_df.columns if col.startswith('param_')]

    # Calculate average MAPE across experiments for each model/parameter set per item
    grouping_cols = ['Forecast Item', 'model'] + param_cols
    # Ensure grouping columns exist before using them
    valid_grouping_cols = [col for col in grouping_cols if col in results_df.columns]

    if not valid_grouping_cols:
         logging.error("No valid grouping columns found in results. Cannot summarize.")
         return pd.DataFrame()

    # Handle potential NaN in MAPE before averaging (e.g., if actuals were missing for a month)
    results_df['mape'] = results_df['mape'].fillna(1.0) # Penalize missing actuals? Or use dropna? Fill with 1 (100% error)

    # Calculate average MAPE
    summary = results_df.groupby(valid_grouping_cols, dropna=False)['mape'].mean().reset_index()
    summary = summary.rename(columns={'mape': 'Average_MAPE'})

    # Get top N results for each model type per item
    top_n = config["results_top_n_params"]
    best_results = summary.sort_values(by=['Forecast Item', 'model', 'Average_MAPE'])
    best_results = best_results.groupby(['Forecast Item', 'model']).head(top_n)

    # Clean up parameter column names for final output
    best_results.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in best_results.columns]

    # Sort final output for clarity
    final_summary = best_results.sort_values(by=['Forecast Item', 'Average_MAPE']).reset_index(drop=True)

    logging.info("Results summarized.")
    return final_summary


# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Run the entire backtesting process
        raw_results = run_backtest(CONFIG)

        if not raw_results.empty:
            # Summarize the results
            summary_df = summarize_results(raw_results, CONFIG)

            # Save the summary
            output_file = CONFIG["output_path"]
            summary_df.to_excel(output_file, index=False)
            logging.info(f"Summary results saved to {output_file}")
            print("\n--- Top Model Configurations ---")
            print(summary_df)
            print("-" * 30)
        else:
            logging.warning("Backtesting generated no results to summarize or save.")

    except FileNotFoundError as fnf_error:
        logging.error(f"File Not Found Error: {fnf_error}. Please check paths in CONFIG.")
    except ValueError as val_error:
        logging.error(f"Value Error: {val_error}. Check configuration or data consistency.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)