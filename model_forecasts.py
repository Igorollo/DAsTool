import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from typing import Dict, List, Any, Optional
import logging


def forecast_moving_average(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp],
    window: int
) -> Optional[pd.DataFrame]:
    """Fits Moving Average and generates forecast for the test period."""
    model_name = "Moving Average"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"

    if train_data.empty or len(train_data) < window:
        logging.warning(f"[Moving Average] Insufficient training data ({len(train_data)}) for window {window}.")
        return None

    try:
        rolling_mean = train_data['history'].rolling(window=window).mean()
        last_known_mean = rolling_mean.iloc[-1]
        if pd.isna(last_known_mean):
            logging.warning(f"[Moving Average] Last rolling mean is NaN for window {window}.")
            return None
        test_start_date, test_end_date = test_range
        test_dates = pd.date_range(start=test_start_date, end=test_end_date, freq='W-MON')
        if test_dates.empty:
            logging.warning(f"[Moving Average] No valid Mondays found in test range for window {window}.")
            return None
        test_forecast_df = pd.DataFrame(index=test_dates)
        test_forecast_df[forecast_col] = last_known_mean
        test_forecast_df[forecast_col] = test_forecast_df[forecast_col].clip(lower=0)
        return test_forecast_df
    except Exception as e:
        logging.error(f"[Moving Average] Error during forecasting with window {window}: {e}", exc_info=True)
        return None


def forecast_exponential_smoothing(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp],
    alpha: float
) -> Optional[pd.DataFrame]:
    """Fits Simple Exponential Smoothing and generates forecast for the test period."""
    model_name = "Exp Smoothing"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"

    if train_data.empty:
        logging.warning(f"[Exp Smoothing] Training data empty for alpha {alpha}.")
        return None
    try:
        train_data_weekly = train_data['history'].asfreq('W-MON')
        if train_data_weekly.isnull().any():
            train_data_weekly = train_data_weekly.ffill()
        if train_data_weekly.isnull().any():
            logging.warning(f"[Exp Smoothing] NaNs in training data after freq conversion for alpha {alpha}.")
            return None
        model = sm.tsa.SimpleExpSmoothing(train_data_weekly, initialization_method='estimated')
        fit = model.fit(smoothing_level=alpha, optimized=False)
        test_start_date, test_end_date = test_range
        last_train_date = train_data_weekly.index.max()
        weeks_needed = (test_end_date - last_train_date).days // 7 + 5
        weeks_needed = max(1, weeks_needed)
        forecast_values = fit.forecast(steps=weeks_needed)
        forecast_values = forecast_values.clip(lower=0)
        forecast_index = pd.date_range(start=last_train_date + timedelta(days=7), periods=weeks_needed, freq='W-MON')
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]
        if test_forecast_series.empty:
            logging.warning(f"[Exp Smoothing] Forecast did not cover test range for alpha {alpha}.")
            return None
        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])
        return test_forecast_df
    except ValueError as ve:
        logging.warning(f"[Exp Smoothing] Value error during model fit (alpha={alpha}): {ve}")
        return None
    except Exception as e:
        logging.error(f"[Exp Smoothing] Error during forecasting with alpha {alpha}: {e}", exc_info=True)
        return None


def forecast_linear_regression(
    train_data: pd.DataFrame,
    test_range: List[pd.Timestamp]
) -> Optional[pd.DataFrame]:
    """Fits Linear Regression (trend) and generates forecast for the test period."""
    model_name = "Linear Regression"
    forecast_col = f"forecast_{model_name.lower().replace(' ','_')}"
    if train_data.empty:
        logging.warning(f"[Linear Regression] Training data empty.")
        return None
    try:
        train_data_lr = train_data.copy()
        train_data_lr['time_index'] = np.arange(len(train_data_lr))
        X_train = train_data_lr[['time_index']]
        y_train = train_data_lr['history']
        model = LinearRegression()
        model.fit(X_train, y_train)
        test_start_date, test_end_date = test_range
        last_train_date = train_data.index.max()
        first_future_index = len(train_data)
        future_dates = pd.date_range(start=last_train_date + timedelta(days=7), end=test_end_date + timedelta(days=6), freq='W-MON')
        future_indices = np.arange(first_future_index, first_future_index + len(future_dates))
        X_future = pd.DataFrame({'time_index': future_indices})
        forecast_values = model.predict(X_future)
        forecast_values = np.maximum(0, forecast_values)
        forecast_series = pd.Series(forecast_values, index=future_dates)
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]
        if test_forecast_series.empty:
            logging.warning(f"[Linear Regression] Forecast did not cover test range.")
            return None
        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])
        return test_forecast_df
    except Exception as e:
        logging.error(f"[Linear Regression] Error during forecasting: {e}", exc_info=True)
        return None


def forecast_multiple_linear_regression(
    item_data_with_features: pd.DataFrame,
    train_range: List[pd.Timestamp],
    test_range: List[pd.Timestamp],
    seasonality: int,
    decay_factor: float,
    train_weeks: int
) -> Optional[pd.DataFrame]:
    """Fits MLR with seasonality and trend decay, generates forecast for the test period."""
    model_name = "MLR"
    forecast_col = f"forecast_{model_name.lower()}"
    if item_data_with_features.empty:
        logging.warning(f"[MLR] Feature data empty for params seasonality={seasonality}, decay_factor={decay_factor}.")
        return None
    try:
        train_start_date, train_end_date = train_range
        test_start_date, test_end_date = test_range
        full_train_data = item_data_with_features.loc[train_start_date:train_end_date - timedelta(days=1)]
        if len(full_train_data) > train_weeks:
            train_data_mlr = full_train_data.iloc[-train_weeks:]
        else:
            train_data_mlr = full_train_data
        if train_data_mlr.empty:
            logging.warning(f"[MLR] Insufficient training data after slicing for {train_weeks} weeks. Params seasonality={seasonality}, decay_factor={decay_factor}.")
            return None
        potential_test_data = item_data_with_features.loc[train_end_date:]
        if potential_test_data.empty:
            logging.warning(f"[MLR] No data available after training end date {train_end_date.date()} for params seasonality={seasonality}, decay_factor={decay_factor}.")
            return None
        seasonal_features = [f'Week_{i}' for i in range(seasonality)]
        if abs(decay_factor - 1.0) < 1e-6:
            trend_feature = 'Decayed_Numerical_Date_Factor_1.0'
        elif abs(decay_factor - 0.05) < 1e-6:
            trend_feature = 'Decayed_Numerical_Date_Factor_0.05'
        else:
            logging.error(f"[MLR] Invalid decay factor {decay_factor} specified for feature selection.")
            return None
        features = seasonal_features + [trend_feature]
        if not all(f in train_data_mlr.columns for f in features):
            missing = [f for f in features if f not in train_data_mlr.columns]
            logging.error(f"[MLR] Missing required features: {missing} for params seasonality={seasonality}, decay_factor={decay_factor}")
            return None
        X_train = train_data_mlr[features]
        y_train = train_data_mlr['history']
        X_test = potential_test_data[features]
        model = LinearRegression()
        model.fit(X_train, y_train)
        forecast_values = model.predict(X_test)
        forecast_values = np.maximum(0, forecast_values)
        forecast_series = pd.Series(forecast_values, index=potential_test_data.index)
        test_forecast_series = forecast_series.loc[test_start_date:test_end_date]
        if test_forecast_series.empty:
            logging.warning(f"[MLR] Forecast did not cover test range for params seasonality={seasonality}, decay_factor={decay_factor}.")
            return None
        test_forecast_df = pd.DataFrame(test_forecast_series, columns=[forecast_col])
        return test_forecast_df
    except Exception as e:
        logging.error(f"[MLR] Error during forecasting with params seasonality={seasonality}, decay_factor={decay_factor}: {e}", exc_info=True)
        return None
