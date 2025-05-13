from forecasting_logic import load_and_prepare_history, load_and_prepare_actuals, prepare_mlr_features, forecast_multiple_linear_regression, CONFIG
import pandas as pd

# 1. Load data
weekly_history = load_and_prepare_history(CONFIG)
monthly_actuals = load_and_prepare_actuals(CONFIG)

# 2. Use the first available item for demonstration (can be changed)
forecast_items = weekly_history['Forecast Item'].unique()
print(f"Forecast Items found: {forecast_items}")

for item_name in forecast_items:
    print(f"\nProcessing item: {item_name}")
    item_data = weekly_history[weekly_history['Forecast Item'] == item_name].copy()
    item_actuals = monthly_actuals

    # Prepare MLR features
    item_data_mlr_features = prepare_mlr_features(
        item_data,
        seasonality_max=23,  # Only need up to 23 for this test
        decay_factors=[1.0]
    )

    # Use the latest available train/test split logic from your config
    # We'll mimic the last experiment from your logic
    # Find the latest possible train/test split
    from forecasting_logic import generate_experiment_splits
    experiment_splits = generate_experiment_splits(weekly_history, CONFIG)
    # Use the last experiment
    last_exp = list(experiment_splits.values())[-1]
    train_range = last_exp["train_range"]
    test_range = last_exp["test_range"]

    # Run MLR with seasonality=23, decay_factor=1.0
    result = forecast_multiple_linear_regression(
        item_data_mlr_features,
        train_range,
        test_range,
        item_actuals,
        item_name,
        seasonality=23,
        decay_factor=1.0,
        config=CONFIG
    )

    if result is not None:
        print(result[[col for col in result.columns if "mape" in col or "MAPE" in col or "actual" in col or "forecast" in col or "Month" in col or "Forecast Item" in col]])
        print("\nAverage MAPE for this test:", result['mape'].mean())
    else:
        print("No result for this configuration.")
