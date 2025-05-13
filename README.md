# Demand Forecasting GUI Application

This application provides a graphical user interface for the demand forecasting model comparison tool. It allows you to evaluate different time-series forecasting models (Moving Average, Exponential Smoothing, Linear Regression, Multiple Linear Regression) to find the best fit for demand forecasting based on historical weekly data.

## Features

- Upload historical weekly data and actual monthly shipments via drag-and-drop or file selection
- Configure forecasting parameters through an intuitive interface
- Run forecasting models and view comparative results
- Visualize model performance with interactive charts
- Download results as Excel files

## Requirements

- Python 3.7+
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run forecasting_app.py
   ```

2. In the web interface:
   - Upload your historical weekly data and monthly actuals files using the sidebar
   - Adjust configuration parameters as needed
   - Click 'Run Forecasting' to begin the analysis
   - View and download the results

## Input File Requirements

### Historical Weekly Data (Excel format)
- Should contain columns for "Forecast Item", "Date", and the history column (default: "Actual.1")
- Weekly demand values for each forecast item

### Actual Monthly Shipments (Excel format)
- Should contain monthly actual shipment data in a transposed format
- Forecast items in the header row

## Configuration

The application allows you to customize various parameters:
- Forecast lag (months)
- Number of experiments for backtesting
- Moving Average window range
- Seasonality parameters for Multiple Linear Regression
- And more through the Advanced Configuration section
