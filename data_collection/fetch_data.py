import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_historical_data(ticker, interval, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance and adjust for stock splits.

    Parameters:
    ticker (str): The ticker symbol of the stock.
    interval (str): The data frequency ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
    start_date (str): The start date for fetching data (YYYY-MM-DD).
    end_date (str): The end date for fetching data (YYYY-MM-DD).

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data adjusted for splits with the index being a sequential number and the date as a column.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(interval=interval, start=start_date, end=end_date)
    
    # Add sequential index
    data.reset_index(inplace=True)
    data['Index'] = range(1, len(data) + 1)
    data.set_index('Index', inplace=True)

    # Extract and round OHLC data
    ohlc_data = data[['Date', 'Open', 'High', 'Low', 'Close']]
    ohlc_data = ohlc_data.round(2)

    # Calculate total time passed
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_time_passed = end - start

    return {'ohlc': ohlc_data, 'time_passed': total_time_passed}

# Example usage:
# result = fetch_historical_data('AAPL', '1d', '2020-01-01', '2021-01-01')
# print(result['ohlc'])
