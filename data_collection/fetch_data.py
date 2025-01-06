import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_historical_data(ticker, interval, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): The ticker symbol of the stock.
    interval (str): The data frequency ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
    start_date (str): The start date for fetching data (YYYY-MM-DD).
    end_date (str): The end date for fetching data (YYYY-MM-DD).

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data with the index being a sequential number and the date as a column.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(interval=interval, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Index'] = range(1, len(data) + 1)
    data.set_index('Index', inplace=True)
    ohlc_data = data[['Date', 'Open', 'High', 'Low', 'Close']]
    ohlc_data = ohlc_data.round(2)  # Round the values to 2 decimal places
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_time_passed = end - start
    
    return {'ohlc':ohlc_data, 'time_passed':total_time_passed}

# Example usage:

# df = fetch_historical_data('AAPL', '1d', '2020-01-01', '2021-01-01')
# print(df)