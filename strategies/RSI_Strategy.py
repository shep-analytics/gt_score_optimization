import pandas as pd

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(data, rsi_buy_threshold=30, rsi_sell_threshold=70, window=14):
    data_copy = data.copy()
    data_copy['RSI'] = calculate_rsi(data_copy, window)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        if data_copy['RSI'].iloc[i] < rsi_buy_threshold and data_copy['RSI'].iloc[i-1] >= rsi_buy_threshold:
            data_copy.at[i, 'action'] = 'buy'
        elif data_copy['RSI'].iloc[i] > rsi_sell_threshold and data_copy['RSI'].iloc[i-1] <= rsi_sell_threshold:
            data_copy.at[i, 'action'] = 'sell'
    
    return data_copy

# Example usage:
# from data_collection.fetch_data import fetch_historical_data
# df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
# result = rsi_strategy(df)
# for index, row in result.iterrows():
#     print(f"Date: {row['Date']}, Close: {row['Close']}, RSI: {row['RSI']}, Action: {row['action']}")