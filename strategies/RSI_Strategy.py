import pandas as pd

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def check_rsi_action(current_rsi, previous_rsi, rsi_buy_threshold, rsi_sell_threshold):
    if current_rsi < rsi_buy_threshold and previous_rsi >= rsi_buy_threshold:
        return 'buy'
    elif current_rsi > rsi_sell_threshold and previous_rsi <= rsi_sell_threshold:
        return 'sell'
    return 'none'

def strategy(data, params={'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14}):
    rsi_buy_threshold = params.get('rsi_buy_threshold')
    rsi_sell_threshold = params.get('rsi_sell_threshold')
    window = params.get('window')
    
    data_copy = data.copy()
    data_copy['RSI'] = calculate_rsi(data_copy, window)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_rsi_action(
            current_rsi=data_copy['RSI'].iloc[i],
            previous_rsi=data_copy['RSI'].iloc[i - 1],
            rsi_buy_threshold=rsi_buy_threshold,
            rsi_sell_threshold=rsi_sell_threshold
        )
    
    return data_copy

def should_buy_live(data, params={'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14}):
    if len(data) < 2:
        raise ValueError("Insufficient data to calculate RSI")
    
    rsi_buy_threshold = params.get('rsi_buy_threshold')
    rsi_sell_threshold = params.get('rsi_sell_threshold')
    window = params.get('window')
    
    recent_data = data[-(window + 1):]  # Only take the last `window` periods
    recent_rsi = calculate_rsi(recent_data, window)
    
    current_rsi = recent_rsi.iloc[-1]
    previous_rsi = recent_rsi.iloc[-2]
    
    return check_rsi_action(current_rsi, previous_rsi, rsi_buy_threshold, rsi_sell_threshold)

# Example usage:
# df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
# params = {'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14}
# print(should_buy_live(df, params))
