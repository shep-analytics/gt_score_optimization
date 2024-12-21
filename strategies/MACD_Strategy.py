import pandas as pd

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def check_macd_action(current_macd, current_signal, previous_macd, previous_signal):
    if current_macd > current_signal and previous_macd <= previous_signal:
        return 'buy'
    elif current_macd < current_signal and previous_macd >= previous_signal:
        return 'sell'
    return 'none'

def strategy(data, params={'short_window': 12, 'long_window': 26, 'signal_window': 9}):
    short_window = params.get('short_window')
    long_window = params.get('long_window')
    signal_window = params.get('signal_window')
    
    data_copy = data.copy()
    data_copy['MACD'], data_copy['Signal'] = calculate_macd(data_copy, short_window, long_window, signal_window)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_macd_action(
            current_macd=data_copy['MACD'].iloc[i-1],
            current_signal=data_copy['Signal'].iloc[i-1],
            previous_macd=data_copy['MACD'].iloc[i-2],
            previous_signal=data_copy['Signal'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'short_window': 12, 'long_window': 26, 'signal_window': 9}):
    if len(data) < params.get('long_window'):
        decision = ['none', 0]
    
    short_window = params.get('short_window')
    long_window = params.get('long_window')
    signal_window = params.get('signal_window')
    
    recent_data = data  # Only take the last `long_window` periods
    recent_macd, recent_signal = calculate_macd(recent_data, short_window, long_window, signal_window)
        
    current_macd = recent_macd.iloc[-1]
    current_signal = recent_signal.iloc[-1]
    previous_macd = recent_macd.iloc[-2]
    previous_signal = recent_signal.iloc[-2]
    
    decision = [check_macd_action(current_macd, current_signal, previous_macd, previous_signal), current_macd]
    
    return decision

# Example usage:
# df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
# params = {'short_window': 12, 'long_window': 26, 'signal_window': 9}
# print(should_buy_live(df, params))