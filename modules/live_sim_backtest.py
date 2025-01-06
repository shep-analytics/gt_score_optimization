import pandas as pd
from datetime import timedelta


def run_live_simulation(strategy_function, data, starting_cash=1000000, commission=0.001, spread=0.01, params=None, save_path=None):
    """
    Run a live simulation backtest based on a strategy function.

    Parameters:
    - strategy_function (callable): The strategy function to test (e.g., should_buy_live).
    - data (DataFrame): Contains 'Date' and 'Close' columns with historical price data.
    - starting_cash (float): The initial amount of cash.
    - commission (float): The percentage commission on each trade.
    - spread (float): The price spread in dollars.
    - params (dict): Parameters to pass to the strategy function.
    - save_path (str): Optional. Path to save the resulting DataFrame as a CSV file.

    Returns:
    - dict: Contains trade details and performance metrics.
    - DataFrame: The trading signals DataFrame with portfolio values.
    """

    if params is None:
        params = {}

    cash = starting_cash
    position = 0
    entry_price = 0
    entry_time = None
    trades_history = []
    portfolio_values = []

    data = data.sort_values('Date').reset_index(drop=True)
    data['action'] = 'none'
    data['RSI'] = None

    for i in range(len(data)):
        current_time = data.at[i, 'Date']
        current_price = data.at[i, 'Close']
        current_data = data.iloc[:i+1]

        # pulls a list that has the action and the RSI value [action, RSI]
        action_full = strategy_function(current_data, params)

        action = action_full[0]
        
    
        data.at[i, 'action'] = action
        data.at[i, 'RSI'] = action_full[1]

        if action == 'buy' and position == 0:
            buy_price = current_price + spread
            buy_cost = cash * (1 - commission)
            position = buy_cost / buy_price
            cash -= buy_cost
            entry_price = buy_price
            entry_time = current_time

        elif action == 'sell' and position > 0:
            sell_price = current_price - spread
            sell_revenue = position * sell_price * (1 - commission)
            cash += sell_revenue
            profit = sell_revenue - (position * entry_price)
            profit_percent = profit / (position * entry_price)
            time_held = current_time - entry_time

            trades_history.append({
                'purchase_price': entry_price,
                'sale_price': sell_price,
                'purchase_date': entry_time,
                'sale_date': current_time,
                'profit_loss_percent': profit_percent,
                'profit_loss_dollars': profit,
                'time_held': time_held
            })

            position = 0
            entry_price = 0
            entry_time = None

        portfolio_value = cash + (position * current_price if position > 0 else 0)
        portfolio_values.append(portfolio_value)

    if position > 0:
        sell_price = data.iloc[-1]['Close'] - spread
        sell_revenue = position * sell_price * (1 - commission)
        cash += sell_revenue
        profit = sell_revenue - (position * entry_price)
        profit_percent = profit / (position * entry_price)
        time_held = data.iloc[-1]['Date'] - entry_time

        trades_history.append({
            'purchase_price': entry_price,
            'sale_price': sell_price,
            'purchase_date': entry_time,
            'sale_date': data.iloc[-1]['Date'],
            'profit_loss_percent': profit_percent,
            'profit_loss_dollars': profit,
            'time_held': time_held
        })

    total_trades = len(trades_history)
    total_money_made = cash - starting_cash
    total_percentage_gain = total_money_made / starting_cash
    total_days = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days or 1

    time_held_list = [trade['time_held'] for trade in trades_history]
    if time_held_list:
        average_time_holding_position = sum(time_held_list, timedelta()) / len(time_held_list)
        longest_time_position_held = max(time_held_list)
    else:
        average_time_holding_position = timedelta(0)
        longest_time_position_held = timedelta(0)

    data['portfolio_value'] = portfolio_values

    if save_path:
        data.to_csv(save_path, index=False)

    return {
        'trades_history': trades_history,
        'total_trades': total_trades,
        'final_cash': cash,
        'total_amount_of_money_made': total_money_made,
        'total_percentage_gain': total_percentage_gain,
        'average_yearly_percentage_gain': total_percentage_gain / (total_days / 365),
        'average_monthly_percentage_gain': total_percentage_gain / (total_days / 30),
        'average_time_holding_position': average_time_holding_position,
        'longest_time_position_held': longest_time_position_held,
        'trading_signals': data
    }, data