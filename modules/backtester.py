def run_backtest(trading_signals, starting_cash=1000000, commission=0.001, spread=0.01):

    import pandas as pd
    from datetime import timedelta

    """
    Run a backtest based on buy and sell actions.

    Parameters:
    - trading_signals (DataFrame): Contains 'datetime', 'action', and 'price' columns.
    - starting_cash (float): The initial amount of cash. Default is $1,000,000.
    - commission (float): The percentage commission on each trade. Default is 0.001 (0.1%).
    - spread (float): The price spread in dollars. Default is $0.01.

    Returns:
    - dict: Contains trade details and performance metrics.
    - DataFrame: The trading signals DataFrame with portfolio values.
    """

    cash = starting_cash
    position = 0
    entry_price = 0
    entry_time = None
    trades_history = []
    portfolio_values = []

    trading_signals = trading_signals.sort_values('Date').reset_index(drop=True)
    for index, row in trading_signals.iterrows():
        action = row['action']
        price = row['Close']
        time = row['Date']

        if action == 'buy' and position == 0:
            buy_price = price + spread
            buy_cost = cash * (1 - commission)
            position = buy_cost / buy_price
            cash -= buy_cost
            entry_price = buy_price
            entry_time = time

        elif action == 'sell' and position > 0:
            sell_price = price - spread
            sell_revenue = position * sell_price * (1 - commission)
            cash += sell_revenue
            profit = sell_revenue - (position * entry_price)
            profit_percent = (profit / (position * entry_price))
            time_held = time - entry_time

            trades_history.append({
                'purchase_price': entry_price,
                'sale_price': sell_price,
                'purchase_date': entry_time,
                'sale_date': time,
                'profit_loss_percent': profit_percent,
                'profit_loss_dollars': profit,
                'time_held': time_held
            })

            position = 0
            entry_price = 0
            entry_time = None

        # Calculate portfolio value
        portfolio_value = cash + (position * price if position > 0 else 0)
        portfolio_values.append(portfolio_value)

    if position > 0:
        sell_price = trading_signals.iloc[-1]['price'] - spread
        sell_revenue = position * sell_price * (1 - commission)
        cash += sell_revenue
        profit = sell_revenue - (position * entry_price)
        profit_percent = (profit / (position * entry_price))
        time_held = trading_signals.iloc[-1]['datetime'] - entry_time

        trades_history.append({
            'purchase_price': entry_price,
            'sale_price': sell_price,
            'purchase_date': entry_time,
            'sale_date': trading_signals.iloc[-1]['datetime'],
            'profit_loss_percent': profit_percent,
            'profit_loss_dollars': profit,
            'time_held': time_held
        })

    total_trades = len(trades_history)
    total_money_made = cash - starting_cash
    total_percentage_gain = (total_money_made / starting_cash)

    total_days = (trading_signals['Date'].iloc[-1] - trading_signals['Date'].iloc[0]).days or 1

    time_held_list = [trade['time_held'] for trade in trades_history]
    if time_held_list:
        average_time_holding_position = sum(time_held_list, timedelta()) / len(time_held_list)
        longest_time_position_held = max(time_held_list)
    else:
        average_time_holding_position = timedelta(0)
        longest_time_position_held = timedelta(0)

    trading_signals['portfolio_value'] = portfolio_values

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
        'trading_signals': trading_signals
    }, trading_signals
