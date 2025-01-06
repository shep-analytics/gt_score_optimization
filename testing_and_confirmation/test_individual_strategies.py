# Example usage
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os

    from data_collection.fetch_data import fetch_historical_data
    df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
    
    from modules.backtester import run_backtest
    from modules.live_sim_backtest import run_live_simulation

    # from strategies.RSI_Strategy import strategy, should_buy_live
    # params = {'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14}

    # from strategies.MACD_Strategy import strategy, should_buy_live
    # params = {'short_window': 12, 'long_window': 26, 'signal_window': 9}
    
    from strategies.BollingerBands_Strategy import strategy, should_buy_live
    params = {'window': 20, 'num_std_dev': 2}

    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Run the backtest
    trading_signals = strategy(df, params)
    backtest_results, trading_signals_with_portfolio = run_backtest(trading_signals)
    # Save the trading signals with portfolio to a CSV file for easy analysis
    trading_signals_with_portfolio.to_csv(os.path.join(output_dir, 'backtest_trading_signals_with_portfolio.csv'), index=False)

    # Run the live simulation backtest
    live_sim_results, live_trading_signals_with_portfolio = run_live_simulation(should_buy_live, df, params=params, save_path=os.path.join(output_dir, 'live_simulation_trading_signals_with_portfolio.csv'))

    # Print summary results for backtest
    print("Backtest Results:")
    print("Total trades:", backtest_results['total_trades'])
    print("Final cash:", backtest_results['final_cash'])
    print("Total amount of money made:", backtest_results['total_amount_of_money_made'])
    print("Total percentage gain:", backtest_results['total_percentage_gain'])
    print("Average yearly percentage gain:", backtest_results['average_yearly_percentage_gain'])
    print("Average monthly percentage gain:", backtest_results['average_monthly_percentage_gain'])
    print("Average time holding position:", backtest_results['average_time_holding_position'])
    print("Longest time position held:", backtest_results['longest_time_position_held'])

    # Print summary results for live simulation
    print("\nLive Simulation Results:")
    print("Total trades:", live_sim_results['total_trades'])
    print("Final cash:", live_sim_results['final_cash'])
    print("Total amount of money made:", live_sim_results['total_amount_of_money_made'])
    print("Total percentage gain:", live_sim_results['total_percentage_gain'])
    print("Average yearly percentage gain:", live_sim_results['average_yearly_percentage_gain'])
    print("Average monthly percentage gain:", live_sim_results['average_monthly_percentage_gain'])
    print("Average time holding position:", live_sim_results['average_time_holding_position'])
    print("Longest time position held:", live_sim_results['longest_time_position_held'])

    # Plot portfolio value over time for backtest
    plt.figure(figsize=(12, 6))
    plt.plot(trading_signals_with_portfolio['Date'], trading_signals_with_portfolio['portfolio_value'], label='Backtest Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Backtest Portfolio Value Over Time')

    # Determine the x-axis ticks
    total_days = (trading_signals_with_portfolio['Date'].iloc[-1] - trading_signals_with_portfolio['Date'].iloc[0]).days
    if total_days > 3 * 365:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    elif total_days > 3 * 30:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(24))
    else:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(90))

    plt.legend()
    plt.savefig(os.path.join(output_dir, 'backtest_portfolio_value.png'))  # Save the plot to a file

    # Plot portfolio value over time for live simulation
    plt.figure(figsize=(12, 6))
    plt.plot(live_trading_signals_with_portfolio['Date'], live_trading_signals_with_portfolio['portfolio_value'], label='Live Simulation Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Live Simulation Portfolio Value Over Time')

    # Determine the x-axis ticks
    total_days = (live_trading_signals_with_portfolio['Date'].iloc[-1] - live_trading_signals_with_portfolio['Date'].iloc[0]).days
    if total_days > 3 * 365:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    elif total_days > 3 * 30:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(24))
    else:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(90))

    plt.legend()
    plt.savefig(os.path.join(output_dir, 'live_simulation_portfolio_value.png'))  # Save the plot to a file
