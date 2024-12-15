# Example usage
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from data_collection.fetch_data import fetch_historical_data
    df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
    
    from strategies.RSI_Strategy import rsi_strategy
    trading_signals = rsi_strategy(df)

    from modules.backtester import run_backtest
    # Run the backtest
    results, trading_signals_with_portfolio = run_backtest(trading_signals)

    # Print summary results
    print("Total trades:", results['total_trades'])
    print("Final cash:", results['final_cash'])
    print("Total amount of money made:", results['total_amount_of_money_made'])
    print("Total percentage gain:", results['total_percentage_gain'])
    print("Average yearly percentage gain:", results['average_yearly_percentage_gain'])
    print("Average monthly percentage gain:", results['average_monthly_percentage_gain'])
    print("Average time holding position:", results['average_time_holding_position'])
    print("Longest time position held:", results['longest_time_position_held'])

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(trading_signals_with_portfolio['Date'], trading_signals_with_portfolio['portfolio_value'], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')

    # Determine the x-axis ticks
    total_days = (trading_signals_with_portfolio['Date'].iloc[-1] - trading_signals_with_portfolio['Date'].iloc[0]).days
    if total_days > 3 * 365:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    elif total_days > 3 * 30:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(24))
    else:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(90))

    plt.legend()
    # plt.show()
    plt.savefig('portfolio_value.png')  # Save the plot to a file
