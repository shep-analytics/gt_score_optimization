print('setting up strategies...')
from strategies.RSI_Strategy import strategy as RSI_strategy, should_buy_live as RSI_should_buy_live, param_space as RSI_param_space, ga_bounds as RSI_ga_bounds
from strategies.MACD_Strategy import strategy as MACD_strategy, should_buy_live as MACD_should_buy_live, param_space as MACD_param_space, ga_bounds as MACD_ga_bounds
from strategies.BollingerBands_Strategy import strategy as BollingerBands_strategy, should_buy_live as BollingerBands_should_buy_live, param_space as BollingerBands_param_space, ga_bounds as BollingerBands_ga_bounds

strategies = [
    {'strategy': RSI_strategy, 'params': {'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14}, 'param_space': RSI_param_space, 'ga_bounds':RSI_ga_bounds},
    {'strategy': MACD_strategy, 'params': {'short_window': 12, 'long_window': 26, 'signal_window': 9}, 'param_space': MACD_param_space, 'ga_bounds':MACD_ga_bounds},
    {'strategy': BollingerBands_strategy, 'params': {'window': 20, 'num_std_dev': 2}, 'param_space': BollingerBands_param_space, 'ga_bounds':BollingerBands_ga_bounds}
]

print('setting up data...')
from data_collection.fetch_data import fetch_historical_data
tickers = ['AAPL', 'WMT', 'TSLA']
data_frames = []
for i in range(len(tickers)):
    data_frames.append(fetch_historical_data(tickers[i], '1d', '2010-01-01', '2024-01-01'))

print('optimizing strategies...')
from machine_learning import optimize
from machine_learning import loss_functions
from data_collection.fetch_data import fetch_historical_data
results = optimize.optimize(strategies, data_frames, loss_functions.simple_loss_function, "hyperopt")

print('results:')
print('Best loss:', results['best_loss'])
print('Best params:', results['best_params'])
print('Best strategy:', results['best_strategy'].__module__)

# run a backtest with the new strategy and params but on the year of 2024 only
print('\nRunning backtest for 2024...')

# Fetch 2024 data
test_data_frames = []
for ticker in tickers:
    test_data_frames.append(fetch_historical_data(ticker, '1d', '2024-01-01', '2024-12-31'))

import modules.backtester as backtest

# Run backtest with optimized parameters
backtest_results_list = []
for df in test_data_frames:
    # Generate trading signals using best strategy and params
    trading_signals = results['best_strategy'](df['ohlc'], results['best_params'])
    # Run backtest
    backtest_results, _ = backtest.run_backtest(trading_signals)
    backtest_results_list.append(backtest_results)

# Combine results from multiple tickers
combined_results = optimize.compile_backtest_results(backtest_results_list, test_data_frames)

print('2024 Backtest Results:')
print('Total Money Made: $', round(combined_results['total_amount_of_money_made'], 2))
print('Total Return: ', round(combined_results['total_percentage_gain'] * 100, 2), '%')
print('Number of Trades:', combined_results['total_trades'])
print('Average Hold Time:', combined_results['average_time_holding_position'])
print('Average Return per Year:', round(combined_results['average_return_per_year'] * 100, 2), '%')
print('Average Trades per Year:', combined_results['average_trades_per_year'])