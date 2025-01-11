print('setting up strategies...')
import matplotlib.pyplot as plt
import pandas as pd
from machine_learning import optimize
from machine_learning import loss_functions
from data_collection.fetch_data import fetch_historical_data
from strategies.import_all import *
from data_collection.fetch_data import fetch_historical_data
import modules.backtester as backtest


print('setting up data...')
tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'META']
training_data = {"start":'2020-01-01', "end":'2023-01-01'}
testing_data = {"start":'2023-01-01',"end":'2025-01-01'}
max_evals = 100
pop_size = 5
optimization_technique = "hyperopt"
loss_function = loss_functions.sharpe_ratio_loss_function

data_frames = []
for i in range(len(tickers)):
    data_frames.append(fetch_historical_data(tickers[i], '1d', training_data["start"], training_data["end"]))

print('optimizing strategies...')

results = optimize.optimize(strategies, data_frames, loss_function, optimization_technique, max_evals=max_evals, population_size=pop_size)

print('results:')
print('Best loss:', results['best_loss'])
print('Best params:', results['best_params'])
print('Best strategy:', results['best_strategy'].__module__)

# print the data from the backtest on the optimized data and how that ended up actually looking

# Run backtest with optimized parameters
backtest_results_list = []
for df in data_frames:
    # Generate trading signals using best strategy and params
    trading_signals = results['best_strategy'](df['ohlc'], results['best_params'])
    # Run backtest
    backtest_results, _ = backtest.run_backtest(trading_signals)
    backtest_results_list.append(backtest_results)
combined_results_training = optimize.compile_backtest_results_sequential(backtest_results_list, data_frames)

market_return = ((combined_results_training['portfolio_values_over_time'][-1]['stock_value'] - combined_results_training['portfolio_values_over_time'][0]['stock_value']) / combined_results_training['portfolio_values_over_time'][0]['stock_value']) * 100


print('The optimized strategy on training data:')
print('Total Money Made: $', round(combined_results_training['total_amount_of_money_made'], 2))
print('Total Return: ', round(combined_results_training['total_percentage_gain'] * 100, 2), '%')
print(f"Market return: {market_return} %")
print('Number of Trades:', combined_results_training['total_trades'])
print('Average Hold Time:', combined_results_training['average_time_holding_position'])
print('Average Return per Year:', round(combined_results_training['average_return_per_year'] * 100, 2), '%')
print('Average Trades per Year:', combined_results_training['average_trades_per_year'])

# run a backtest with the new strategy and params but on the validation data only
print('\nRunning backtest on Validation data...')

# Fetch 2024 data
test_data_frames = []
for ticker in tickers:
    test_data_frames.append(fetch_historical_data(tickers[i], '1d', testing_data['start'], testing_data['end']))

# Run backtest with optimized parameters
backtest_results_list = []
for df in test_data_frames:
    # Generate trading signals using best strategy and params
    trading_signals = results['best_strategy'](df['ohlc'], results['best_params'])
    # Run backtest
    backtest_results, _ = backtest.run_backtest(trading_signals)
    backtest_results_list.append(backtest_results)

# Combine results from multiple tickers
combined_results = optimize.compile_backtest_results_sequential(backtest_results_list, test_data_frames)

print('Validation on unseen data Backtest Results:')
print('Total Money Made: $', round(combined_results['total_amount_of_money_made'], 2))
print('Total Return: ', round(combined_results['total_percentage_gain'] * 100, 2), '%')
print('Number of Trades:', combined_results['total_trades'])
print('Average Hold Time:', combined_results['average_time_holding_position'])
print('Average Return per Year:', round(combined_results['average_return_per_year'] * 100, 2), '%')
print('Average Trades per Year:', combined_results['average_trades_per_year'])

# Plot the portfolio values over time
portfolio_values = pd.DataFrame(combined_results['portfolio_values_over_time'])

plt.figure(figsize=(10, 6))
plt.plot(portfolio_values['date_time'], portfolio_values['value'], label='Portfolio Value')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time (2024)')
plt.legend()
plt.grid()

# Save the plot as a PNG image
plot_filename = 'portfolio_value_2024.png'
plt.savefig(plot_filename, dpi=300)
plt.close()

print(f'Portfolio value chart saved as {plot_filename}.')
# Save portfolio values over time to a CSV file
csv_filename = 'portfolio_values_2024.csv'
portfolio_values.to_csv(csv_filename, index=False)
print(f'Portfolio values saved as {csv_filename}.')
