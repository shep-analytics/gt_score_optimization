from modules import backtester
import pandas as pd


def optimize(strategies, data_frames, loss_function):
    """
    Optimize the given strategies using the given data.

    Parameters:
    strategies (list of dictionaries): [{'strategy': strategy_function, 'params': {'param1': value1, 'param2': value2}}, ...]
    data_frames (list of pandas.DataFrame): List of data frames to optimize the strategies on
    loss_function (function): Function that takes in the backtest results and returns a loss value
    """
    
    # compiles all backtests into a single result for evaluation so that multiple datasets can be evaluated for a single loss value
    def compile_backtest_results(results):
        compiled_results = results[0]
        for i in range(1, len(results)):
            for key in compiled_results:
                if isinstance(compiled_results[key], list):
                    compiled_results[key].extend(results[i][key])
                elif isinstance(compiled_results[key], pd.DataFrame):
                    compiled_results[key] = pd.concat([compiled_results[key], results[i][key]], ignore_index=True)
                else:
                    compiled_results[key] += results[i][key]
        return compiled_results
        
    best_loss = float('inf')
    best_params = None
    best_strategy = None
    
    # iterate through each strategy and run the backtest
    for strategy_dict in strategies:
        strategy = strategy_dict['strategy']
        params = strategy_dict['params']
        
        # if we are using multiple tickers, the strategy will be run on each ticker and the results will be compiled to compute a single loss value training across the entire dataset
        if len(data_frames) > 1:
            results = []
            for data in data_frames:
                trading_signals = strategy(data, params)
                backtest_results, _ = backtester.run_backtest(trading_signals)
                results.append(backtest_results)
            backtest_results = compile_backtest_results(results)
        # if we are using a single ticker, we can just run the backtest on that single ticker
        else:
            backtest_results, _ = backtester.run_backtest(strategy(data_frames[0], params))
        
        # calculate the loss value for the backtest results
        loss = loss_function(backtest_results)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_strategy = strategy
    
    optimization_results = {
        'best_loss': best_loss,
        'best_params': best_params,
        'best_strategy': best_strategy
    }
    return(optimization_results)