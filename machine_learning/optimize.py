# optimize.py

import random
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import timedelta
import copy
from datetime import datetime
from collections import defaultdict
from modules import backtester

warnings.filterwarnings("ignore")

# Optional: Hyperopt for Bayesian-like optimization
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_INSTALLED = True
except ImportError:
    HYPEROPT_INSTALLED = False

# Optional: DEAP for genetic algorithms
try:
    from deap import base, creator, tools, algorithms
    DEAP_INSTALLED = True
except ImportError:
    DEAP_INSTALLED = False


def calculate_average_yearly_gain(portfolio_values):
    """
    Calculates the average yearly percentage gain from a list of portfolio values over time.
    
    Args:
        portfolio_values (list): A list of dictionaries with "date_time" (pandas Timestamp) and "value" keys.

    Returns:
        float: The average yearly percentage gain as a percentage (e.g., 8.0 for 8%).
    """
    # Parse the data into a dictionary grouped by year
    yearly_values = defaultdict(list)
    for record in portfolio_values:
        # Ensure date_time is a pandas Timestamp and extract the year
        date = record["date_time"].to_pydatetime() if hasattr(record["date_time"], "to_pydatetime") else record["date_time"]
        yearly_values[date.year].append(record["value"])
    
    yearly_gains = []
    
    # Calculate yearly gains
    for year, values in sorted(yearly_values.items()):
        if len(values) >= 2:  # Ensure we have at least start and end values
            start_value = values[0]
            end_value = values[-1]
            yearly_gain = ((end_value - start_value) / start_value)
            yearly_gains.append(yearly_gain)
    
    # Handle cases where years are incomplete
    if len(yearly_gains) > 0:
        average_yearly_gain = sum(yearly_gains) / len(yearly_gains)
    else:
        average_yearly_gain = 0.0

    return average_yearly_gain


def compile_backtest_results_sequential(results, data_frames):
    """
    Combine multiple backtest results (dicts) into one dictionary.

    This version preserves chronological order by shifting dates and scaling portfolio values
    to ensure continuity across multiple tickers or timeframes.

    Parameters:
    - results: List of backtest result dictionaries.
    - data_frames: List of DataFrames used in the backtests.

    Returns:
    - dict: Compiled results with adjusted portfolio values and chronological order.
    """
    import pandas as pd
    from datetime import timedelta
    first_backtest = copy.deepcopy(results[0])
    compiled_results = {}
        
    compiled_portfolio_values = first_backtest['portfolio_values_over_time']
    compiled_trades_history = first_backtest['trades_history']

    # Track the last value and date of the previous backtest
    last_value = compiled_portfolio_values[-1]['value']
    last_date = pd.to_datetime(compiled_portfolio_values[-1]['date_time'])

    last_stock_value = compiled_portfolio_values[-1]['stock_value']

    for i in range(1, len(results)):
        
        current_results = copy.deepcopy(results[i])
        current_portfolio_values = copy.deepcopy(current_results['portfolio_values_over_time'])

        # Shift the dates for the current portfolio to follow the previous
        first_date = pd.to_datetime(current_portfolio_values[0]['date_time'])
        date_offset = last_date + timedelta(days=1) - first_date

        # Calculate the scaling factor for the portfolio values
        first_value = current_portfolio_values[0]['value']
        scaling_factor = last_value / first_value if first_value != 0 else 1

        # Calculate the scaling factor for the stock values
        first_stock_value = current_portfolio_values[0]['stock_value']
        stock_scaling_factor = last_stock_value / first_stock_value if first_stock_value != 0 else 1

        # Adjust the portfolio values and dates
        adjusted_portfolio_values = []
        for entry in current_portfolio_values:
            adjusted_value = entry['value'] * scaling_factor
            adjusted_stock_value = entry['stock_value'] * stock_scaling_factor
            adjusted_date = pd.to_datetime(entry['date_time']) + date_offset
            adjusted_portfolio_values.append({
                'date_time': adjusted_date,
                'value': adjusted_value,
                'stock_value': adjusted_stock_value
            })

        # Append the adjusted portfolio values to the compiled list
        compiled_portfolio_values.extend(adjusted_portfolio_values)

        # Adjust and append trades history
        for trade in current_results['trades_history']:
            adjusted_trade = trade.copy()
            adjusted_trade['purchase_date'] += date_offset
            adjusted_trade['sale_date'] += date_offset
            compiled_trades_history.append(adjusted_trade)

        # Update last_value and last_date for the next iteration
        last_value = adjusted_portfolio_values[-1]['value']
        last_date = adjusted_portfolio_values[-1]['date_time']
        last_stock_value = adjusted_portfolio_values[-1]['stock_value']

    # Update compiled results with the new sequential portfolio values
    compiled_results['portfolio_values_over_time'] = compiled_portfolio_values
    compiled_results['trades_history'] = compiled_trades_history

    # Calculate total time passed
    total_time_passed = compiled_portfolio_values[-1]['date_time'] - compiled_portfolio_values[0]['date_time']
    compiled_results['total_time_passed'] = total_time_passed
    
    total_years = total_time_passed / timedelta(days=365.25)  # Average number of days in a year
    compiled_results['total_years'] = total_years
    
    # use the portfolio_values_over_time to calcualte average yearly gain
    starting_value = compiled_portfolio_values[0]['value']
    ending_value = compiled_portfolio_values[-1]['value']

    compiled_results['total_percentage_gain'] = (ending_value / starting_value) - 1
    
    
    compiled_results['average_return_per_year'] = calculate_average_yearly_gain(compiled_portfolio_values)
    
    # get the total amount of money made variable
    total_money_made = 0
    for r in results:
        total_money_made = total_money_made + r['total_amount_of_money_made']
    compiled_results['total_amount_of_money_made'] = total_money_made

    # get the total number of trades
    compiled_results['total_trades'] = len(compiled_trades_history)
    
    # calculate the average time holding any given position
    time_held_list = [trade['time_held'] for trade in compiled_trades_history]
    if time_held_list:
        average_time_holding_position = sum(time_held_list, timedelta()) / len(time_held_list)
        longest_time_position_held = max(time_held_list)
    else:
        average_time_holding_position = timedelta(0)
        longest_time_position_held = timedelta(0)
    compiled_results['average_time_holding_position'] = average_time_holding_position

    # Calculate average number of trades per year
    compiled_results['average_trades_per_year'] = len(compiled_trades_history) / total_years

    return compiled_results


def optimize(strategies, data_frames, loss_function, optimization_method='random', max_evals=10, population_size=10):
    """
    Optimize the given strategies using the given data, loss function, and a chosen method.

    Parameters
    ----------
    strategies : list of dict
        Example:
        [
          {
            'strategy': strategy_function,
            'params': {'param1': value1, 'param2': value2},
            'param_space': {
              'param1': hp.quniform('param1', 10, 30, 1),
              'param2': hp.choice('param2', [10, 15, 20])
            },
            'ga_bounds': {
              'param1': (10, 30),   # (min, max)
              'param2': (10, 20)
            }
          }, ...
        ]

    data_frames : list of pandas.DataFrame
        List of data frames to optimize the strategies on.

    loss_function : function
        Takes in the backtest results (dict) and returns a float loss.

    optimization_method : str
        "random", "hyperopt", or "genetic".

    max_evals : int
        Max evaluations/iterations for the chosen optimization method.

    Returns
    -------
    dict
        {
          'best_loss': float,
          'best_params': dict,
          'best_strategy': function
        }
    """

    best_loss = float('inf')
    best_params = None
    best_strategy = None

    # Helper function to evaluate strategy across data frames
    def evaluate_strategy(strategy, params):
        if len(data_frames) > 1:
            results = []
            for df in data_frames:
                trading_signals = strategy(df['ohlc'], params)
                backtest_results, _ = backtester.run_backtest(trading_signals)
                results.append(backtest_results)
            combined_results = compile_backtest_results_sequential(results, data_frames)
            return loss_function(combined_results)
        else:
            trading_signals = strategy(data_frames[0]['ohlc'], params)
            backtest_results, _ = backtester.run_backtest(trading_signals)
            return loss_function(backtest_results)

    # 1. RANDOM SEARCH
    if optimization_method == 'random':
        
        # Create progress bar for evaluations
        eval_bar = tqdm(range(max_evals), desc="Evaluations")
        
        for eval_num in eval_bar:
            # Try each strategy
            for strategy_dict in strategies:
                strategy = strategy_dict['strategy']
                base_params = strategy_dict.get('params', {})
                bounds = strategy_dict.get('ga_bounds', {})
                
                # Randomly sample parameters within bounds
                current_params = {}
                for param, (min_val, max_val) in bounds.items():
                    # Use uniform random sampling for numerical parameters
                    current_params[param] = random.uniform(min_val, max_val)
                
                # Merge with base params
                current_params.update({k:v for k,v in base_params.items() if k not in current_params})
                
                # Evaluate strategy with current parameters
                loss = evaluate_strategy(strategy, current_params)
                
                # Update best if better
                if loss < best_loss:
                    best_loss = loss
                    best_params = current_params.copy()
                    best_strategy = strategy
                    
                # Update progress bar description with current status
                eval_bar.set_postfix({
                    'Strategy': strategy.__name__,
                    'Loss': f"{loss:.4f}",
                    'Best': f"{best_loss:.4f}"
                })

    # 2. HYPEROPT (Tree-structured Parzen Estimator)
    elif optimization_method == 'hyperopt':
        if not HYPEROPT_INSTALLED:
            raise ImportError(
                "Hyperopt not installed. Install with 'pip3 install hyperopt'."
            )

        for strategy_dict in strategies:
            strategy = strategy_dict['strategy']
            params = strategy_dict['params']
            param_space = strategy_dict['param_space']

            def objective(params):
                current_loss = evaluate_strategy(strategy, params)
                return {'loss': current_loss, 'status': STATUS_OK, 'params': params}

            trials = Trials()
            best_params_for_strategy = fmin(
                fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
            final_loss = evaluate_strategy(strategy, best_params_for_strategy)

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = best_params_for_strategy
                best_strategy = strategy

    # 3. GENETIC ALGORITHM (via DEAP)
    elif optimization_method == 'genetic':
        if not DEAP_INSTALLED:
            raise ImportError("DEAP not installed. Install with 'pip3 install deap'.")

        for strategy_dict in strategies:
            strategy = strategy_dict['strategy']
            params = strategy_dict['params']
            bounds = strategy_dict['ga_bounds']
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            toolbox = base.Toolbox()
            
            # Register parameter generation
            for param_name in bounds:
                min_val, max_val = bounds[param_name]
                toolbox.register(f"attr_{param_name}", 
                                random.uniform, min_val, max_val)
            
            def safe_cxTwoPoint(ind1, ind2):
                if len(ind1) > 1 and len(ind2) > 1:
                    return tools.cxTwoPoint(ind1, ind2)
                else:
                    # Return unchanged individuals if crossover is invalid
                    return ind1, ind2
            
            # Structure initializers
            param_names = list(bounds.keys())
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             [getattr(toolbox, f"attr_{name}") for name in param_names], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                param_dict = dict(zip(param_names, individual))
                return (evaluate_strategy(strategy, param_dict),)
                
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", safe_cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            pop = toolbox.population(n=population_size)
            result, logbook = algorithms.eaSimple(pop, toolbox, 
                                                cxpb=0.7, mutpb=0.3,
                                                ngen=max_evals,
                                                verbose=False)
                                                
            best_ind = tools.selBest(pop, 1)[0]
            final_params = dict(zip(param_names, best_ind))
            final_loss = evaluate_strategy(strategy, final_params)
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = final_params
                best_strategy = strategy
                # Progress monitoring for GA
                def show_progress(gen, num_gen, fits, best):
                    mean = sum(fits) / len(fits)
                    print(f"Gen {gen}/{num_gen} - Best: {min(fits):.4f}, Avg: {mean:.4f}")

                # Add stats tracking
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("min", np.min)
                stats.register("avg", np.mean)

                # Add the progress callback to eaSimple
                result, logbook = algorithms.eaSimple(pop, toolbox,
                                                    cxpb=0.7, mutpb=0.3,
                                                    ngen=max_evals,
                                                    stats=stats,
                                                    verbose=True)
    else:
        raise ValueError("Invalid optimization_method. Use 'random', 'hyperopt', or 'genetic'.")

    return {
        'best_loss': best_loss,
        'best_params': best_params,
        'best_strategy': best_strategy
    }