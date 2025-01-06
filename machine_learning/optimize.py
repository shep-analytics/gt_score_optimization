# optimize.py

import random
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm


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


def compile_backtest_results(results,data_frames):
    """
    Combine multiple backtest results (dicts) into one dictionary.
    """
    compiled_results = results[0]
    for i in range(1, len(results)):
        for key in compiled_results:
            if isinstance(compiled_results[key], list):
                compiled_results[key].extend(results[i][key])
            elif isinstance(compiled_results[key], pd.DataFrame):
                compiled_results[key] = pd.concat(
                    [compiled_results[key], results[i][key]],
                    ignore_index=True
                )
            else:
                compiled_results[key] += results[i][key]
    
    # Calculate total time passed
    total_time_passed = sum([df['time_passed'] for df in data_frames], pd.Timedelta(0))
    compiled_results['total_time_passed'] = total_time_passed

    # Calculate average return per year
    total_years = total_time_passed.days / 365.25
    compiled_results['average_return_per_year'] = compiled_results['total_percentage_gain'] / total_years

    # Calculate average number of trades per year
    compiled_results['average_trades_per_year'] = compiled_results['total_trades'] / total_years
    
    return compiled_results


def optimize(strategies, data_frames, loss_function, optimization_method='random', max_evals=20, population_size=20):
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
            combined_results = compile_backtest_results(results, data_frames)
            return loss_function(combined_results)
        else:
            trading_signals = strategy(data_frames[0], params)
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
            
            # Structure initializers
            param_names = list(bounds.keys())
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             [getattr(toolbox, f"attr_{name}") for name in param_names], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                param_dict = dict(zip(param_names, individual))
                return (evaluate_strategy(strategy, param_dict),)
                
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
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