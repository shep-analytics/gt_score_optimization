def compile_backtest_results(results_original,data_frames):
    """
    Combine multiple backtest results (dicts) into one dictionary.
    """
    results = copy.deepcopy(results_original)
    
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
    
    compiled_results['average_hold_time'] = (sum([trade['time_held'] for trade in compiled_results['trades_history']], timedelta(0)))  / len(compiled_results)
    
    return compiled_results