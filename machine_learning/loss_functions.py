import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from scipy.stats import linregress
import math

def simple_loss_function(backtest_results):
    total_profit_loss = backtest_results['total_amount_of_money_made']
    return -total_profit_loss

def sharpe_ratio_loss_function(backtest_results):
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        return 0.0

    returns = np.diff(values) / values[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0.0

    return -sharpe_ratio

def ridge_regression_loss_function(backtest_results, objective='profit'):
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        raise ValueError("Not enough data to compute returns for Ridge regression.")

    returns = np.diff(values) / values[:-1]
    min_length = len(returns) - 5
    if min_length <= 0:
        raise ValueError("Not enough data to create lagged features for Ridge regression.")
    X = np.hstack([returns[i : i + min_length].reshape(-1, 1) for i in range(5)])
    y = returns[5 : 5 + min_length]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    if objective == 'mse':
        mse = np.mean((y - y_pred) ** 2)
        return mse
    elif objective == 'profit':
        profit = np.sum(np.sign(y_pred) * y)
        return -profit
    else:
        raise ValueError(f"Unknown objective '{objective}'")

def elastic_net_loss_function(backtest_results, objective='profit'):
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        raise ValueError("Not enough data to compute returns for Elastic Net regression.")

    returns = np.diff(values) / values[:-1]
    min_length = len(returns) - 5
    if min_length <= 0:
        raise ValueError("Not enough data to create lagged features for Elastic Net regression.")
    X = np.hstack([returns[i : i + min_length].reshape(-1, 1) for i in range(5)])
    y = returns[5 : 5 + min_length]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    en = ElasticNet(alpha=0.5, l1_ratio=0.5)
    en.fit(X_scaled, y)

    y_pred = en.predict(X_scaled)

    if objective == 'mse':
        mse = np.mean((y - y_pred) ** 2)
        return mse
    elif objective == 'profit':
        profit = np.sum(np.sign(y_pred) * y)
        return -profit
    else:
        raise ValueError(f"Unknown objective '{objective}'")

def find_stabilized_variance(data, min_period=20, max_period=100):
# \[
# \text{Stabilized Variance Period } (n^*) = 
# \begin{cases} 
# n, & \text{if } \Delta \sigma_n \leq \epsilon \text{ for recent periods} \\
# 50, & \text{if no stabilization is observed for } n \in [n_{\min}, n_{\max}]
# \end{cases}
# \]

# \text{where } \Delta \sigma_n = \frac{1}{k} \sum_{i=1}^{k} |\sigma_{n_i} - \sigma_{n_{i-1}}|, \quad k \geq 3
# \]

# \[
# \sigma_{n_i} = \frac{1}{m} \sum_{j=1}^{m} \left( r_{j} - \mu \right)^2, \quad \mu = \frac{1}{m} \sum_{j=1}^{m} r_{j}
# \]

# \[
# r_{j} = \frac{v_{j, \text{end}} - v_{j, \text{start}}}{v_{j, \text{start}}}
# \]

# \[
# n^* = \text{optimal number of periods where variance stabilizes.}
# \]
# Explanation of Terms

#     n∗n∗: The optimal number of periods where the variance of returns stabilizes. This is the final output of the method.
#     nn: The number of periods currently under evaluation.
#     ΔσnΔσn​: The average change in variance between consecutive periods over the last kk periods. It is used to assess stabilization.
#     ϵϵ: A small stabilization threshold (e.g., 1% of the mean variance), indicating that the variance changes have plateaued.
#     nmin⁡nmin​: The minimum number of periods to consider (e.g., user-defined or based on data constraints).
#     nmax⁡nmax​: The maximum number of periods to consider.
#     σniσni​​: The variance of returns within period nini​, calculated as the mean squared deviation of returns from the mean return (μμ).
#     μμ: The mean return for the period, defined as the average of individual returns.
#     rjrj​: The return for a specific sub-period jj, defined as the relative change in value from the start (vj,startvj,start​) to the end (vj,endvj,end​) of the period.
#     kk: The number of recent periods used to calculate the average change in variance.
#     mm: The total number of returns within a period.

    """
    Find the number of periods where the variance of returns stabilizes naturally.
    Returns 50 if stabilization is not found within a given range.

    Parameters:
    - data: List of dictionaries with 'date_time' (Timestamp) and 'value' (float).
    - min_period: Minimum number of periods to start with.
    - max_period: Maximum number of periods to check.

    Returns:
    - Optimal number of periods where variance stabilizes, or 50 if not found.
    """
    df = pd.DataFrame(data)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.sort_values('date_time', inplace=True)

    total_time = (df['date_time'].max() - df['date_time'].min()).days
    results = []
    
    # Start binary-like search to optimize the number of periods
    low = min_period
    high = max_period
    while low <= high:
        num_periods = (low + high) // 2
        period_length = total_time / num_periods
        df['period'] = ((df['date_time'] - df['date_time'].min()).dt.days // period_length).astype(int)

        # Compute returns per period
        returns = df.groupby('period')['value'].apply(
            lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None
        ).dropna()

        if len(returns) < 2:
            low = num_periods + 1  # Not enough data for this split, go larger
            continue

        variance = np.var(returns)
        results.append((num_periods, variance))

        # Check stabilization by comparing recent variances
        if len(results) > 3:
            recent_variances = [v[1] for v in results[-4:]]  # Last 4 variances
            changes = [abs(recent_variances[i] - recent_variances[i - 1]) for i in range(1, len(recent_variances))]
            avg_change = np.mean(changes)
            
            # If the variance change has plateaued, return num_periods
            if avg_change <= np.mean(recent_variances) * 0.01:  # 1% of the average variance
                return num_periods

        # Adjust search range
        if len(results) > 1 and variance < results[-2][1]:
            high = num_periods - 1  # Variance decreasing, search smaller periods
        else:
            low = num_periods + 1  # Variance increasing, search larger periods

    # If stabilization is not found, return 50 as the default value
    return 50


def get_period_returns(data, num_periods):
    df = pd.DataFrame(data)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.sort_values('date_time', inplace=True)

    # Calculate total days and period length in days
    total_time = (df['date_time'].max() - df['date_time'].min()).days
    period_length = total_time / num_periods
    
    # Assign each row to a period based on integer division
    df['period'] = ((df['date_time'] - df['date_time'].min()).dt.days // period_length).astype(int)

    # Compute portfolio returns
    portfolio_returns = (
        df.groupby('period')['value']
        .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None)
        .dropna()
        .tolist()
    )

    # Compute market (stock) returns
    market_returns = (
        df.groupby('period')['stock_value']
        .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None)
        .dropna()
        .tolist()
    )

    return portfolio_returns, market_returns

def gt_function(backtest_results, stabilize=False, t_or_p="trades"):
    """
    Calculates the GTScore (Golden-Ticket Score).

    Parameters:
    - period_percentage_returns (list): Percentage changes in total portfolio value per period.
    - percentage_returns_by_trade (list): Percentage returns of individual trades.
    - period_percentage_returns_market (list): Percentage returns of a buy-and-hold market strategy per period.
    - num_trades (int): Total number of trades completed.
    - num_periods (int): Total number of periods in the dataset.

    Returns:
    - float: GTScore value.
    """
    
    # Get the number of periods
    if stabilize:
        num_periods = find_stabilized_variance(backtest_results['portfolio_values_over_time'])
    else:
        num_periods = 50
    
    # Get the number of trades
    num_trades = len(backtest_results["trades_history"])
    
    num_trades
    if num_trades <= num_periods:
        # need to push the score towards more trades
        # best score possible with under periods is 100
        interval = (999 - 100) / num_periods
        return 999 - (num_trades*interval)  # GTScore is not valid if num_trades <= num_periods

    
    # Get the returns from each period
    percentage_returns_by_trade = []
    for trade in backtest_results["trades_history"]:
            percentage_returns_by_trade.append(trade["profit_loss_percent"]) 
    
    if t_or_p == "portfolio_value":
        period_percentage_returns, period_percentage_returns_market = get_period_returns(backtest_results['portfolio_values_over_time'], num_periods)
    elif t_or_p == "trades":
        period_percentage_returns = percentage_returns_by_trade
        period_percentage_returns_market = []
        #just make a list of the actual mean return based on same num trades
        starting_market_value = backtest_results["portfolio_values_over_time"][0]["stock_value"]
        ending_market_value = backtest_results["portfolio_values_over_time"][-1]["stock_value"]
        the_mum = (ending_market_value / starting_market_value) ** (1 / num_trades) - 1
        for i in range(0,num_trades):
            period_percentage_returns_market.append(the_mum)
    
    # Calculate necessary values
    mu = np.mean(period_percentage_returns)
    mum = np.mean(period_percentage_returns_market)
    r2 = linregress(range(len(percentage_returns_by_trade)), percentage_returns_by_trade).rvalue ** 2

    negative_returns = [r for r in period_percentage_returns if r < 0]
    sigma_d = np.std(negative_returns) if negative_returns else 1e-6  # Avoid division by zero

    sigma = np.std(period_percentage_returns)

    # Calculate z and GT
    z = (mu - mum) / (sigma / np.sqrt(num_trades))
    
    # print(f"market return: {mum}")
    # print(f"strategy return: {mu}")
    # starting_market_value = backtest_results["portfolio_values_over_time"][0]["stock_value"]
    # ending_market_value = backtest_results["portfolio_values_over_time"][-1]["stock_value"]
    # print(f"total market change {starting_market_value} -> {ending_market_value}")
    # print(f'starting_portfolio_value {backtest_results["portfolio_values_over_time"][0]["value"]} -> {backtest_results["portfolio_values_over_time"][-1]["value"]}')
    # print(f"total number of trades {num_trades}")
    # print(f"total num periods {num_periods}")
    # print('')
    
    # z = abs(z)
    if z <= 0:
        # worse than buy and hold, but we still need it to have something to optimize toward
        score = 100 + (100 * (1 - math.exp(-abs(z - 1))))
        return score  # Logarithm not defined for z <= 1
    elif z <= 1:
        score = 100 * (1 - math.exp(-abs(z - 1)))
        return score  # Logarithm not defined for z <= 1

    ln_z = math.log(z)
    gt_score = (mu * ln_z * r2) / sigma_d
    
    # take the inverse because we want to minimize
    gt_score = -gt_score

    return gt_score
