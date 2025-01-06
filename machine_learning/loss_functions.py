import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def simple_loss_function(backtest_results):
    """
    Simple loss function that computes the total profit/loss and total percentage gain.

    Parameters:
    backtest_results (dict): Dictionary containing the backtest results

    Returns:
    float: Loss value based on the total profit/loss and total percentage gain
    """
    total_profit_loss = backtest_results['total_amount_of_money_made']
    total_percentage_gain = backtest_results['total_percentage_gain']

    # Example loss function: negative of total profit/loss (since we want to maximize profit)
    loss = -total_profit_loss

    return loss

def sharpe_ratio_loss_function(backtest_results):
    """
        Loss function that computes the negative Sharpe ratio.

    Parameters:
    backtest_results (dict): Dictionary containing the backtest results

    Returns:
    float: Loss value based on the negative Sharpe ratio
    """
    returns = backtest_results['trading_signals']['portfolio_value'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    # Assuming a risk-free rate of 0 for simplicity
    sharpe_ratio = mean_return / std_return

    # We want to maximize the Sharpe ratio, so we return its negative value as the loss
    loss = -sharpe_ratio

    return loss

def ridge_regression_loss_function(backtest_results):
    """
    Loss function that uses Ridge Regression to predict future returns and computes the mean squared error.
    """
    trading_signals = backtest_results['trading_signals']
    returns = trading_signals['portfolio_value'].pct_change().dropna().values.reshape(-1, 1)
    
    # Determine the minimum length for consistent slicing
    min_length = len(returns) - 5  # Ensure we have at least 5 lagged values
    X = np.hstack([returns[i:i + min_length] for i in range(5)])  # Create features using lagged returns
    y = returns[5:5 + min_length]  # Ensure `y` matches the length of `X`

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge Regression model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    # Predict and calculate mean squared error
    y_pred = ridge.predict(X_scaled)
    mse = np.mean((y - y_pred) ** 2)

    return mse

def elastic_net_loss_function(backtest_results):
    """
    Loss function that uses Elastic Net regression to predict future returns and computes the mean squared error.
    """
    trading_signals = backtest_results['trading_signals']
    returns = trading_signals['portfolio_value'].pct_change().dropna().values.reshape(-1, 1)
    
    # Determine the minimum valid length to ensure all slices are consistent
    min_length = len(returns) - 5
    if min_length <= 0:
        raise ValueError("Not enough data points to calculate lagged returns for Elastic Net Loss Function")

    # Adjust slicing to create consistent arrays
    X = np.hstack([returns[i:i + min_length] for i in range(5)])  # Features
    y = returns[5:5 + min_length]  # Target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Elastic Net model
    elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.5)
    elastic_net.fit(X_scaled, y)

    # Predict and calculate mean squared error
    y_pred = elastic_net.predict(X_scaled)
    mse = np.mean((y - y_pred) ** 2)

    return mse