# Overcoming Overfitting

This repository contains the codebase for the dissertation "Overcoming Overfitting" by Alexander P. Sheppert. The dissertation explores various strategies and methodologies to mitigate overfitting in financial trading algorithms.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Strategies](#strategies)
- [Backtesting](#backtesting)
- [Live Simulation](#live-simulation)
- [Optimization](#optimization)
- [Loss Functions](#loss-functions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project aims to develop and test different trading strategies while addressing the issue of overfitting. It includes various modules for data collection, strategy implementation, backtesting, live simulation, and optimization.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/teancum/dissertation.git
cd dissertation
pip install -r requirements.txt
```

## Usage

### Fetching Data

To fetch historical data for a specific ticker:

```python
from data_collection.fetch_data import fetch_historical_data

df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
```

### Running Backtests

To run a backtest on a specific strategy:

```python
from modules.backtester import run_backtest
from strategies.MACD_Strategy import strategy

trading_signals = strategy(df, params={'short_window': 12, 'long_window': 26, 'signal_window': 9})
backtest_results, trading_signals_with_portfolio = run_backtest(trading_signals)
```

### Running Live Simulations

To run a live simulation:

```python
from modules.live_sim_backtest import run_live_simulation
from strategies.MACD_Strategy import should_buy_live

live_sim_results, live_trading_signals_with_portfolio = run_live_simulation(should_buy_live, df, params={'short_window': 12, 'long_window': 26, 'signal_window': 9})
```

## Strategies

The repository includes several trading strategies, such as:

- Exponential Moving Average (EMA) Strategy
- Moving Average Convergence Divergence (MACD) Strategy
- Parabolic SAR Strategy
- Simple Moving Average (SMA) Strategy
- Small Moving Average Crossover Strategy

Each strategy is implemented in its respective file under the `strategies` directory.

## Backtesting

The backtesting module allows you to evaluate the performance of trading strategies on historical data. It calculates various metrics such as total trades, final cash, total money made, and total percentage gain.

## Live Simulation

The live simulation module simulates real-time trading by applying the strategy to historical data as if it were live. It records trades, calculates portfolio value, and provides detailed trade history.

## Optimization

The optimization module uses different techniques to find the best parameters for the trading strategies. Supported optimization methods include:

- Random Search
- Hyperopt (Tree-structured Parzen Estimator)
- Genetic Algorithm (via DEAP)

## Loss Functions

Several loss functions are provided to evaluate the performance of trading strategies during optimization:

- Simple Loss Function
- Sharpe Ratio Loss Function
- Ridge Regression Loss Function
- Elastic Net Loss Function
- GT Function

## Results

The results of backtests and live simulations are saved in the `output` directory. The `make_output.py` script generates detailed reports and visualizations of the results.
