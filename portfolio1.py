import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Alpha Vantage API setup
API_KEY = 'IV7AM6UL2FN5EZS3'  # Replace with your actual API key
BASE_URL = 'https://www.alphavantage.co/query'

# Stock symbols
symbols = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'META']


def get_intraday_data(symbol, interval='1min', outputsize='compact'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': API_KEY,
        'outputsize': outputsize,
        'datatype': 'json'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    time_series = data.get(f'Time Series ({interval})', {})

    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={'4. close': symbol})
    df[symbol] = df[symbol].astype(float)
    df.index = pd.to_datetime(df.index)
    return df[[symbol]]


def get_combined_data(symbols):
    price_data = []
    for sym in symbols:
        print(f"Fetching data for {sym}...")
        df = get_intraday_data(sym)
        price_data.append(df)
    combined = pd.concat(price_data, axis=1).sort_index()
    combined = combined.dropna()
    return combined


# Get price data
prices = get_combined_data(symbols)

# Compute log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Initial weights
initial_weights = np.array([0.2] * len(symbols))


# Portfolio statistics
def portfolio_performance(weights, returns):
    port_return = np.sum(weights * returns.mean()) * 252 * 390  # Annualize 1-min returns
    port_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252 * 390, weights)))
    sharpe_ratio = port_return / port_std
    return port_return, port_std, sharpe_ratio


# Objective function: negative Sharpe Ratio
def negative_sharpe(weights, returns):
    return -portfolio_performance(weights, returns)[2]


# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Bounds for weights: between 0 and 1
bounds = tuple((0, 1) for _ in range(len(symbols)))

# Optimization
optimized = minimize(negative_sharpe,
                     initial_weights,
                     args=(returns,),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

# Results
opt_weights = optimized.x
ret, std, sharpe = portfolio_performance(opt_weights, returns)

print("\n📈 Optimized Portfolio Weights (Max Sharpe Ratio):")
for sym, w in zip(symbols, opt_weights):
    print(f"{sym}: {w:.4f}")
print(f"\nExpected Annual Return: {ret:.2%}")
print(f"Volatility (Std Dev): {std:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")