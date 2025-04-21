# 📊 Intraday Portfolio Optimization with Alpha Vantage API

This project implements an intraday portfolio optimization model using Python. It fetches 1-minute interval price data for selected stocks using the Alpha Vantage API, calculates log returns, and performs **mean-variance optimization** to maximize the **Sharpe Ratio** of the portfolio.

---

## 🚀 Features

- Fetches real-time intraday data for selected stock symbols (`AAPL`, `TSLA`, `AMZN`, `GOOGL`, `META`)
- Computes log returns from 1-minute price data
- Applies portfolio optimization using **Scipy's SLSQP** method
- Maximizes **Sharpe Ratio** under constraints (weights sum to 1, bounded between 0 and 1)
- Outputs:
  - Optimized asset weights
  - Expected annual return
  - Portfolio volatility
  - Sharpe Ratio

---

## 🛠️ Requirements

Install the required Python libraries:

```bash
pip install requests pandas numpy scipy
