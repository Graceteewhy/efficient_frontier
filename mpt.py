import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def calculate_efficient_frontier(tickers, start_date, end_date, risk_free_rate):
    # Fetch historical price data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate log returns
    returns = np.log(data / data.shift(1))
    
    num_assets = len(tickers)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Create empty lists for storing results
    port_returns = []
    port_volatility = []
    weights_list = []
    
    # Perform Monte Carlo simulation to generate portfolio weights
    for _ in range(1000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        returns = np.dot(weights, mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        port_returns.append(returns)
        port_volatility.append(volatility)
        weights_list.append(weights)
    
    # Convert lists to arrays
    port_returns = np.array(port_returns)
    port_volatility = np.array(port_volatility)
    weights_list = np.array(weights_list)
    
    # Calculate the Sharpe ratio for each portfolio
    sharpe_ratio = (port_returns - risk_free_rate) / port_volatility
    
    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratio)
    max_sharpe_return = port_returns[max_sharpe_idx]
    max_sharpe_volatility = port_volatility[max_sharpe_idx]
    max_sharpe_weights = weights_list[max_sharpe_idx]
    
    return port_returns, port_volatility, weights_list, max_sharpe_return, max_sharpe_volatility, max_sharpe_weights
def main():
    st.title("Efficient Frontier")

    # Create input for tickers, date range, and risk-free rate
    st.subheader("Asset Selection")
    tickers_input = st.text_input(
        "Enter ticker symbols (comma-separated)",
        "AAPL, MSFT, GOOGL"
    )
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    risk_free_rate = st.number_input

