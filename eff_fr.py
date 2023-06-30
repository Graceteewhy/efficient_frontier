import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
from typing import List, Tuple
from functools import cache  
from IPython.display import display
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import date
from nsepy import get_history as gh
plt.style.use('fivethirtyeight')
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import  risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
from pandas_datareader.data import DataReader
import copy
import plotly.express as px
import seaborn as sns
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


st.title('Portfolio Optimization')
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \ WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
tickers = tickers_string.split(',')


def load_data(tickers):
    data = yf.download(tickers, period='3y', interval='1d')['Adj Close']
    return data


data_load_state = st.text('Loading data...')
data = load_data(tickers)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Normalized price
data_nor = data.divide(data.iloc[0] / 100)

# Get optimized weights
ef = EfficientFrontier(mu, S)
ef.max_sharpe(risk_free_rate=0.02)
weights = ef.clean_weights()
expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']

# Calculate returns of portfolio with optimized weights
data['Optimized Portfolio'] = 0
for ticker, weight in weights.items():
	data['Optimized Portfolio'] += data[ticker]*weight

# Plot Cumulative Returns of Optimized Portfolio
def plot_cum_returns(data, title):    
    daily_cum_returns = (1 + data).cumprod()*100
    fig = px.line(daily_cum_returns, title=title)
    return fig
fig_cum_returns_optimized = plot_cum_returns(data['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')



# Display everything on Streamlit
st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
st.plotly_chart(fig_cum_returns_optimized)

st.subheader("Optimized Max Sharpe Portfolio Weights")
st.dataframe(weights_df)

st.subheader("Optimized Max Sharpe Portfolio Performance")
st.image(fig_efficient_frontier)

st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))

st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
st.plotly_chart(fig_price)
st.plotly_chart(fig_cum_returns)

    
