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
from datetime import datetime
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pydot
import graphviz
from keras.models import load_model
from keras.utils.vis_utils import plot_model 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

st.write("""
# PORTFOLIO OPTIMIZATION
Portfolio selecttion
""")

tickers = st.sidebar.text_input('Enter all stock tickers to be included in portfolio separated by commas \
 WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
tickers = tickers.split(',')


    

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

data = yf.download(tickers, START, TODAY)['Close']
data=data.dropna(axis=1,how='all')


st.subheader('Stock raw data')
st.write(data.tail())



# calculating expected annual return and annualized sample covariance matrix of daily assets returns

mean = expected_returns.mean_historical_return(data)

S = risk_models.sample_cov(data)

st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.figure(figsize=(15, 6))
for i in range(data.shape[1]):
    plt.plot(data.iloc[:,i], label=data.columns.values[i])
plt.legend(loc='upper left', fontsize=12)
plt.title('Stock close price trend', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Price in $')
st.pyplot()


# Normalized price
data_nor = data.divide(data.iloc[0] / 100)

fig = plt.figure(figsize=(15, 6))
for i in range(data_nor.shape[1]):
    plt.plot(data_nor.iloc[:,i], label=data_nor.columns.values[i])
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Normalized prices($)')
plt.title('Stock Normalized Prices Trends', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Price in $')
st.pyplot()


plt.style.use('ggplot')
fig = plt.figure()
sb.heatmap(S,xticklabels=S.columns, yticklabels=S.columns,
cmap='RdBu_r', annot=True, linewidth=0.5)
st.subheader('Covariance of stocks in your portfolio')
st.pyplot()

ef = EfficientFrontier(mean,S)
weights = ef.max_sharpe() #for maximizing the Sharpe ratio #Optimization
cleaned_weights = ef.clean_weights() #to clean the raw weights
# Get the Keys and store them in a list
labels = list(cleaned_weights.keys())
# Get the Values and store them in a list
values = list(cleaned_weights.values())
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.0f%%')
st.subheader('Portfolio Allocation')
st.pyplot()


rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11 
TRADING_DAYS_PER_YEAR = 250


# Needed for type hinting
class Asset:
  pass


def get_log_period_returns(price_history: pd.DataFrame):
  close = price_history['Close'].values  
  return np.log(close[1:] / close[:-1]).reshape(-1, 1)


# daily_price_history has to at least have a column, called 'Close'
class Asset:
  def __init__(self, name: str, daily_price_history: pd.DataFrame):
    self.name = name
    self.daily_returns = get_log_period_returns(daily_price_history)
    self.expected_daily_return = np.mean(self.daily_returns)
  
  @property
  def expected_return(self):
    return TRADING_DAYS_PER_YEAR * self.expected_daily_return

  def __repr__(self):
    return f'<Asset name={self.name}, expected return={self.expected_return}>'

  @staticmethod
  @cache
  def covariance_matrix(assets: Tuple[Asset]):  # tuple for hashing in the cache
    product_expectation = np.zeros((len(assets), len(assets)))
    for i in range(len(assets)):
      for j in range(len(assets)):
        if i == j:
          product_expectation[i][j] = np.mean(assets[i].daily_returns * assets[j].daily_returns)
        else:
          product_expectation[i][j] = np.mean(assets[i].daily_returns @ assets[j].daily_returns.T)
    
    product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

    expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
    product_of_expectations = expected_returns @ expected_returns.T

    return product_expectation - product_of_expectations


def random_weights(weight_count):
    weights = np.random.random((weight_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)


class Portfolio:
  def __init__(self, assets: Tuple[Asset]):
    self.assets = assets
    self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
    self.covariance_matrix = Asset.covariance_matrix(assets)
    self.weights = random_weights(len(assets))
    
  def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float):
    res = minimize(
      lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
      random_weights(self.weights.size),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
      ],
      bounds=[(0., 1.) for i in range(self.weights.size)]
    )

    assert res.success, f'Optimization failed: {res.message}'
    self.weights = res.x.reshape(-1, 1)
  
  def optimize_with_risk_tolerance(self, risk_tolerance: float):
    assert risk_tolerance >= 0.
    return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)
  
  def optimize_with_expected_return(self, expected_portfolio_return: float):
    res = minimize(
      lambda w: self._variance(w),
      random_weights(self.weights.size),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return},
      ],
      bounds=[(0., 1.) for i in range(self.weights.size)]
    )

    assert res.success, f'Optimization failed: {res.message}'
    self.weights = res.x.reshape(-1, 1)

  def optimize_sharpe_ratio(self):
    # Maximize Sharpe ratio = minimize minus Sharpe ratio
    res = minimize(
      lambda w: -(self._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
      random_weights(self.weights.size),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
      ],
      bounds=[(0., 1.) for i in range(self.weights.size)]
    )

    assert res.success, f'Optimization failed: {res.message}'
    self.weights = res.x.reshape(-1, 1)

  def _expected_return(self, w):
    return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]
  
  def _variance(self, w):
    return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

  @property
  def expected_return(self):
    return self._expected_return(self.weights)
  
  @property
  def variance(self):
    return self._variance(self.weights)

  def __repr__(self):
    return f'<Portfolio assets={[asset.name for asset in self.assets]}, expected return={self.expected_return}, variance={self.variance}>'


def yf_retrieve_data(tickers: List[str]):
  dataframes = []

  for ticker_name in tickers:
    ticker = yf.Ticker(ticker_name)
    history = ticker.history(period='10y')

    if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
      history = history.iloc[1:]
  
    assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
    dataframes.append(history)
  
  return dataframes


daily_dataframes = yf_retrieve_data(tickers)
assets = tuple([Asset(name, daily_df) for name, daily_df in zip(tickers, daily_dataframes)])

X = []
y = []

# Drawing random portfolios
for i in range(3000):
  portfolio = Portfolio(assets)
  X.append(np.sqrt(portfolio.variance))
  y.append(portfolio.expected_return)

plt.scatter(X, y, label='Random portfolios')

# Drawing the efficient frontier
X = []
y = []
for rt in np.linspace(-300, 200, 1000):
  portfolio.unsafe_optimize_with_risk_tolerance(rt)
  X.append(np.sqrt(portfolio.variance))
  y.append(portfolio.expected_return)

plt.plot(X, y, 'k', linewidth=3, label='Efficient frontier')

# Drawing optimized portfolios
portfolio.optimize_with_risk_tolerance(0)
plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'm+', markeredgewidth=5, markersize=20, label='optimize_with_risk_tolerance(0)')

portfolio.optimize_with_risk_tolerance(20)
plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'r+', markeredgewidth=5, markersize=20, label='optimize_with_risk_tolerance(20)')

portfolio.optimize_with_expected_return(0.25)
plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'g+', markeredgewidth=5, markersize=20, label='optimize_with_expected_return(0.25)')

portfolio.optimize_sharpe_ratio()
plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'y+', markeredgewidth=5, markersize=20, label='optimize_sharpe_ratio()')

plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio standard deviation')
plt.ylabel('Portfolio expected (logarithmic) return')
plt.legend(loc='lower right')
st.subheader('Efficient frontier')
st.pyplot()

st.write(dict(cleaned_weights))

st.write(ef.portfolio_performance(verbose=True))


latest_prices = get_latest_prices(data)


total_portfolio_value = st.sidebar.slider('Amount to invest', 100,10000)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value)

allocation, leftover = da.greedy_portfolio()
st.write("Discrete allocation:", allocation)
st.write("Funds remaining: ${:.2f}".format(leftover))

st.subheader('Hierarchical Risk Parity (HRP)')
returns = data.pct_change().dropna()
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
hrp.portfolio_performance(verbose=True)
st.write(dict(hrp_weights))

da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value)

allocation, leftover = da_hrp.greedy_portfolio()
st.write("Discrete allocation (HRP):", allocation)
st.write("Funds remaining (HRP): ${:.2f}".format(leftover))

st.subheader('Mean Conditional Value at Risk (mCVAR)')
S = data.cov()
ef_cvar = EfficientCVaR(mean, S)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
st.write(dict(cleaned_weights))

da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value)

allocation, leftover = da_cvar.greedy_portfolio()
st.write("Discrete allocation (CVAR):", allocation)
st.write("Funds remaining (CVAR): ${:.2f}".format(leftover))



st.title('Stock Price Prediction')

stock = st.sidebar.text_input("Enter the stock ticker and hit Enter")
stock = stock.upper()


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#@st.cache
def load_data(stock):
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(stock)
data_load_state.text('Loading data... done!')

st.subheader('Stock raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()



# LSTM

#Set Target Variable
output_var = pd.DataFrame(data['Close'])
#Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(data[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=data.index)
feature_transform.head()

#Creating a Training Set and a Test Set for Stock Market Prediction

#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)

# Show and plot forecast
#st.subheader('Predicted Price')
#st.write("Discrete allocation (CVAR):", allocation)
st.write('Predicted Price', y_pred[-1])

