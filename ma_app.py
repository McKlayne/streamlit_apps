# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
# IMPORTING PACKAGES
import streamlit as st
import os
import alpaca_trade_api as tradeapi
import pandas as pd 
import matplotlib.pyplot as plt 
import requests
import math
from termcolor import colored as cl 
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

# EXTRACTING DATA
def get_ticker_data(symbol,days):
    iex_api_key = 'Tsk_30a2677082d54c7b8697675d84baf94b'
    api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/max?token={iex_api_key}'
    data = pd.DataFrame.from_dict(requests.get(api_url).json())[['date','open','high','low','close','volume']].tail(days).reset_index(drop=True)
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index)
    return data

def calc_moving_average(data, symbol, windows):
    for i in range(len(windows)):
        data[f'{window[i]} day MA'] = data.close.rolling(window = windows[i]).mean()
    
    if len(windows) > 1:
        data = data[data[f'{window[1]} day MA'] > 0]
    else:
        data = data[data[f'{window[0]} day MA'] > 0]

    return data

def ma_backtest(data, window, strategy='single', sellShort=False, slippage = 0.003):
    '''
    data is a df that contains the closing price of the stock and moving averages
    window can be a single value for price crossover or a list for moving average crossover
    crossover equals price or ma to determine which strategy should be use
    '''

    # a long/short ma strategy
    
    # define the trades
    # does this dataframe have dates?
    # use the sellShort parameter to determine if a sell signal will result in a short position or just closing out position
    if sellShort:
        sellSignal = -1
    else:
        sellSignal = 0
        
    if strategy == 'Single Moving Average':
        data['Side'] = data.apply(lambda row: 1 if row.close >= row[f'{window[0]} day MA'] else sellSignal, axis=1)
    elif strategy == 'Crossover Moving Average':
        data['Side'] = data.apply(lambda row: 1 if row[f'{window[0]} day MA'] >= row[f'{window[1]} day MA'] else sellSignal, axis=1)
        
    # calculate returns with slippage
    data['LagPrice'] = data['close'].shift(1)
    data['PctChange'] = (data['close'] - data['LagPrice']) / data['LagPrice']
    
        # loop through and define all days where the position swithched from long to short or short to long
        
    buyPrice = []
    sellPrice = []
    
    for i in range(len(data.close)):
        if i < 1:
            if data.Side[i] < 0:
                sellPrice.append(data.close[i])
                buyPrice.append(np.nan)
            elif data.Side[i] == 0:
                buyPrice.append(np.nan)
                sellPrice.append(np.nan)
            else:
                buyPrice.append(data.close[i])
                sellPrice.append(np.nan)
        elif data['Side'][i] > data['Side'][i-1]:
            buyPrice.append(data.close[i])
            sellPrice.append(np.nan)
        elif data['Side'][i] < data['Side'][i-1]:
            sellPrice.append(data.close[i])
            buyPrice.append(np.nan)
        else:
            buyPrice.append(np.nan)
            sellPrice.append(np.nan)

    data['buyPrice'] = buyPrice
    data['sellPrice'] = sellPrice
    data['Slippage'] = ((data.buyPrice.fillna(0) + data.sellPrice.fillna(0)) * slippage) / data.close
    data['Return'] = data.Side * data.PctChange - data.Slippage
    data['Return'][0] = -data.Slippage[0]  
    data['Cumulative'] = data.Return.cumsum()

    return data

def plot(data,windows,strategy):
    plt.plot(data['close'], alpha = 0.3, label = ticker)
    
    if len(windows) > 1:
        plt.plot(data[f'{window[0]} day MA'], alpha = 0.6, label = f'{window[0]} day MA')
        plt.plot(data[f'{window[1]} day MA'], alpha = 0.6, label = f'{window[1]} day MA')
    else:
        plt.plot(data[f'{window[0]} day MA'], alpha = 0.6, label = f'{window[0]} day MA')
        
    plt.scatter(data.index, data.buyPrice, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(data.index, data.sellPrice, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')

    plt.title(f'{ticker} {strategy} Trading Signals')
    return plot

st.title("Moving Average Analysis")
ticker = st.sidebar.text_input("Please enter a ticker symbol","SPY")
days = st.sidebar.number_input("Please enter the number of days of data you would like",31)
strategy = st.sidebar.radio('Select Strategy', ['Single Moving Average','Moving Average Crossover'])
if strategy == 'Single Moving Average':
    single_ma = st.sidebar.number_input("Please enter your moving average window",5)
    single_ma = [single_ma]
else:
    lma = st.sidebar.number_input("Please enter your long MA window",30)
    sma = st.sidebar.number_input("Please enter your short MA window",5)

enable_short = st.sidebar.radio('Enable Short Selling',['Yes','No'])

#get data
data = get_ticker_data(ticker, days)

if strategy == 'Single Moving Average':
    strategy = 'Single Moving Average'
    window = single_ma
else:
    strategy = 'Crossover Moving Average'
    window = [sma,lma]

if enable_short == 'Yes':
    sellShort = True
else:
    sellShort = False
data = calc_moving_average(data, ticker, window)
sma_trade = ma_backtest(data,window,strategy,sellShort)
plot(sma_trade, window, strategy)
st.pyplot()
st.write('Percent return on this strategy would have been {:.2%}'.format(float(sma_trade.Cumulative.tail(1))))
st.write('Percent return on buy and hold would have been {:.2%}'.format((float(sma_trade.close.tail(1))-float(sma_trade.close[0]))/float(sma_trade.close[0])))
st.dataframe(data)



