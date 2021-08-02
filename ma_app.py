#import packages
import streamlit as st
import os
import alpaca_trade_api as tradeapi
import pandas as pd 
import matplotlib.pyplot as plt 
import requests
import math
from termcolor import colored as cl 
import numpy as np

#set fixed parameters
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
api = tradeapi.REST('PKCTTJV8JJ6LU5SU4HVV','doBeh0aSOWNN4rkqCT6uP6z6M5SEZHw76TAA85JR', api_version='v2')

#grab market data using alpaca
def get_ticker_data(ticker, api, days):
    data = api.get_barset(ticker, 'day', limit=days)[ticker].df.reset_index()
    data = data.set_index('index')
    data.index = pd.to_datetime(data.index)
    return(data)

#calculate the moving averages for both strategies
def calc_moving_average(data, symbol, windows):
    for i in range(len(windows)):
        data[f'{window[i]} day MA'] = data.close.rolling(window = windows[i]).mean()
    
    if len(windows) > 1:
        data = data[data[f'{window[1]} day MA'] > 0]
    else:
        data = data[data[f'{window[0]} day MA'] > 0]

    return data

#backtest for both strategies
def ma_backtest(data, window, strategy='single', sellShort=False, slippage = 0.003):
    '''
    data is a df that contains the closing price of the stock and moving averages
    window can be a single value for price crossover or a list for moving average crossover
    crossover equals price or ma to determine which strategy should be use
    '''
    #catch the enabling of short selling at the beginning
    if sellShort:
        sellSignal = -1
    else:
        sellSignal = 0
    
    #vectorized backtests by strategy
    if strategy == 'Single Moving Average':
        data['Side'] = data.apply(lambda row: 1 if row.close >= row[f'{window[0]} day MA'] else sellSignal, axis=1)
    elif strategy == 'Crossover Moving Average':
        data['Side'] = data.apply(lambda row: 1 if row[f'{window[0]} day MA'] >= row[f'{window[1]} day MA'] else sellSignal, axis=1)

    #metrics for calculating return   
    data['LagPrice'] = data['close'].shift(1)
    data['PctChange'] = (data['close'] - data['LagPrice']) / data['LagPrice']

    #variables to capture the buy and sell prices  
    buyPrice = []
    sellPrice = []
    
    #Logic for noting each buy and sell by strategy/short selling included
    for i in range(len(data.close)):
        if data['Side'][i] > data['Side'][i-1]:
            buyPrice.append(data.close[i])
            sellPrice.append(np.nan)
        elif data['Side'][i] < data['Side'][i-1]:
            sellPrice.append(data.close[i])
            buyPrice.append(np.nan)
        else:
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
            else:
                buyPrice.append(np.nan)
                sellPrice.append(np.nan)

    #putting it all together
    data['buyPrice'] = buyPrice
    data['sellPrice'] = sellPrice
    data['Slippage'] = ((data.buyPrice.fillna(0) + data.sellPrice.fillna(0)) * slippage) / data.close
    data['Return'] = data.Side * data.PctChange - data.Slippage
    data['Return'][0] = -data.Slippage[0]  
    data['Cumulative'] = data.Return.cumsum()

    return data

#tell the story with some visuals
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

#streamlit app
st.title("Moving Average Analysis")

#inputs from user
ticker = st.sidebar.text_input("Please enter a ticker symbol","SPY")
days = st.sidebar.number_input("Please enter the number of days of data you would like",30)
strategy = st.sidebar.radio('Select Strategy', ['Single Moving Average','Moving Average Crossover'])

#filter moving average windows by strategy
if strategy == 'Single Moving Average':
    single_ma = st.sidebar.number_input("Please enter your moving average window",5)
    single_ma = [single_ma]
    strategy = 'Single Moving Average'
    window = single_ma
else:
    lma = st.sidebar.number_input("Please enter your long MA window",30)
    sma = st.sidebar.number_input("Please enter your short MA window",5)
    strategy = 'Crossover Moving Average'
    window = [sma,lma]

#short selling optioin
enable_short = st.sidebar.radio('Enable Short Selling',['Yes','No'])
if enable_short == 'Yes':
    sellShort = True
else:
    sellShort = False

#main, after everything is set up. Run it through each of the functions
data = get_ticker_data(ticker, api, days)
data = calc_moving_average(data, ticker, window)
sma_trade = ma_backtest(data, window, strategy, sellShort)
plot(sma_trade, window, strategy)
st.pyplot()

strategy_return = sma_trade.Cumulative[-1]
buy_hold = ((sma_trade.close[-1]-sma_trade.close[0])/sma_trade.close[0])

if strategy_return > buy_hold:
    st.success('Percent return on this strategy would have been {:.2%}'.format(sma_trade.Cumulative[-1]))
    st.info('Percent return on buy and hold would have been {:.2%}'.format((sma_trade.close[-1]-sma_trade.close[0])/sma_trade.close[0]))
else:
    st.warning('Percent return on this strategy would have been {:.2%}'.format(sma_trade.Cumulative[-1]))
    st.success('Percent return on buy and hold would have been {:.2%}'.format((sma_trade.close[-1]-sma_trade.close[0])/sma_trade.close[0]))
st.dataframe(data)