import pandas as pd
import matplotlib.pyplot as plt 
import pytz
import datetime as dt
from datetime import date
import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import os
import alpaca_trade_api as tradeapi
from sklearn.linear_model import LinearRegression
import numpy as np

def get_orders(api):
    count = 0
    search = True

    while search:

        if count < 1:
            # get most recent activities
            data = api.get_activities()
            # Turn the activities list into a dataframe for easier manipulation
            data = pd.DataFrame([activity._raw for activity in data])
            # get the last order id for pagination
            split_id = data.id.iloc[-1]

            trades = data

        else:
            data = api.get_activities(direction='desc', page_token=split_id)
            data = pd.DataFrame([activity._raw for activity in data])

            if data.empty:
                search = False

            else:
                split_id = data.id.iloc[-1]
                trades = trades.append(data)

        count += 1

    # filter out partially filled orders
    trades = trades[trades.order_status == 'filled']
    trades = trades.reset_index(drop=True)
    trades = trades.sort_index(ascending=False).reset_index(drop=True)

    # calculate profit
    # dfProfit = trades

   # convert filled_at to date
    trades['transaction_time'] = pd.to_datetime(trades['transaction_time'], format="%Y-%m-%d")

    # remove time
    trades['transaction_time'] = trades['transaction_time'].dt.strftime("%Y-%m-%d")
    # sort first based on symbol, then type as per the list above, then submitted date
    trades.sort_values(by=['symbol', 'transaction_time', 'type'], inplace=True, ascending=True)

    # reset index
    trades.reset_index(drop=True, inplace=True)
    # add empty 'profit' column
    trades['profit'] = ''

    totalProfit = 0.0
    profitCnt   = 0
    lossCnt     = 0
    slCnt       = 0
    ptCnt       = 0
    trCnt       = 0
    qty         = 0
    profit      = 0
    sign        = {'buy': -1, 'sell': 1, 'sell_short': 1}


    for index, row in trades.iterrows():

        if index > 0:
            if trades['symbol'][index - 1] != trades['symbol'][index]:
                qty    = 0
                profit = 0

        side      = trades['side'][index]
        filledQty = int(trades['cum_qty'][index]) * sign[side]
        qty       = qty + filledQty
        price     = float(trades['price'][index])
        pl        = filledQty * price
        profit    = profit + pl

        if qty==0:
            # complete trade
            trCnt = trCnt + 1
            # put the profit in its column
            trades.loc[index, 'profit'] = round(profit, 2)
            totalProfit = totalProfit + profit
            if profit >= 0:
                profitCnt = profitCnt + 1
                if trades['type'][index] == 'limit':
                    ptCnt = ptCnt + 1
            else:
                lossCnt = lossCnt + 1
                if trades['type'][index] == 'stop_limit':
                    slCnt = slCnt + 1
            profit = 0

    # change empty rows to 0
    trades.profit = [0 if p == '' else p for p in trades.profit]

    return(trades)

def get_portfolio_history(api, startingDate):
    '''
    startingDate is the beginning of portfolio history, format YYYY-MM-DD
    '''
    history = api.get_portfolio_history(period='2500D', timeframe='1D').df.reset_index()

    print(history.head())

    # convert transaction_time to date
    history['timestamp'] = pd.to_datetime(history['timestamp'], format="%Y-%m-%d")
    # remove time
    history['timestamp'] = history['timestamp'].dt.strftime("%Y-%m-%d")
    
    # filter by date
    dateFilter = '2021-01-19'
    history = history[history.timestamp >= startingDate].reset_index(drop=True)
    
    # another filter because alpaca...
    history = history[history.equity > 0].reset_index(drop=True)
    history = history[history.equity < 85000].reset_index(drop=True)
    
    history.set_index('timestamp', inplace=True)

    return(history)

def get_current_positions(api):
    positions = api.list_positions()
    side = {'long':1, 'short':-1}

    ticker = []
    current_price = []
    cost_basis = []
    shares = []
    today_change = []
    total_change = []

    for position in positions:
        ticker.append(position.symbol)
        current_price.append(float(position.current_price))
        cost_basis.append(float(position.cost_basis))
        shares.append(float(position.qty) * side[position.side])
        today_change.append(float(position.unrealized_intraday_pl))
        total_change.append(float(position.unrealized_plpc))

    portfolio = pd.DataFrame({
        'Ticker':ticker,
        'Current Price':current_price,
        'Cost Basis':cost_basis,
        'Shares':shares,
        'Change Today ($)':today_change,
        'Total Return (%)':total_change
    })
    
    return(portfolio)

def analyze_trades(api, trades, ticker):
    # filter trades to the ticker of interest
    filteredTrades = trades[trades.symbol == ticker].copy()
    
    currentTradeLen = 0
    tradeLenList = []
    for p in filteredTrades.profit:
        if p == 0:
            currentTradeLen += 1
        else:
            tradeLenList.append(currentTradeLen)
            currentTradeLen = 0

    numberOfTrades = len(tradeLenList)
    avgTradeLen = sum(tradeLenList) / numberOfTrades
    avgProfit = sum(filteredTrades.profit) / numberOfTrades
    maxProfit = max(filteredTrades.profit)
    minProfit = min(filteredTrades.profit)
    totalProfit = sum(filteredTrades.profit)
    
    results = pd.DataFrame({
        'Trades':numberOfTrades,
        'Avg Trade Length (Days)':avgTradeLen,
        'Avg Profit':avgProfit,
        'Max Profit':maxProfit,
        'Min Profit':minProfit,
        'Total Profit':totalProfit
    }
    , index=[ticker])
    
    return(results, filteredTrades)

def plot_trades(ticker, trades):
    '''
    trades should be the result of analyze trades
    '''
    # get stock price data
    data = api.get_barset(ticker, 'day', 1000)[ticker].df.reset_index()
    # convert time to date
    data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d")
    # remove time
    data['time'] = data['time'].dt.strftime("%Y-%m-%d")
    # reset index
    # join tradeData and price data
    trades = pd.merge(trades, data, left_on='transaction_time', right_on='time', how='right')
    
    buyPrice = []
    sellPrice = []
    for i in range(len(trades.price)):
        if (trades.side[i] != 'buy') & (trades.side[i] != 'sell') & (trades.side[i] != 'sell_short'):
            buyPrice.append(np.nan)
            sellPrice.append(np.nan)
        elif trades.side[i] == 'buy':
            buyPrice.append(trades.price[i])
            sellPrice.append(np.nan)
        elif (trades.side[i] == 'sell') | (trades.side[i] == 'sell_short'):
            sellPrice.append(trades.price[i])
            buyPrice.append(np.nan)

    trades['buyPrice'] = buyPrice
    trades['sellPrice'] = sellPrice
    trades.set_index('time', inplace=True)

    plt.plot(trades.index, trades.open, alpha=0.6, label=f'{ticker} price data')
     
    plt.scatter(trades.index, trades.buyPrice, marker = '^', s = 5, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(trades.index, trades.sellPrice, marker = 'v', s = 5, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')
        
    return None

def analyze_portfolio(history, percentile=.95, varDaysHorizon=30):
    
    # beta
    # get sp500 returns for beta calculation
    sp500 = api.get_barset('SPY', 'day', 1000)['SPY'].df.reset_index()[['time', 'close']]
    # convert date
    sp500['time'] = pd.to_datetime(sp500['time'], format="%Y-%m-%d")
    # remove time
    sp500['time'] = sp500['time'].dt.strftime("%Y-%m-%d")
    
    history = pd.merge(history, sp500, left_on='timestamp', right_on='time', how='left')
    portfolio_returns = np.array(history['profit_loss_pct']).reshape((-1,1))
    sp500_returns = np.array(history.close)
    beta = LinearRegression().fit(portfolio_returns, sp500_returns).coef_[0]
    
    # VaR
    sorted_history = history['profit_loss_pct'].sort_values()
    percentile_value = round(len(sorted_history) * (1 - (percentile / 100)))
    VaR = sorted_history[percentile_value] * np.sqrt(varDaysHorizon)
    
    # max drawdown
    peak = history.equity.cummax()
    daily_drawdown = history.equity / peak - 1.0
    max_daily_drawdown = daily_drawdown.cummin().iloc[-1]
    results = pd.DataFrame({
        'Beta':beta,
        f'VaR ({varDaysHorizon} day horizon)':VaR,
        'Max Draw':max_daily_drawdown
    }, index=[history.index[-1]])
    
    return(results)
 
st.title("Alpaca Dashboard")
account = st.sidebar.radio('Select the Account to pull', ['Paper', 'Live'])

if account == 'Paper':
    url = "https://paper-api.alpaca.markets"
else:
    url = "https://api.alpaca.markets"

key = st.sidebar.text_input('Please enter your Alpaca Key','PK5S94NMW8O14WQPAKIT')
secret = st.sidebar.text_input('Please enter your Alpaca Secret','CHOxjFZZlwOz8s5lNRs0LzeQew5bk63ZMrorCB7h')
startingDate = st.sidebar.date_input('Please enter the date you would like to begin pulling trade history from',dt.datetime(2021,1,1))

# if account == 'Paper':
#     url = "https://paper-api.alpaca.markets"
#     key = 'PK5S94NMW8O14WQPAKIT'
#     secret = 'CHOxjFZZlwOz8s5lNRs0LzeQew5bk63ZMrorCB7h'
#     startingDate = '2021-02-04'
# elif account == 'Live':
    
os.environ["APCA_API_BASE_URL"] = url
api = tradeapi.REST(key, secret, api_version='v2')

data = get_orders(api)
history = get_portfolio_history(api, str(startingDate))
portfolio_analysis = analyze_portfolio(history)
positions = get_current_positions(api)

# note, the current pct_gain_loss from portfolio history does not work. RIP
st.header('Account Equity')
equityChange = positions['Change Today ($)'].sum()
st.subheader(f'Change in Account equity today is {equityChange / history.equity.iloc[-2] * 100}%')
plt.plot(history.equity, label=f'{account} equity')
st.pyplot()

st.header('Portfolio statistics')
st.dataframe(portfolio_analysis)

st.header('Current Positions')
st.dataframe(positions)
st.header(f'Analysis of trades by Ticker')
ticker = st.selectbox('Select a ticker from the trading history', data.symbol.unique())
st.subheader(f'Analysis of trades for {ticker}')
tickerResults = analyze_trades(api, data, ticker)
st.dataframe(tickerResults[0])
plot = plot_trades(ticker, tickerResults[1])
st.pyplot()

st.subheader(f'All trade details for {ticker}')
st.dataframe(tickerResults[1][['symbol', 'transaction_time', 'price', 'qty', 'side', 'profit']])
