##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Fri Apr 13 20:30:53 2018
#
#@author: arash

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime
import time

from talib.abstract import *
import talib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn import preprocessing
from sklearn import decomposition
    
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import smtplib
import email.mime.multipart
import email.mime.text

from df2gspread import df2gspread as d2g
spreadsheet = '/spreadsheets/altcoin_predictions'
wks_name = 'Sheet1'

from pytrends.request import TrendReq

quandl.ApiConfig.api_key = 'UX1w8dNS5nXfuCqLQSx5'
data_dir = './data/'


# Pull pricing data for 3 more BTC exchanges
exchanges = ['COINBASE','BITSTAMP','ITBIT', 'KRAKEN']

binance_coins = ['ETH', 'LTC', 'ETC','XRP', 'LSK', 'XMR',
                 'STRAT', 'ZEC', 'DASH', 'BTS', 'VIA', 'STEEM'] 


base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015



def sendMail(FROM,TO,SUBJECT,TEXT):

    msg = email.mime.multipart.MIMEMultipart()
    msg['From'] = FROM
    msg['To'] = TO
    msg['Subject'] = SUBJECT
    msg.attach(email.mime.text.MIMEText(TEXT, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('arash.asn94@gmail.com', 'VotNiowkUch3')
    server.sendmail(FROM, TO, msg.as_string())
    server.quit()
    
    
def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), period)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}{}.pkl'.format(data_dir, quandl_id.replace('/','-'))
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas", start_date="2010-01-01")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)


def get_json_data(json_url, file_name):
    '''Download and cache JSON data, return as a dataframe.'''
    cache_path = '{}{}.pkl'.format(data_dir, file_name)
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(file_name))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached response at {}'.format(cache_path))
    return df



    
while True:
    os.system("rm -r ./data")
    os.system("mkdir ./data")
    
    
    exchange_data = {}
    
    for exchange in exchanges:
        exchange_code = 'BCHARTS/{}USD'.format(exchange)
        btc_exchange_df = get_quandl_data(exchange_code)
        exchange_data[exchange] = btc_exchange_df
    
    btc_usd_df = pd.DataFrame()
    
    # Merge the BTC price dataseries' into a single dataframe
    btc_usd_df['weightedAverage'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Weighted Price').replace(0,np.nan).mean(axis=1)  
    btc_usd_df['quoteVolume'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Volume (Currency)') .replace(0, np.nan).mean(axis=1)
    btc_usd_df['volume'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Volume (BTC)').replace(0, np.nan).mean(axis=1)
    btc_usd_df['close'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Close').replace(0, np.nan).mean(axis=1)
    btc_usd_df['low'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Low').replace(0, np.nan).mean(axis=1)
    btc_usd_df['high'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'High').replace(0, np.nan).mean(axis=1)
    btc_usd_df['open'] = merge_dfs_on_column(list(exchange_data.values()),
                                           list(exchange_data.keys()),
                                           'Open').replace(0, np.nan).mean(axis=1)
    
    
    
    
    
    
    
    end_date = datetime.now() # up until today
    period = 86400 # pull daily data (86,400 seconds per day)
    
    
    altcoin_data = {}
    for altcoin in binance_coins:
        coinpair = 'BTC_{}'.format(altcoin)
        crypto_price_df = get_crypto_data(coinpair)
        altcoin_data[altcoin] = crypto_price_df
        
        
        
        
    
    
    
    
    def getData(altcoin, predDays, threshold):
        if altcoin == 'Visa':
            df = visa_stock
        elif altcoin == 'BTC':
            df = btc_usd_df
        else:
            # selecting the features of the given altcoin
            df = altcoin_data[altcoin][['open', 'close', 'high', 'low', 'quoteVolume', 'volume', 'weightedAverage']]
            
    
    
        # renaming the weightedAverage column
        df = df.rename(columns={'weightedAverage': 'price'})
    
        df['btc'] = btc_usd_df['weightedAverage']
        
    
        # disable panda warning
        pd.options.mode.chained_assignment = None  # default='warn'
    
    
        
        # adding future days prices as new features
        n = predDays
        df['price+'] = 0
        for i in range(n):
            df['price+'] += (n-i)*df['price'].shift(-i-1)    
    
        df['price+'] = df['price+'] / ((n+1)*n/2)
    #     # remove the last samples since they do not have a tomorow samples
    #     df = df.drop(df.index[-n:])
    
    #     df['price+'] = df['price'].shift(-1)
        
    #     # remove the last sample since it does not have a tomorow sample
    #     df = df.drop(df.index[-1])    
        
        
    
        df['priceRate+'] = df['price+']/df['price'] - 1
    
        
        df['vol'] = df['volume']/df['quoteVolume']
    
    
        # adding previous 3 days elements as new features
        for i in range(3):
            df['btc-%d'%(i+1)] = df['btc'].shift(i+1)
            df['price-%d'%(i+1)] = df['price'].shift(i+1)
            df['close-%d'%(i+1)] = df['close'].shift(i+1)
            df['open-%d'%(i+1)] = df['open'].shift(i+1)
            df['vol-%d'%(i+1)] = df['vol'].shift(i+1)
            df['totVol-%d'%(i+1)] = df['quoteVolume'].shift(i+1)
    
    
        # remove the first n samples since they don't have previous day data
        df = df.drop(df.index[0:3])
    
    
        df['btcRate-2'] = df['btc-2']/df['btc-3'] - 1
        df['btcRate-1'] = df['btc-1']/df['btc-2'] - 1
        df['btcRate'] = df['btc']/df['btc-1'] - 1
        df['btcAcc'] = df['btcRate'] - df['btcRate-1']
        df['btcAcc-1'] = df['btcRate-1'] - df['btcRate-2']
        df['btcJerk'] = df['btcAcc'] - df['btcAcc-1']
    
    
        df['volRate-2'] = df['vol-2']/df['vol-3'] - 1
        df['volRate-1'] = df['vol-1']/df['vol-2'] - 1
        df['volRate'] = df['vol']/df['vol-1'] - 1
        df['volAcc'] = df['volRate'] - df['volRate-1']
        df['volAcc-1'] = df['volRate-1'] - df['volRate-2']
        df['volJerk'] = df['volAcc'] - df['volAcc-1']
    
    
        df['totVolRate-2'] = df['totVol-2']/df['totVol-3'] - 1
        df['totVolRate-1'] = df['totVol-1']/df['totVol-2'] - 1
        df['totVolRate'] = df['quoteVolume']/df['totVol-1'] - 1
        df['totVolAcc'] = df['totVolRate'] - df['totVolRate-1']
        df['totVolAcc-1'] = df['totVolRate-1'] - df['totVolRate-2']
        df['totVolJerk'] = df['totVolAcc'] - df['totVolAcc-1']
    
    
    
        df['closeRate-2'] = df['close-2']/df['close-3'] - 1
        df['closeRate-1'] = df['close-1']/df['close-2'] - 1
        df['closeRate'] = df['close']/df['close-1'] - 1
        df['closeAcc'] = df['closeRate'] - df['closeRate-1']
        df['closeAcc-1'] = df['closeRate-1'] - df['closeRate-2']
        df['closeJerk'] = df['closeAcc'] - df['closeAcc-1']
    
    
        df['openRate-2'] = df['open-2']/df['open-3'] - 1
        df['openRate-1'] = df['open-1']/df['open-2'] - 1
        df['openRate'] = df['open']/df['open-1'] - 1
        df['openAcc'] = df['openRate'] - df['openRate-1']
        df['openAcc-1'] = df['openRate-1'] - df['openRate-2']
        df['openJerk'] = df['openAcc']- df['openAcc-1']
    
    
        df['priceRate-2'] = df['price-2']/df['price-3'] - 1
        df['priceRate-1'] = df['price-1']/df['price-2'] - 1
        df['priceRate'] = df['price']/df['price-1'] - 1
        df['priceAcc'] = df['priceRate'] - df['priceRate-1']
        df['priceAcc-1'] = df['priceRate-1'] - df['priceRate-2']
        df['priceJerk'] = df['priceAcc'] - df['priceAcc-1']
    
        df['priceRange'] = df['high']/df['low'] - 1
    
        df['priceSwing'] = df['close']/df['open'] - 1
        
        
        
        # For adding technical indicators
        raw = df[['open', 'high', 'low', 'close', 'volume']]
    
        df.head()
        functions = talib.get_functions()
        remove_list = ['MAVP']
        for item in remove_list:
            functions.remove(item)
        
    
    
        
        
        data = df[['priceRate+',
                    'low', 'high', 'priceRange',
                   'open', 'close', 'priceSwing',
                   'btc', 'btcRate', 'btcAcc', 'btcJerk',
                  'vol', 'volRate', 'volAcc', 'volJerk',
                  'price', 'priceRate', 'priceAcc', 'priceJerk',
                  'quoteVolume', 'totVolRate', 'totVolAcc', 'totVolJerk']]
        
        
        date = end_date.date()
        dateStr = date.strftime('%Y-%m-%d')
        
        # Google trend interests of the altcoin
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([altcoin], cat=0, timeframe='2015-01-01 %s'%dateStr, geo='', gprop='')
        interests = pytrends.interest_over_time()
        data[altcoin+'Interest'] = interests[altcoin]
        
        # Google trend interests of the bitcoin
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(['bitcoin'], cat=0, timeframe='2015-01-01 %s'%dateStr, geo='', gprop='')
        interests = pytrends.interest_over_time()
        data['bitcoinInterest'] = interests['bitcoin']
        
        
        # technical indicators
        for func in functions:
            outputs = eval(func)(raw)
    
            if (type(outputs) is pd.Series):
                data[func] = outputs
        
        n = 150
    #     remove the first n samples
        data = data.drop(data.index[0:n])
    #     data = data.dropna(axis=0, how='any')
    
        
        
        
        features = data.drop(['priceRate+'], axis = 1)
        
        
        # getting today information for prediction
        x_today = np.nan_to_num(features.iloc[-1].values)
        
        today = features.iloc[[-1]].index.date
        
        # remove the last samples since they do not have a tomorow samples
        features = features.drop(df.index[-predDays:])
        data = data.drop(df.index[-predDays:])
    
    
        X = features.values
        X = np.nan_to_num(X)
    #     print('Shape of X: ',X.shape)
    
    
        tomorrrowPriceChange = data['priceRate+'].values
        tomorrrowPriceChange = np.nan_to_num(tomorrrowPriceChange)
            
        # finds the median weighted price changes
        priceChangeMed = np.median(tomorrrowPriceChange)
    #     print('Median daily price change: %0.4f'%priceChangeMed)
    
        # computes the target label as whether the tomorrow value exceeds the threshold
        # in order to make the dataset balanced, i.e., number of pos and neg labels be equal
        y = (tomorrrowPriceChange > threshold)*1
    
    
    #     delta = 0.00
    #     y = (np.sign(tomorrrowPriceChange - delta) + np.sign(tomorrrowPriceChange + delta)) / 2.0
    
    #     buyCount = np.sum(y==1)
    #     sellCount = np.sum(y==-1)
    #     holdCount = np.sum(y==0)
    
        # print(buyCount)
        # print(sellCount)
        # print(holdCount)
        
        
    #     y = (tomorrrowPriceChange > delta)
    
        return X, y, x_today, today, tomorrrowPriceChange, X.shape[0], priceChangeMed
        
        
    
    
    
    
    def preprocess(X, x_today, y, testSize, k):
            
        X_train = X[0:-(testSize-1), :]
        y_train = y[0:-(testSize-1)]
        
        # normalize the dataFrame
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X = min_max_scaler.fit_transform(X)
        x_today = min_max_scaler.transform(x_today.reshape(1, -1))
        
        # feature selection
        selector = SelectKBest(mutual_info_classif, k=100)
        X_train = selector.fit_transform(X_train, y_train)
        X = selector.transform(X)
        x_today = selector.transform(x_today)
        
        # dimensionality reduction
        pca = decomposition.PCA(n_components=k)
        pca.fit(X)
        X = pca.transform(X)
        x_today = pca.transform(x_today)
        
        return X, x_today
    
    
    
    
    
    
    
    
    
    testSize = 200
    predDays = 1
    threshold = 0.01
    df_results = pd.DataFrame(data=np.zeros(shape=(len(binance_coins), 7)),
                                  columns = ['altcoin', 'days', 'median',
                                             'train_score', 'test_score', 'baseline', 'performance'])
    
    classifier = LogisticRegression()
    count = 0
    for coin in binance_coins:
        print(coin)
        X, y, x_today, today, y_cont, size, median = getData(coin, predDays, threshold)
        X, x_today = preprocess(X, x_today, y, testSize, k=20)
        
        t_start = time.clock()
    
        train_scores = np.zeros(testSize)
        test_scores = np.zeros(testSize)
        y_preds = np.zeros(testSize)
        for i in range(testSize):
            X_train = X[0:-(i+1), :]
            y_train = y[0:-(i+1)]
    
            X_test = X[-(i+1), :].reshape(1, -1)
            y_test = y[-(i+1)].reshape(1, -1)
    
            classifier.fit(X_train, y_train)
            train_scores[i] = classifier.score(X_train, y_train)
            test_scores[i] = classifier.score(X_test, y_test)
            y_preds[i] = classifier.predict(X_test)
    
        t_end = time.clock()
        t_diff = t_end - t_start
    
        df_results.loc[count,'altcoin'] = coin
        df_results.loc[count,'days'] = size
        df_results.loc[count,'median'] = median
        df_results.loc[count,'train_score'] = np.mean(train_scores)
        df_results.loc[count,'test_score'] = np.mean(test_scores)
        df_results.loc[count,'time'] = t_diff
    
    
        y_test = y[testSize:]
        baseline = np.sum(y_test==np.argmax([np.sum(y_test==0),
                                             np.sum(y_test==1)]))/np.size(y_test)
        df_results.loc[count,'baseline'] = baseline
        
        df_results.loc[count,'performance'] = (np.mean(test_scores) - baseline)/(1 - baseline)
    
        count+=1
        
    selected_coins = df_results['altcoin'].values[df_results['performance'] > 0]
    
    
    
    
    
    
    
    testSize = 2
    predDays = 1
    threshold = 0.02
    df_results = pd.DataFrame(data=np.zeros(shape=(len(selected_coins), 5)),
                                  columns = ['Altcoin', 'Day', 'Sell', 'Buy', 'Prediction'])
    
    classifier = LogisticRegression()
    count = 0
    for coin in selected_coins:
        print(coin)
        X, y, x_today, today, y_cont, size, median = getData(coin, predDays, threshold)
        X, x_today = preprocess(X, x_today, y, testSize, k=20)
        
    
        classifier.fit(X, y)
        probs = classifier.predict_proba(x_today)
        prediction = classifier.predict(x_today)
    
    
        df_results.loc[count,'Altcoin'] = coin
        df_results.loc[count,'Day'] = today
        df_results.loc[count,'Sell'] = probs[0][0]
        df_results.loc[count,'Buy'] = probs[0][1]
        df_results.loc[count,'Prediction']
    
    
        count+=1
        
    df_results = df_results.sort_values(by='Buy', ascending=False)
    
    d2g.upload(df_results, spreadsheet, wks_name)
    
    best_res = df_results.iloc[0]    
    if best_res['Buy']>0.7:
        FROM = "arash.asn94@gmail.com"
        TO = "arash.ashrafnejad@gmail.com"
        SUBJECT = "BitPredict Price Alert!"
        TEXT = best_res['Altcoin'] +' '+ str(best_res['Day'][0]).replace('/', ',') + ' --> %0.3f'%best_res['Buy']
        sendMail(FROM,TO,SUBJECT,TEXT)
