#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:39:43 2018

@author: arash
"""

import os
import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd
import pickle

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

#from df2gspread import df2gspread as d2g
#spreadsheet = '/spreadsheets/altcoin_predictions'
#wks_name = 'Sheet1'

#from pytrends.request import TrendReq

import quandl
quandl.ApiConfig.api_key = 'UX1w8dNS5nXfuCqLQSx5'
data_dir = './data/'

from binance.client import Client
api_key = 'e0LdDWHVXTAdEAOPFT7r3Kgy4WX0iuyIsfKZRLFHStp2rnsTyddoYRBLHFfAzzi3'
api_secret = 'Ps0tL8vZJyUGFdDl6C5FbYnroSi4j0kOEYa2OlLGu5vZLx6NVyc1EsgJdlgZOApR'
client = Client(api_key, api_secret)


# disable panda warning
pd.options.mode.chained_assignment = None  # default='warn'

start_date = "1 Jan, 2018"  
interval = Client.KLINE_INTERVAL_1HOUR
testSize = 200
predDays = 12
threshold = 0.5
performance_threshold = 0.0
email_threshold = 0.8
max_request_delay = 1
binance_coins = [ 'USDT',
'TRX','XVG','NCASH','MCO','ETH','XRP','XLM','ADA','GRS','NEO'
,'EOS','ICX','BNB','BCC','STORM','BAT','ONT','NANO','IOTA','LTC','VEN','XMR'
,'ETC','IOST','OMG','SUB','WAN','NEBL','QTUM','MTL','ELF','GVT','AION'
#,'CLOAK',
,'QLC','LINK','SNT','WAVES','BTG','ENJ','BQX','EDO','STRAT','POA','NULS','TRIG'
,'SALT','STEEM','LEND','VIBE','BCPT','POWR','DGD','ZIL','CMT','WTC','DASH','POE'
,'LSK','LUN','ENG','ZRX','XEM','ADX','WPR','ARN','ZEC','XZC','PPT','ARK','INS'
,'CND','RCN','AMB','DLT','VIB','OST','BTS','GAS','BRD','DNT','GTO','HSR','FUN'
,'CHAT','NAV','LRC','TNB','QSP','REQ','BLZ','KMD','APPC','KNC','AE','BCD','SYS'
,'RPX','SNGLS','MDA','WABI','FUEL','TNT','VIA','MTH','GXS','EVX','RLC','CDT'
,'AST','WINGS','YOYO','STORJ','PIVX','SNM','BNT','ICN','RDN','OAX','MANA','MOD'
]

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
    

def getData(altcoin_data, altcoin, predDays, threshold):
    # selecting the features of the given altcoin
    df = altcoin_data[altcoin][['time', 'open', 'close',
                     'high', 'low', 'volume']].astype('float')
    if altcoin!='USDT':
        df_btc = altcoin_data['USDT'][['time', 'open', 'close',
                     'high', 'low', 'volume']].astype('float')
        
        

    df['bullish'] = (df['open'] < df['close'])*1

    
    # adding future days bullish candles as new features
    n = predDays
    df['bullish+'] = 0
    for i in range(n):
        df['bullish+'] += (n-i)*df['bullish'].shift(-i-1)    

    df['bullish+'] = df['bullish+'] / ((n+1)*n/2)
    


    # For adding technical indicators
    raw = df[['open', 'high', 'low', 'close', 'volume']]

    df.head()
    functions = talib.get_functions()
    remove_list = ['MAVP']
    for item in remove_list:
        functions.remove(item)
    
    
    
    data = df[['time', 'bullish', 'bullish+']]
    # technical indicators
    for func in functions:
        outputs = eval(func)(raw)

        if (type(outputs) is pd.Series):
            data[func] = outputs
    

    
    if altcoin!='USDT':
        raw_btc = df_btc[['open', 'high', 'low', 'close', 'volume']]
        # technical indicators
        for func in functions:
            outputs = eval(func)(raw_btc)
    
            if (type(outputs) is pd.Series):
                data[func+'_btc'] = outputs
                
                
                
   
    data.fillna(method='backfill', inplace=True)
    
    features = data.copy()
    features.drop(['bullish+'], axis = 1, inplace=True)
    
    # getting last sample for prediction
    x_today = np.nan_to_num(features.iloc[-1].values)
    
    today = features.iloc[-1]['time']
    
    # remove the last samples since they do not have a tomorow samples
    features.drop(df.index[-predDays:])
    data.drop(df.index[-predDays:])


    X = features.values
    X = np.nan_to_num(X)


    tomorrrowPriceChange = data['bullish+'].values
    tomorrrowPriceChange = np.nan_to_num(tomorrrowPriceChange)
  

    # computes the target label as whether the tomorrow value exceeds the threshold
    # in order to make the dataset balanced, i.e., number of pos and neg labels be equal
    y = (tomorrrowPriceChange > threshold)*1


    return X, y, x_today, today, tomorrrowPriceChange, X.shape[0]
    

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

    



while True:
    altcoin_data = {}
    for altcoin in binance_coins:
#        print(altcoin)
        if altcoin=='USDT':
            coinpair = "BTCUSDT"
        else:
            coinpair = '{}BTC'.format(altcoin)
        klines = client.get_historical_klines(coinpair, interval, start_date)
        klines_df = pd.DataFrame(klines)
        klines_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time',
                         'quote asset volume', 'number of trades', 'taker buy base asset volume',
                         'taker buy quote asset volume', 'ignore']
        klines_df.index = klines_df['time']
        altcoin_data[altcoin] = klines_df
        random_sleep_time = np.random.rand()*max_request_delay
        time.sleep(random_sleep_time)
    
    
    
    
    df_results = pd.DataFrame(data=np.zeros(shape=(len(binance_coins), 6)),
                                  columns = ['altcoin', 'days', 'train_score',
                                             'test_score', 'baseline', 'performance'])
    
    classifier = LogisticRegression()
    count = 0
#    for coin in binance_coins:
##        print(coin)
#        X, y, x_today, today, y_cont, size = getData(altcoin_data, coin, predDays, threshold)
#        X, x_today = preprocess(X, x_today, y, testSize, k=20)
#        
#        t_start = time.clock()
#    
#        train_scores = np.zeros(testSize)
#        test_scores = np.zeros(testSize)
#        y_preds = np.zeros(testSize)
#        for i in range(testSize):
#            X_train = X[0:-(i+1), :]
#            y_train = y[0:-(i+1)]
#    
#            X_test = X[-(i+1), :].reshape(1, -1)
#            y_test = y[-(i+1)].reshape(1, -1)
#    
#            classifier.fit(X_train, y_train)
#            train_scores[i] = classifier.score(X_train, y_train)
#            test_scores[i] = classifier.score(X_test, y_test)
#            y_preds[i] = classifier.predict(X_test)
#    
#        t_end = time.clock()
#        t_diff = t_end - t_start
#    
#        df_results.loc[count,'altcoin'] = coin
#        df_results.loc[count,'days'] = size
#        df_results.loc[count,'train_score'] = np.mean(train_scores)
#        df_results.loc[count,'test_score'] = np.mean(test_scores)
#        df_results.loc[count,'time'] = t_diff
#    
#    
#        y_test = y[testSize:]
#        baseline = np.sum(y_test==np.argmax([np.sum(y_test==0),
#                                             np.sum(y_test==1)]))/np.size(y_test)
#        df_results.loc[count,'baseline'] = baseline
#        
#        df_results.loc[count,'performance'] = (np.mean(test_scores) - baseline)/(1 - baseline)
#    
#        count+=1
#        
#    selected_coins = df_results['altcoin'].values[df_results['performance'] > performance_threshold]
#    print(df_results)
#    
    
    
#    df_results = pd.DataFrame(data=np.zeros(shape=(len(selected_coins), 5)),
#                                  columns = ['Altcoin', 'Day', 'Sell', 'Buy', 'Prediction'])
    
    df_results = pd.DataFrame(data=np.zeros(shape=(len(binance_coins), 5)),
                                  columns = ['Altcoin', 'Day', 'Sell', 'Buy', 'Prediction'])
    classifier = LogisticRegression()
    count = 0
#    for coin in selected_coins:
    for coin in binance_coins:
#        print(coin)
        X, y, x_today, today, y_cont, size = getData(altcoin_data, coin, predDays, threshold)
        X, x_today = preprocess(X, x_today, y, testSize = 2, k=30)
        
    
        classifier.fit(X, y)
        probs = classifier.predict_proba(x_today)
        prediction = classifier.predict(x_today)
    
    
        df_results.loc[count,'Altcoin'] = coin
        df_results.loc[count,'Day'] = datetime.fromtimestamp(today/1000).strftime('%Y-%m-%d %H:%M') 
        df_results.loc[count,'Sell'] = probs[0][0]
        df_results.loc[count,'Buy'] = probs[0][1]
        df_results.loc[count,'Prediction']
    
    
        count+=1
        
    df_results = df_results.sort_values(by='Buy', ascending=False)
    
     #   d2g.upload(df_results, spreadsheet, wks_name)
    
    if len(df_results) > 3:
        best_res = df_results.iloc[0]
        best_res1 = df_results.iloc[1]
        best_res2 = df_results.iloc[2]
        best_res3 = df_results.iloc[3]
        if best_res['Buy']>email_threshold:
            FROM = "arash.asn94@gmail.com"
            TO = "arash.ashrafnejad@gmail.com"
            SUBJECT = "BitPredict Price Alert!"
            TEXT = best_res['Altcoin'] +' '+ str(best_res['Day']).replace('/', ',') + ' --> %0.3f'%best_res['Buy']+'\n'+ \
            best_res1['Altcoin'] +' '+ str(best_res1['Day']).replace('/', ',') + ' --> %0.3f'%best_res1['Buy']+'\n'+ \
            best_res2['Altcoin'] +' '+ str(best_res2['Day']).replace('/', ',') + ' --> %0.3f'%best_res2['Buy']+'\n'+ \
            best_res3['Altcoin'] +' '+ str(best_res3['Day']).replace('/', ',') + ' --> %0.3f'%best_res3['Buy']+'\n'
            sendMail(FROM,TO,SUBJECT,TEXT)
            
    
    print(df_results)
