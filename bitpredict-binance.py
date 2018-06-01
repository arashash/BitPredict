#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:39:43 2018

@author: arash
"""

import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd

import time

from talib.abstract import *
import talib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import Ridge

from sklearn import preprocessing
from sklearn import decomposition

from sklearn.metrics import r2_score

from tensorflow import reset_default_graph
reset_default_graph()
import tflearn

import smtplib
import email.mime.multipart
import email.mime.text

from df2gspread import df2gspread as d2g
spreadsheet = '/spreadsheets/altcoin_predictions'

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
interval = Client.KLINE_INTERVAL_12HOUR
testSize = 30
featureSize = 50
predDays = 1
nEpoch = 10
email_threshold = 10
binance_coins = [ 'USDT',
'TRX','XVG','NCASH',
'MCO','ETH','XRP','XLM','ADA','GRS','NEO'
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
    

def getFeatures(altcoin_data, altcoin, predDays):
    # selecting the features of the given altcoin
    df = altcoin_data[altcoin][['time', 'open', 'close',
                     'high', 'low', 'volume']].astype('float')
    if altcoin!='USDT':
        df_btc = altcoin_data['USDT'][['time', 'open', 'close',
                     'high', 'low', 'volume']].astype('float')
        
        

    df['change'] = (df['high'] - df['open']) / df['open']

    
    # adding future days bullish candles as new features
    df['change+'] = 0
    for i in range(predDays):
        df['change+'] += df['change'].shift(-i-1)    

    


    # For adding technical indicators
    raw = df[['open', 'high', 'low', 'close', 'volume']]

    df.head()
    functions = talib.get_functions()
    remove_list = ['MAVP']
    for item in remove_list:
        functions.remove(item)
    
    
    
    data = df[['time', 'change', 'change+']]
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
#                
                
                
   
    data.fillna(method='backfill', inplace=True)
    
    features = data.copy()
    features.drop(['change+'], axis = 1, inplace=True)
    
    # getting last sample for prediction
    x_today = np.nan_to_num(features.iloc[-1].values)
    
    today = features.iloc[-1]['time']
    
    # remove the last samples since they do not have a tomorow samples
    features.drop(df.index[-predDays:])
    data.drop(df.index[-predDays:])


    X = features.values
    X = np.nan_to_num(X)


    nextPriceAction = data['change+'].values
    nextPriceAction = np.nan_to_num(nextPriceAction)

    # computes the target label as whether the tomorrow value exceeds the threshold
    # in order to make the dataset balanced, i.e., number of pos and neg labels be equal
    y = nextPriceAction.reshape((-1, 1))

    return X, y, x_today, today, nextPriceAction, X.shape[0]
    

def preprocess(X, x_today, y, testSize, k):
        
    X_train = X[0:-(testSize-1), :]
    y_train = y[0:-(testSize-1)]
    
    # normalize the dataFrame
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X = min_max_scaler.fit_transform(X)
    x_today = min_max_scaler.transform(x_today.reshape(1, -1))
    
    # feature selection
    selector = SelectKBest(mutual_info_regression, k=100)
    X_train = selector.fit_transform(X_train, y_train)
    X = selector.transform(X)
    x_today = selector.transform(x_today)
    
    # dimensionality reduction
    pca = decomposition.PCA(n_components=k)
    pca.fit(X)
    X = pca.transform(X)
    x_today = pca.transform(x_today)
    
    return X, x_today

    



# Building deep neural network
input_layer = tflearn.input_data(shape=[None, featureSize])

dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.5)

dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.5)

dense3 = tflearn.fully_connected(dropout2, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout3 = tflearn.dropout(dense2, 0.5)

dense4 = tflearn.fully_connected(dropout3, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout4 = tflearn.dropout(dense2, 0.5)

dense5 = tflearn.fully_connected(dropout4, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout5 = tflearn.dropout(dense2, 0.5)

softmax = tflearn.fully_connected(dropout5, 1, activation='linear')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
net = tflearn.regression(softmax, optimizer=sgd, metric='R2',
                         loss='mean_square')

predictor = tflearn.DNN(net, tensorboard_verbose=0)

#predictor = Ridge(alpha=1.0)




pred_results = pd.DataFrame(data=np.zeros(shape=(len(binance_coins), 5)),
                              columns = ['altcoin', 'samples', 'train_score',
                                         'test_score', 'prediction'])
while True:
        
    altcoin_data = {}
    count = 0
    for coin in binance_coins:
        print(coin)       
        
        # the the historical data of the coins and add to the dataframe
        if coin=='USDT':
            coinpair = "BTCUSDT"
        else:
            coinpair = '{}BTC'.format(coin)
            
        try:
            klines = client.get_historical_klines(coinpair, interval, start_date)
        except:
            time.sleep(600)
            klines = client.get_historical_klines(coinpair, interval, start_date)
            
        klines_df = pd.DataFrame(klines)
        klines_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time',
                         'quote asset volume', 'number of trades', 'taker buy base asset volume',
                         'taker buy quote asset volume', 'ignore']
        klines_df.index = klines_df['time']
        altcoin_data[coin] = klines_df


        # get technical indicators as features
        X, y, x_today, today, y_cont, size = getFeatures(altcoin_data, coin, predDays)
        
        # do normalization, feature selection and dimentionality reduction
        X, x_today = preprocess(X, x_today, y, testSize, k=featureSize)
        
        
        # split to train-test sets
        X_train = X[0:-testSize, :]
        y_train = y[0:-testSize]

        X_test = X[-testSize:-1, :]
        y_test = y[-testSize:-1]

        # learn a regularized regression model and determine alpha value with CV
        predictor.fit(X_train, y_train, n_epoch=nEpoch, show_metric=True)
#        predictor.fit(X_train, y_train)
        
        train_predicted = predictor.predict(X_train)
        train_score = r2_score(y_train, train_predicted)
        print('Train score = %0.4f'%train_score)
        
        test_predicted = predictor.predict(X_test)
        test_score = r2_score(y_test, test_predicted)

        print('Test score = %0.4f'%test_score)
        
        predictor.fit(X, y, n_epoch=nEpoch, show_metric=True)
#        predictor.fit(X, y)
        
        prediction = np.mean(predictor.predict(x_today))*100
        print('prediction = %0.4f\n'%prediction)
        
        pred_results.loc[count,'altcoin'] = coin
        pred_results.loc[count,'samples'] = size
        pred_results.loc[count,'train_score'] = train_score
        pred_results.loc[count,'test_score'] = test_score
        pred_results.loc[count,'prediction'] = prediction
        
    
        count+=1
        
        
    # save the scores and predictions in a Google sheet
    pred_results = pred_results.sort_values(by='prediction', ascending=False)
    d2g.upload(pred_results, spreadsheet, 'predictions')
    print(pred_results)
    
    
#    # send email notification of the best three results if the test_score
#    # reached the threshold
#    best_res = pred_results.iloc[0]
#    best_res1 = pred_results.iloc[1]
#    best_res2 = pred_results.iloc[2]
#    best_res3 = pred_results.iloc[3]
#    if best_res['prediction']>email_threshold:
#        FROM = "arash.asn94@gmail.com"
#        TO = "arash.ashrafnejad@gmail.com"
#        SUBJECT = "BitPredict Price Alert!"
#        TEXT = best_res['altcoin'] +' '+ str(best_res['test_score']).replace('/', ',') + ' --> %0.3f'%best_res['prediction'] +'\n'+ \
#        best_res1['altcoin'] +' '+ str(best_res1['test_score']).replace('/', ',') + ' --> %0.3f'%best_res1['prediction']+'\n'+ \
#        best_res2['altcoin'] +' '+ str(best_res2['test_score']).replace('/', ',') + ' --> %0.3f'%best_res2['prediction']+'\n'+ \
#        best_res3['altcoin'] +' '+ str(best_res3['test_score']).replace('/', ',') + ' --> %0.3f'%best_res3['prediction']+'\n'
#        sendMail(FROM,TO,SUBJECT,TEXT)
            
    
    
