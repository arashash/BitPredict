#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:26:55 2018

@author: arash
"""

import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
np.set_printoptions(3)


window = 100
nb_epoch = 100
validation_split = 0.1
batch_size = 768
verbose = 1
coin = 'TRX'

df_BTC = pd.read_csv('/home/arash/BitPredict/data/BTC.csv')
df = pd.read_csv('/home/arash/BitPredict/data/%s.csv'%coin)


df['btc_open'] = df_BTC['open']
df['btc_high'] = df_BTC['high']
df['btc_low'] = df_BTC['low']
df['btc_close'] = df_BTC['close']
df['btc_volume'] = df_BTC['volume']
df['btc_change'] = (df_BTC['high'] - df_BTC['open']) / df_BTC['open']*100


df['change'] = (df['high'] - df['open']) / df['open']*100

features = df[['open', 'high', 'low', 'close', 'volume',
#               'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume',
               coin+' price', 'btc_change', 'change']]

def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape
    
    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
    return X_train, X_test



def preprocess_data (features, seq_len):
    features = features[:: -1]
    amount_of_features = len(features.columns)
    data = features.as_matrix()
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index : index + sequence_length])
        
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]
#    train, result = standard_scaler(train, result)
    
    X_train = train[:, : -1]
    y_train = train[:, -1][: ,-1]
    X_test = result[int(row) :, : -1]
    y_test = result[int(row) :, -1][ : ,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

#    print('X_train:')
#    print(X_train)
#    
#    print('y_train:')
#    print(y_train)
#    
#    print('X_test:')
#    print(X_test)
#    
#    print('y_test:')
#    print(y_test)
    return [X_train, y_train, X_test, y_test]


def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model




X_train, y_train, X_test, y_test = preprocess_data(features, window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)



model = build_model([X_train.shape[2], window, 100, 1])



model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    validation_split=validation_split,
    verbose=verbose)


trainScore = model.evaluate(X_train, y_train, verbose=verbose)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=verbose)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))



diff = []
ratio = []
pred = model.predict(X_test)
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))
    
    
    
import matplotlib.pyplot as plt2

plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_test, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()