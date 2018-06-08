#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 00:40:21 2018

@author: arash
"""


import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
coin = 'XVG'
df = pd.read_csv('/home/arash/BitPredict/data/%s.csv'%coin)
#pd.stats.var.granger_causality()

df['change'] = (df['high'] - df['low']) / df['low']*100

df[[coin+' price']].plot()
df[['change']].plot()


sub_df = df[['change', coin+' price']]
corrs = sub_df.corr()
print(corrs)
autocorrelation_plot(sub_df)
plt.show()

results = grangercausalitytests(sub_df, 100)
print(results[1])