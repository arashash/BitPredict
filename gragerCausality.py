#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 00:40:21 2018

@author: arash
"""


import pandads as pd
from pandas.stats import VAR

coin = 'BTC'
data = pd.read_csv('/home/arash/BitPredict/pyTrendsData/%s.csv'%coin)
VAR.granger_causality()