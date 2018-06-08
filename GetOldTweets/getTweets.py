#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:43:42 2018

@author: arash
"""
import os
import pandas as pd
from textblob import TextBlob
import datetime

UTC_OFFSET_TIMEDELTA = datetime.datetime.utcnow() - datetime.datetime.now()

coin_dict = {
    'BTC': 'Bitcoin',
    'TRX': 'TRON',    
    'XVG': 'Verge'    
}


coin = 'BTC'
os.system('python Exporter.py --querysearch "%s %s" --since 2018-01-01'%(coin_dict[coin], coin))
df = pd.read_csv('/home/arash/BitPredict/GetOldTweets/output_got.csv', sep=';')
os.system('rm /home/arash/BitPredict/GetOldTweets/output_got.csv')

df['sentiment'] = 0
for index, row in df.iterrows():
    
    # sentiment analysis
    analysis = TextBlob(row['text'])
    df.ix[index, 'sentiment'] = analysis.sentiment.polarity
    
    # convert to utc time
    local_datetime = datetime.datetime.strptime(row['date'], "%Y-%m-%d %H:%M")
    result_utc_datetime = local_datetime + UTC_OFFSET_TIMEDELTA
    result_utc_str = result_utc_datetime.strftime("%Y-%m-%d %H:%M")
    
    # round down to the nearest hour
    df.ix[index, 'date'] = result_utc_str[:-2]+'00:00'
        

sub_df = df.groupby('date').mean()[['sentiment']]
df['date'] = pd.to_datetime(df['date'])


print(sub_df)


