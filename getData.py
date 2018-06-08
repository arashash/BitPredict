#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 20:27:35 2018

@author: arash
"""
import os
import time
import numpy as np
import pandas as pd
from textblob import TextBlob

from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=0)

from datetime import datetime
UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()

from binance.client import Client
api_key = 'e0LdDWHVXTAdEAOPFT7r3Kgy4WX0iuyIsfKZRLFHStp2rnsTyddoYRBLHFfAzzi3'
api_secret = 'Ps0tL8vZJyUGFdDl6C5FbYnroSi4j0kOEYa2OlLGu5vZLx6NVyc1EsgJdlgZOApR'
client = Client(api_key, api_secret)



end_date = datetime.utcnow()
date = end_date.date()
dateStr = date.strftime('%Y,%m,%d,%H,%M,%S').split(',')
dateNums = [ int(x) for x in dateStr ]


coins = ['BTC',
'TRX','XVG',
'MCO','ETH','XRP','XLM','ADA', 'NEO'
,'EOS','ICX','BNB','ONT','NANO','IOTA','LTC','VEN','XMR'
#,'ETC','IOST','OMG','SUB','WAN','NEBL','QTUM','MTL','ELF','GVT','AION'
##,'CLOAK',
#,'QLC','LINK','SNT','WAVES','BTG','ENJ','BQX','EDO','STRAT','POA','NULS','TRIG'
#,'SALT','STEEM','LEND','VIBE','BCPT','POWR','DGD','ZIL','CMT','WTC','DASH','POE'
#,'LSK','LUN','ENG','ZRX','XEM','ADX','WPR','ARN','ZEC','XZC','PPT','ARK','INS'
#,'CND','RCN','AMB','DLT','VIB','OST','BTS','GAS','BRD','DNT','GTO','HSR','FUN'
#,'CHAT','NAV','LRC','TNB','QSP','REQ','BLZ','KMD','APPC','KNC','AE','BCD','SYS'
#,'RPX','SNGLS','MDA','WABI','FUEL','TNT','VIA','MTH','GXS','EVX','RLC','CDT'
#,'AST','WINGS','YOYO','STORJ','PIVX','SNM','BNT','ICN','RDN','OAX','MANA','MOD'
]


coin_dict = {
    'BTC': 'Bitcoin',
    'TRX': 'TRON',    
    'XVG': 'Verge',
    'MCO': 'Monaco',
    'ETH': 'Ethereum',
    'XRP': 'Ripple',
    'XLM': 'Stellar',
    'ADA': 'Cordano',
    'NEO': 'NEO', 
    'EOS': 'EOS',
    'ICX': 'ICON',
    'BNB': 'Binance Coin',
    'ONT': 'Ontology',
    'NANO': 'Nano', 
    'IOTA': 'IOT',
    'LTC': 'Litecoin',
    'VEN': 'VeChain',
    'XMR': 'Monero'
    }
    
for coin in coins:
    print(coin)
    try:
        df = pd.read_csv('/home/arash/BitPredict/data/%s.csv'%coin)
        print(len(df))
    except:

        # the the historical data of the coins and add to the dataframe
        if coin=='BTC':
            coinpair = "BTCUSDT"
        else:
            coinpair = '{}BTC'.format(coin)
            
        start_date = "1 Jan, 2018"  
        end_date = "1 June, 2018" 
        interval = Client.KLINE_INTERVAL_1HOUR
        klines = client.get_historical_klines(coinpair, interval, start_date, end_date)
        df = pd.DataFrame(klines)
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time',
                         'quote asset volume', 'number of trades', 'taker buy base asset volume',
                         'taker buy quote asset volume', 'ignore']
        df['date'] = pd.to_datetime(df['time'], unit='ms')
        df.index = df['date']
        
        # the Google trends historical interests
        searchQuery = coin_dict[coin]+' '+coin
        kw_list = [searchQuery]
        trendsData = pytrends.get_historical_interest(kw_list,
                                 year_start=2018, month_start=1, day_start=1, hour_start=0,
                                 year_end=2018, month_end=6, day_end=1, hour_end=0,
                                 cat=0, geo='', gprop='', sleep=0)
        
        # add to the final dataframe
        df = df.join(trendsData)

        
        
        # get Twitter sentiments
        os.system('python ./GetOldTweets/Exporter.py --querysearch "%s %s" --since 2018-01-01 --until 2018-06-01'%(coin_dict[coin], coin))
        twData = pd.read_csv('/home/arash/BitPredict/output_got.csv', sep=';')
        os.system('rm /home/arash/BitPredict/output_got.csv')
        
        twData['sentiment'] = 0
        for index, row in twData.iterrows():
            
            # sentiment analysis
            analysis = TextBlob(row['text'])
            twData.ix[index, 'sentiment'] = analysis.sentiment.polarity
            
            # convert to utc time
            local_datetime = datetime.strptime(row['date'], "%Y-%m-%d %H:%M")
            result_utc_datetime = local_datetime + UTC_OFFSET_TIMEDELTA
            result_utc_str = result_utc_datetime.strftime("%Y-%m-%d %H:%M")
            
            # round down to the nearest hour
            twData.ix[index, 'date'] = result_utc_str[:-2]+'00:00'
                
        
        # group and average over hourly dates
        sub_twData = twData.groupby('date').mean()[['sentiment']]
        
        # add to the final dataframe
        df = df.join(sub_twData)
        

        df = df [['open', 'high', 'low', 'close', 'volume', 'number of trades', searchQuery, 'sentiment']]
        
        length = len(df)
        print(length)
        df.to_csv('/home/arash/BitPredict/data/%s.csv'%coin)
        
        rand = 60*np.random.rand()
        time.sleep(rand)
        
