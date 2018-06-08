#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 20:27:35 2018

@author: arash
"""
import time
import numpy as np
import pandas as pd

from pytrends.request import TrendReq
from datetime import datetime

pytrends = TrendReq(hl='en-US', tz=0)


from binance.client import Client
api_key = 'e0LdDWHVXTAdEAOPFT7r3Kgy4WX0iuyIsfKZRLFHStp2rnsTyddoYRBLHFfAzzi3'
api_secret = 'Ps0tL8vZJyUGFdDl6C5FbYnroSi4j0kOEYa2OlLGu5vZLx6NVyc1EsgJdlgZOApR'
client = Client(api_key, api_secret)



end_date = datetime.now()
date = end_date.date()
dateStr = date.strftime('%Y,%m,%d,%H,%M,%S').split(',')
dateNums = [ int(x) for x in dateStr ]


coins = [ 'BTC',
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


for coin in coins:
    print(coin)
    try:
        df = pd.read_csv('/home/arash/BitPredict/data/%s.csv'%coin)
        print(len(df))
    except:
        # the Google trends historical interests
        kw_list = []
        kw_list.append(coin+' price')
        trendsData = pytrends.get_historical_interest(kw_list,
                                 year_start=2018, month_start=1, day_start=1, hour_start=0,
                                 year_end=dateNums[0], month_end=dateNums[1], day_end=dateNums[2], hour_end=dateNums[3],
                                 cat=0, geo='', gprop='', sleep=0)

        # the the historical data of the coins and add to the dataframe
        if coin=='BTC':
            coinpair = "BTCUSDT"
        else:
            coinpair = '{}BTC'.format(coin)
            
        start_date = "1 Jan, 2018"  
        interval = Client.KLINE_INTERVAL_1HOUR
        
        klines = client.get_historical_klines(coinpair, interval, start_date)
        df = pd.DataFrame(klines)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'close time',
                         'quote asset volume', 'number of trades', 'taker buy base asset volume',
                         'taker buy quote asset volume', 'ignore']

        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.index = df['date']
        
        df = df.join(trendsData)
        df = df.dropna()
        print(len(df))
        df.to_csv('/home/arash/BitPredict/data/%s.csv'%coin)
        rand = 600*np.random.rand()
        time.sleep(rand)
        
