#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 20:27:35 2018

@author: arash
"""
import pandas as pd

from pytrends.request import TrendReq
from datetime import datetime

pytrends = TrendReq(hl='en-US', tz=0)

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
    kw_list = []
    kw_list.append(coin+' price')
    data = pytrends.get_historical_interest(kw_list,
                             year_start=2017, month_start=1, day_start=1, hour_start=0,
                             year_end=dateNums[0], month_end=dateNums[1], day_end=dateNums[2], hour_end=dateNums[3],
                             cat=0, geo='', gprop='', sleep=0)
    
    data.to_csv('/home/arash/BitPredict/pyTrendsData/%s.csv'%coin)
    data = pd.read_csv('/home/arash/BitPredict/pyTrendsData/%s.csv'%coin)
    print(data.tail())