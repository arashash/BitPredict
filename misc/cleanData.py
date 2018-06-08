#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:14:30 2018

@author: arash
"""

import pandas as pd
import os

class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])
    
    
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




df = pd.read_csv('/home/arash/BitPredict/data/BTC.csv')

df.index = df['date']
df = df[['open', 'low', 'close', 'volume', 'number of trades', 'BTC price']]
#os.system('rm /home/arash/BitPredict/data/BTC.csv')

df.to_csv('/home/arash/BitPredict/data/BTC.csv')
btc_length = len(df)

for coin in coins:
    print(coin)
    try:
        df = pd.read_csv('/home/arash/BitPredict/data/%s.csv'%coin)
        length = len(df)
        
        os.system('rm /home/arash/BitPredict/data/%s.csv'%coin)
        if length  == btc_length:
            
            df.index = df['date']
            df = df[['open', 'low', 'close', 'volume', 'number of trades', '%s price'%coin]]

            df.index = df['date']
            df.to_csv('/home/arash/BitPredict/data/%s.csv'%coin)
    except:
        print(coin+' doesnt exist!')


        
        



