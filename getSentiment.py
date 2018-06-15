import os
import time

import numpy as np
import pandas as pd
from datetime import datetime
from textblob import TextBlob

from df2gspread import df2gspread as d2g
spreadsheet = '/spreadsheets/altcoin_predictions'


coins = [
'BTC',
'TRX'
,'XVG',
'MCO','ETH','XRP','XLM','ADA', 'NEO',
'EOS','ICX','BNB','ONT','NANO','IOTA','LTC','VEN','XMR',
'ETC','OMG','WAN','QTUM','ELF','AION',
'LINK','SNT','WAVES','BTG','STRAT','NULS',
'SALT','STEEM','DGD','ZIL','WTC','DASH',
'LSK','ENG','ZRX','XEM','ZEC','XZC','PPT','ARK',
'BTS','GAS','GTO','HSR','FUN',
'LRC','BLZ','KMD','KNC','AE','BCD','SYS',
'VIA','RLC',
'STORJ','PIVX','BNT','MANA'
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
    'XMR': 'Monero',
    'ETC': 'Ethereum Classic',
    'OMG': 'OmiseGO',
    'WAN': 'Wanchain',
    'QTUM': 'Qtum',
    'ELF': 'aelf',
    'AION': 'Aion',
    'LINK': 'ChainLink',
    'SNT': 'Status',
    'WAVES': 'Waves',
    'BTG': 'Bitcoin Gold',
    'STRAT': 'Stratis',
    'NULS': 'Nuls',
    'SALT': 'SALT',
    'STEEM': 'Steem',
    'DGD': 'DigixDAO',
    'ZIL': 'Zilliqa',
    'WTC': 'Waltonchain',
    'DASH': 'Dash',
    'LSK': 'Lisk',
    'ENG': 'Enigma',
    'ZRX': '0x',
    'XEM': 'NEM',
    'ZEC': 'Zcash',
    'XZC': 'ZCoin',
    'PPT': 'Populous',
    'ARK': 'Ark',
    'BTS': 'BitShares',
    'GAS': 'Gas',
    'GTO': 'Gifto',
    'HSR': 'Hshare',
    'FUN': 'FunFair',
    'LRC': 'Loopring',
    'BLZ': 'Bluzelle',
    'KMD': 'Komodo',
    'KNC': 'Kyber Network',
    'AE': 'Aeternity',
    'BCD': 'Bitcoin Diamond',
    'SYS': 'Syscoin',
    'VIA': 'Viacoin',
    'RLC': 'iExec',
    'STORJ': 'Storj',
    'PIVX': 'PIVX',
    'BNT': 'Bancor',
    'MANA': 'Decentraland'
    }



while(True):
    nowDate = datetime.now()
    UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()


    start_date = datetime(nowDate.year, nowDate.month, nowDate.day, 0, 0)

    sentiments = pd.DataFrame()

    for coin in coins:
        print(coin)
        searchQuery = coin_dict[coin]+' '+coin
        # get Twitter sentiments
        os.system('python ./GetOldTweets/Exporter.py --querysearch "%s" --since %s --output ./data/%s_today_tweets.csv'%(searchQuery,start_date.strftime("%Y-%m-%d"),coin))
        twData = pd.read_csv('/home/arash/BitPredict/data/%s_today_tweets.csv'%coin)
        os.system('rm /home/arash/BitPredict/data/%s_today_tweets.csv'%coin)

        twData['sentiment'] = np.zeros(len(twData))
        for index, row in twData.iterrows():
            # sentiment analysis
            try:
            	analysis = TextBlob(row['text'])
            	twData.ix[index, 'sentiment'] = analysis.sentiment.polarity

            except:
            	twData.ix[index, 'sentiment'] = 0.0
            	print('Got invalid row!')

        	# convert to utc time
            local_datetime = datetime.strptime(row['date'], "%Y-%m-%d %H:%M")
            result_utc_datetime = local_datetime + UTC_OFFSET_TIMEDELTA
            result_utc_str = result_utc_datetime.strftime("%Y-%m-%d %H:%M")

        	# round down to the nearest hour
            twData.ix[index, 'date'] = result_utc_str[:-2]+'00:00'

            # group and average over hourly dates
            sub_twData = twData.groupby('date').mean()[['sentiment']]

            if coin == 'BTC':
                sentiments = sub_twData
                sentiments = sentiments.rename(columns={'sentiment': 'BTC'})
            else:
                sentiments[coin] = sub_twData['sentiment']
    sentiments = sentiments.fillna(0)

    # get average of sentimets over hours of today
    avgSentiments = pd.DataFrame(sentiments.mean(axis=0).sort_values(ascending=False))
    d2g.upload(avgSentiments, spreadsheet, 'avgSentiments')
    print(avgSentiments.head())

    # get hourly prices of today sorted based on last hour
    sentiments = sentiments.transpose()
    sentiments = pd.DataFrame(sentiments.sort_values(sentiments.columns[-1], ascending=False))
    d2g.upload(sentiments, spreadsheet, 'sentiments')
    sentiments.to_csv('/home/arash/BitPredict/data/today_sentiments.csv')
    print(sentiments.head())
