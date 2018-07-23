#!/usr/bin/env python3
# -*- coding: utf-8. -*-
"""
Created on Tue Jun  5 20:27:35 2018

@author: arash
"""
# import os
# import time
# import numpy as np
import pandas as pd
from datetime import datetime
# from textblob import TextBlob
from pytrends.request import TrendReq
from binance.client import Client

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
    'MANA': 'Decentraland',
    'BCC': 'BitConnect'
}

# UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()
pytrends = TrendReq(hl='en-US', tz=0)

api_key = 'e0LdDWHVXTAdEAOPFT7r3Kgy4WX0iuyIsfKZRLFHStp2rnsTyddoYRBLHFfAzzi3'
api_secret = 'Ps0tL8vZJyUGFdDl6C5FbYnroSi4j0kOEYa2OlLGu5vZLx6NVyc1EsgJdlgZOApR'
client = Client(api_key, api_secret)


start_date = datetime(2018, 4, 1, 0, 0)
end_date = datetime.now()


# coins = [
#     'BTC', 'TRX', 'XVG',
#     'MCO', 'ETH', 'XRP', 'XLM', 'ADA', 'NEO',
#     'EOS', 'ICX', 'BNB', 'ONT', 'NANO', 'IOTA', 'LTC', 'VEN', 'XMR',
#     'ETC', 'OMG', 'WAN', 'QTUM', 'ELF', 'AION',
#     'LINK', 'SNT', 'WAVES', 'BTG', 'STRAT', 'NULS',
#     'SALT', 'STEEM', 'DGD', 'ZIL', 'WTC', 'DASH',
#     'LSK', 'ENG', 'ZRX', 'XEM', 'ZEC', 'XZC', 'PPT', 'ARK',
#     'BTS', 'GAS', 'GTO', 'HSR', 'FUN',
#     'LRC', 'BLZ', 'KMD', 'KNC', 'AE', 'BCD', 'SYS',
#     'VIA', 'RLC',
#     'STORJ', 'PIVX', 'BNT', 'MANA'
# ]

# youngers:'EOS', 'TRX', 'ONT', 'XRP',
# 'ADA', 'IOTA', 'ETC', 'ICX', 'VEN', 'XLM',
coins = ['BTC', 'ETH', 'BNB', 'BCC', 'LTC', 'NEO', 'QTUM']


for coin in coins:
    print(coin)

    # the the historical data of the coins and add to the dataframe
    # if coin == 'BTC':
    #     coinpair = "BTCUSDT"
    # else:
    #     coinpair = '{}BTC'.format(coin)

    coinpair = '{}USDT'.format(coin)
    start_date_str = start_date.strftime("%d %B, %Y")
    end_date_str = end_date.strftime("%d %B, %Y")
    interval = Client.KLINE_INTERVAL_1HOUR
    klines = client.get_historical_klines(
        coinpair, interval, start_date_str, end_date_str)
    df = pd.DataFrame(klines)
    df.columns = ['time', 'open', 'high', 'low', 'close',
                  'volume', 'close time', 'quote asset volume',
                  'number of trades', 'taker buy base asset volume',
                  'taker buy quote asset volume', 'ignore']
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df.index = df['date']

    # the Google trends historical interests
    searchQuery = coin_dict[coin] + ' ' + coin
    kw_list = [searchQuery]
    trendsData = pytrends.get_historical_interest(kw_list,
                                                  year_start=start_date.year,
                                                  month_start=start_date.month,
                                                  day_start=start_date.day,
                                                  hour_start=start_date.hour,
                                                  year_end=end_date.year,
                                                  month_end=end_date.month,
                                                  day_end=end_date.day,
                                                  hour_end=end_date.hour,
                                                  cat=0, geo='',
                                                  gprop='', sleep=0)

    # add to the final dataframe
    df = df.join(trendsData)

    # try:
    #     twData = pd.read_csv(
    #         '/home/arash/BitPredict/data/%s_tweets.csv' % coin)
    # except:
    #     try:
    #         os.system('rm /home/arash/BitPredict/data/%s_tweets.csv' % coin)
    #     except:
    #         print('File does not exist!')
    #     # get Twitter sentiments
    #     os.system('python ./GetOldTweets/Exporter.py --querysearch "%s"
    # --since %s --until %s --output ./data/%s_tweets.csv
    # --maxtweets 100000000 --toptweets' %
    # (searchQuery, start_date.strftime("%Y-%m-%d"),
    # end_date.strftime("%Y-%m-%d"), coin))
    #     twData = pd.read_csv(
    #         '/home/arash/BitPredict/data/%s_tweets.csv' % coin)
    #
    # twData['sentiment'] = np.zeros(len(twData))
    # for index, row in twData.iterrows():
    #
    #     # sentiment analysis
    #     try:
    #         analysis = TextBlob(row['text'])
    #         twData.ix[index, 'sentiment'] = analysis.sentiment.polarity
    #         # convert to utc time
    #         local_datetime = datetime.strptime(
    #             row['date'], "%Y-%m-%d %H:%M")
    #         result_utc_datetime = local_datetime + UTC_OFFSET_TIMEDELTA
    #         result_utc_str = result_utc_datetime.strftime("%Y-%m-%d %H:%M")
    #
    #         # round down to the nearest hour
    #         twData.ix[index, 'date'] = result_utc_str[:-2] + '00:00'
    #
    #     except:
    #         twData.ix[index, 'sentiment'] = 0.0
    #         print('Got invalid row!')
    #
    # # group and average over hourly dates
    # sub_twData = twData.groupby('date').mean()[['sentiment']]
    #
    # # add to the final dataframe
    # df = df.join(sub_twData)

    df = df[['open', 'high', 'low', 'close', 'volume',
             'number of trades', searchQuery]]
    df = df.fillna(0.0)
    length = len(df)
    print(length)
    df.to_csv('/home/arash/BitPredict/data/%s.csv' % coin)
