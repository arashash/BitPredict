import binance



binance_coins = [
'TRX','XVG','NCASH'
,'MCO','ETH','XRP','XLM','ADA','GRS','NEO'
,'EOS','ICX','BNB','BCC','STORM','BAT','ONT','NANO','IOTA','LTC','VEN','XMR'
,'ETC','IOST','OMG','SUB','WAN','NEBL','QTUM','MTL','ELF','GVT','AION'
,'CLOAK'
,'QLC','LINK','SNT','WAVES','BTG','ENJ','BQX','EDO','STRAT','POA','NULS','TRIG'
,'SALT','STEEM','LEND','VIBE','BCPT','POWR','DGD','ZIL','CMT','WTC','DASH','POE'
,'LSK','LUN','ENG','ZRX','XEM','ADX','WPR','ARN','ZEC','XZC','PPT','ARK','INS'
,'CND','RCN','AMB','DLT','VIB','OST','BTS','GAS','BRD','DNT','GTO','HSR','FUN'
,'CHAT','NAV','LRC','TNB','QSP','REQ','BLZ','KMD','APPC','KNC','AE','BCD','SYS'
,'RPX','SNGLS','MDA','WABI','FUEL','TNT','VIA','MTH','GXS','EVX','RLC','CDT'
,'AST','WINGS','YOYO','STORJ','PIVX','SNM','BNT','ICN','RDN','OAX','MANA','MOD'
]

spreads = {}

while(True):
    ticks = binance.tickers()
    for coin in binance_coins:
        tick = ticks[coin+'BTC']
        bid = float(tick['bid'])
        ask = float(tick['ask'])

        spread = (ask/bid - 1)*100
        spreads[coin] = spread

    maxCoin = max(spreads, key=spreads.get)
    maxSpread = spreads[maxCoin]

    print('%s : %.2f'%(maxCoin, maxSpread))
