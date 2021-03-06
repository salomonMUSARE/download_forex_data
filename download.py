# import pandas as pd
# import time
# from collections import deque
# from iqoptionapi.stable_api import IQ_Option
# Iq=IQ_Option("umuneke.44@gmail.com","Mybusiness@me1")
# Iq.connect()#connect to iqoption
# goal="BTCUSD"

# def getdata(rangee):
#     Iq=IQ_Option("umuneke.44@gmail.com","Mybusiness@me1")
#     Iq.connect()#connect to iqoption
#     goal="BTCUSD"
#     end_from_time=time.time()
#     ANS=[]
#     for i in range(rangee):
#         print(i)

#         data=Iq.get_candles("BTCUSD", 3600, 1000, end_from_time)
#         ANS =data+ANS
#         end_from_time=int(data[0]["from"])-1
#     df = pd.DataFrame(ANS)
#     del df['at']
#     df.columns=['id', 'Date', 'to','Open', 'Close',  'Low', 'High', 'Volume']
#     del df["to"]
#     del df["id"]
#     # print(df)
#     return df

# df = getdata(40)
# df=df[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
# print((df.head()))
# df.to_csv("1h_btcusd.csv")

import bitfinex
import datetime
import time
import pandas as pd

# Define query parameters
pair = 'BTCUSD' # Currency pair of interest
TIMEFRAME = '15m'#,'4h','1h','15m','1m'
TIMEFRAME_S = 15*60 # seconds in TIMEFRAME

# Define the start date
t_start = datetime.datetime(2019, 10, 12, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

# Define the end date
t_stop = datetime.datetime(2021, 10, 11, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

def fetch_data(start, stop, symbol, interval, TIMEFRAME_S):
    limit = 1000    # We want the maximum of 1000 data points
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    hour = TIMEFRAME_S * 1000
    step = hour * limit
    data = []

    total_steps = (stop-start)/hour
    tracker = 0
    while total_steps > 0:
        tracker +=1
        if total_steps < limit: # recalculating ending steps
            step = total_steps * hour

        end = start + step
        print(tracker)
        # print(end)
        data += api_v2.candles(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
        # print(pd.to_datetime(start, unit='ms'), pd.to_datetime(end, unit='ms'), "steps left:", total_steps)
        start = start + step +1
        total_steps -= limit
        time.sleep(1.5)
        if tracker == 8:
            break
    return data
count = 0


print(count)

result = fetch_data(t_start, t_stop, pair, TIMEFRAME, TIMEFRAME_S)
names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
df = pd.DataFrame(result, columns=names)
df.drop_duplicates(inplace=True)
# df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df_name = f"{pair}_{TIMEFRAME}_new.csv"
df.to_csv(df_name)

df = pd.read_csv(df_name)

for i in range(len(df) -2):
  
  this = df.loc[i,"Date"]
  thiss = df.loc[i + 1,"Date"]
  diff = this - thiss
  if thiss - this != TIMEFRAME_S*1000:
    count += 1
print(count)