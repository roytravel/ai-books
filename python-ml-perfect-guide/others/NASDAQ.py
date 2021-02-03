from matplotlib import pyplot as plt
import pandas_datareader.data as web
import pandas as pd

symbol = "NASDAQCOM"
data = pd.DataFrame()
data[symbol] = web.DataReader(symbol, data_source="fred", start="2000-01-01", end="2020-01-01")[symbol]
data = data.dropna()
data.plot(legend=True)
plt.xlabel("year")
plt.title("INDEXNASDAQ")
plt.show()
