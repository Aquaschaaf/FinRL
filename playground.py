import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("/home/matthias/Projects/FinRL/datasets/raw/1d/DOW_30_2014-01-06_2022-11-01")

print(df.head(10))

x = df["datetime"].values
c = df["close"].values
print(x)

df.set_index("date")
df.index = pd.to_datetime(df.datetime)



# df.plot()
df["close"].plot()

plt.figure()
plt.plot(x, c)

plt.figure()
plt.plot(c)

plt.show()