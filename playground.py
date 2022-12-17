import pandas as pd
import matplotlib.pyplot as plt

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",  # Commented for testing reasons - Takes long to compute. Uncomment agian!
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

rolling_normalize = ["boll_ub", "boll_lb"]
factor_normalize = {"rsi_30": 0.01, "cci_30":0.01}
subtract_close = ["close_30_sma", "close_60_sma"]

df = pd.read_pickle("tmp_TARSHME")


def rolling_norm(data, window):

    mean = data.rolling(window, min_periods=1).mean()
    std = data.rolling(window, min_periods=1).std()

    return ((data - mean) / std)


WINDOW = 100



for ind in INDICATORS:

    fig, axs = plt.subplots(2, 1)
    plt.suptitle('{}'.format(ind))
    axs[0].plot(df["close"], label="close")

    if ind in rolling_normalize:
        df[ind] = rolling_norm(df[ind], WINDOW)

    if ind in factor_normalize:
        df[ind] = df[ind] * factor_normalize[ind]

    if ind == "dx_30":
        df[ind] *= 0.01
        df[ind] -= 0.5


    if ind in subtract_close:
        df[ind] -= df["close"]
        df[ind] = rolling_norm(df[ind], WINDOW)

    axs[1].plot(df[ind], label='{}'.format(ind))
    axs[1].plot((df["close"] / df["close"].max()) * df[ind].max(), label="CloseRescaled")
    # axs[1].plot(rolling_norm(df["close"], WINDOW), label="CloseNormed")
    axs[1].legend()



plt.show()

