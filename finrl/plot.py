from __future__ import annotations

from copy import deepcopy
import traceback
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries

from finrl import config
from finrl.data.preprocessor.yahoodownloader import YahooDownloader

import logging
logger = logging.getLogger(__name__)


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def backtest_plot(
    account_value,
    baseline_start=config.TRADE_START_DATE,
    baseline_end=config.TRADE_END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )


def get_baseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
        selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

        tic_plot = df_trade[
            (df_trade["tic"] == df_trx_temp.name)
            & (df_trade["date"].isin(df_trx.index))
        ]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(
            tic_plot,
            "^",
            markersize=10,
            color="m",
            label="buying signal",
            markevery=buying_signal,
        )
        plt.plot(
            tic_plot,
            "v",
            markersize=10,
            color="k",
            label="selling signal",
            markevery=selling_signal,
        )
        plt.title(
            f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])}"
        )
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()


def plot_signals_on_chart(gb, stock, actions, gs, num_plot=0):

    stock_hist = gb.get_group(stock)
    stock_hist.date = pd.to_datetime(stock_hist.date)
    stock_hist.index = stock_hist.date
    stock_hist.index = pd.to_datetime(stock_hist.index)

    merge = pd.merge(stock_hist, actions, how='left', left_index=True, right_index=True)[["date", "close", "Action"]]
    merge = merge[merge['Action'].notna()].values

    max_plot_size = 100
    min_plot_size = 10

    try:

        buy_actions = np.argwhere(merge[:, -1] > 0).reshape((-1,))
        if len(buy_actions)>0:
            buy_signals = merge[:, :2][buy_actions]
            buy_amounts = merge[:, 2][buy_actions].flatten()
            if not isinstance(buy_amounts[0], float):
                buy_amounts = np.array([a[0] for a in buy_amounts])
            buy_amounts = buy_amounts.astype(np.float32)

        else:
            buy_amounts = [0]
            buy_signals = None

        sell_actions = np.argwhere(merge[:, -1] < 0).reshape((-1,))
        if len(sell_actions)>0:
            sell_signals = merge[:, :2][sell_actions]
            sell_amounts = merge[:, 2][sell_actions].flatten()
            if not isinstance(sell_amounts[0], float):
                sell_amounts = np.array([a[0] for a in sell_amounts])
            sell_amounts = sell_amounts.astype(np.float32)

        else:
            sell_amounts = [0]
            sell_signals = None

        min_action = np.min([np.min(sell_amounts), np.min(buy_amounts), 0])
        max_axtion = np.max([np.max(sell_amounts), np.max(buy_amounts), 1])
        buy_sizes = np.interp(buy_amounts, (min_action, max_axtion), (min_plot_size, max_plot_size))
        sell_sizes = np.interp(sell_amounts, (min_action, max_axtion), (min_plot_size, max_plot_size))

        ax = plt.subplot(gs[num_plot])
        ax.plot(merge[:, 0], merge[:, 1])
        if buy_signals is not None:
            ax.scatter(buy_signals[:, 0], buy_signals[:, 1], s=buy_sizes, alpha=0.5, c='g')

        if sell_signals is not None:
            ax.scatter(sell_signals[:, 0], sell_signals[:, 1], s=sell_sizes, alpha=0.5, c='r')

        ax.set_title("{}".format(stock))

    except Exception as e:
        print("PLOT FAIL")
        print(e, traceback.format_exc())
        print("M: ", merge)
        print("M: ", merge.shape)

        print("B: ", buy_actions)
        print("BA: ", buy_amounts)
        print("BA t: ", type(buy_amounts))
        print("BA t[0]: ", type(buy_amounts[0]))
        print("BS: ", buy_signals)
        print("BSize: ", buy_sizes)
        print("S: ", sell_actions)
        print("S: ", sell_amounts)
        print("SS: ", sell_signals)
        print("SSize: ", sell_sizes)

        exit()


def plot_actions(trade, df_actions):

    stock_ticker = trade['tic'].unique()
    gb = trade.groupby(['tic'])

    num_rows = int(np.round(np.sqrt(len(stock_ticker))))
    num_cols = int(np.ceil(np.sqrt(len(stock_ticker))))

    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(num_rows, num_cols)  # height_ratios=[5, 1]

    if len(stock_ticker) == 1:
        stock = stock_ticker[0]
        actions = df_actions

        actions.columns = ["date", "Action"]
        actions["date"] = pd.to_datetime(actions["date"])
        actions.index = actions["date"]
        actions.drop(['date'], axis=1, inplace=True)

        plot_signals_on_chart(gb, stock, actions, gs, 0)

    else:
        for i, stock in enumerate(stock_ticker):

            actions = df_actions[stock].to_frame()
            actions.columns = ["Action"]
            actions.index = pd.to_datetime(actions.index)

            plot_signals_on_chart(gb, stock, actions, gs, i)

        # total = np.cumsum(merge[:, -1], dtype=float)
        # invested = merge[:, 1][buy_actions] * merge[:, 2][buy_actions]
        # invested_sum = np.sum(invested)
        # return_of_sales = merge[:, 1][sell_actions] * np.abs(merge[:, 2][sell_actions])
        # return_of_sales_sum = np.sum(return_of_sales)
        # remaining_value = merge[-1, 1] * total[-2]
        # performance = remaining_value + return_of_sales_sum - invested_sum

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    # # Now we can save it to a numpy array.
    img_data_perstock = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data_perstock = img_data_perstock.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_data_perstock

def plot_states(df_states):

    df_states.index = pd.to_datetime(df_states.index)

    # weights_per_stock
    stocks = [c.split("_")[-1] for c in df_states.columns if "Shares" in c]
    weight_per_asset = {}
    weight_cash = (df_states.cash / df_states.TotalValue).values
    for s in stocks:
        weight_per_asset[s] = ((df_states["Price_{}".format(s)] * df_states["Shares_{}".format(s)]) / df_states.TotalValue).values

    fig, axs = plt.subplots(3,1, figsize=(15,10))
    axs[0].plot(df_states.index, df_states['TotalValue'].values)
    axs[0].set_title("TotalValue")
    # axs[0].legend()

    axs[1].plot(df_states.index, weight_cash, label='cash')
    axs[1].plot(df_states.index, df_states.col_12.values, label='NormedPrice')
    for w in weight_per_asset:
        axs[1].plot(df_states.index, weight_per_asset[w], label='{}'.format(w))
    axs[1].set_title("Weight per asset")
    axs[1].legend()

    axs[2].plot(df_states.index, df_states["BuyPrice_{}".format(stocks[0])], label="BuyPrice_{}".format(stocks[0]))
    axs[2].plot(df_states.index, df_states["Price_{}".format(stocks[0])], label="Price_{}".format(stocks[0]))
    axs[2].set_title("Prices")
    axs[2].legend()

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    # # Now we can save it to a numpy array.
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return img_data


# fig = plt.figure()
# # for stock, values in stock_values.items():
# #     plt.plot(values, label='{}'.format(stock))
# #     plt.text(len(values)-2, values[-2], '{}'.format(stock))
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# #           fancybox=True, shadow=True, ncol=5)
# # plt.tight_layout()
# fig.canvas.draw()
# img_data_all = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# img_data_all = img_data_all.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# #
# # if not plot:
# #     plt.close('all')
# return img_data_perstock, img_data_all



