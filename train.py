"""

Part 1. Problem Definition

This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.

The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:

* Action: The action space describes the allowed actions that the agent interacts with the
environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively

* Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
values at state s′ and s, respectively

* State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
our trading agent observes many different features to better learn in an interactive environment.

* Environment: Dow 30 consituents


The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
"""
# import libraries
from __future__ import annotations

import datetime
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.conftest import axis

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.environment.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.environment.env_stock_trading.env_stocktrading_multiday import StockTradingEnvMd
from finrl.data.preprocessor.preprocessors import data_split
from finrl.data.dataset import DatasetFactory
from finrl.data.data_processor import DataProcessor
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import plot_actions
from finrl.sb3_trainer import SB3Trainer
import finrl.config as config
import finrl.config_tickers as config_tickers


matplotlib.use('TkAgg')


# sys.path.append("../FinRL-Library")

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format="[%(levelname)s] %(filename)s %(funcName)s %(message)s",
    handlers=[
        logging.FileHandler("debug.log", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


if config.TRAIN_NEW_AGENT == True or config.RETRAIN_AGENT == True:
    assert config.TRAIN_NEW_AGENT != config.RETRAIN_AGENT, "Both Training and Retraining are active. Choose one!"

# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

check_and_make_directories(config.DIRS)

# Create Dataset and log information
ticker_list_name = config.TICKERS
ticker = config_tickers.TICKERS[config.TICKERS]
df = DatasetFactory(ticker, ticker_list_name).create_dataset()

train = data_split(df, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
# train = DataProcessor.normalize_data(train, config.INDICATORS, window=20)
test = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
# test = DataProcessor.normalize_data(test, config.INDICATORS, window=20)
trade = data_split(df, config.TRADE_START_DATE, config.TRADE_END_DATE)
# trade = DataProcessor.normalize_data(trade, config.INDICATORS, window=20)

logger.info("Num samples for Train: {} | Test: {} | Trade: {}".format(len(train), len(test), len(trade)))
logger.info("Train data interval: {} - {}".format(train.head(1).date.values[0], train.tail(1).date.values[0]))
logger.info("Test data interval: {} - {}".format(test.head(1).date.values[0], test.tail(1).date.values[0]))
logger.info("Trade data interval: {} - {}".format(trade.head(1).date.values[0], trade.tail(1).date.values[0]))
logger.info("Using indicators: {}".format(config.INDICATORS))
logger.info("Ticker contained in final DataFrame: {}".format(df.tic.unique()))
remaining_nans = train[train.isnull().any(axis=1)]
if not remaining_nans.empty:
    logger.error("Remaining NaNs in DataFrame: {}".format(remaining_nans))
    exit()

logger.error("NOMAILIZATION CURRENTLY BRAKES CODE IN ENV FOR MULTISTOCK - THE RESET INDEX IS NOT GOOD")
# train = DataProcessor.normalize_data(train, config.INDICATORS, config.NORMALIZATION_WINDOW)
# test = DataProcessor.normalize_data(test, config.INDICATORS, config.NORMALIZATION_WINDOW)
# trade = DataProcessor.normalize_data(trade, config.INDICATORS, config.NORMALIZATION_WINDOW)
# for i, ds in enumerate([train, test, trade]):
#     ds = ds.dropna(axis=0, how='all')
#     remaining_nans = ds[ds.isnull().any(axis=1)]
#     if not remaining_nans.empty:
#         logger.error("Remaining NaNs in DataFrame {}: {} UU".format(i, remaining_nans))
#         exit()

# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a
# includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action
# can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of
# shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares
# of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy
# is defined on a Gaussian distribution, which needs to be normalized and symmetric.
# """

stock_dimension = len(train.tic.unique())
# State: Current Cash, Current amount of stock?, ?
state_space = 1 + 3 * stock_dimension + len(config.INDICATORS) * stock_dimension
logger.info(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": config.ENV_HMAX,
    "initial_amount": config.ENV_INIT_AMNT,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": config.ENV_REWARD_SCALE,
}

# Turbulence indicators for trading - WHAT IS THIS DOING? ToDo
data_risk_indicator = df[(df.date < config.TRADE_END_DATE) & (df.date >= config.TRAIN_START_DATE)]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=["date"])
if config.USE_VIX:
    insample_risk_indicator.vix.describe()
    insample_risk_indicator.vix.quantile(0.996)
if config.USE_TURBULENCE:
    insample_risk_indicator.turbulence.describe()
    insample_risk_indicator.turbulence.quantile(0.996)
    t_thresh = 70
else:
    t_thresh = None

e_train_gym = StockTradingEnv(df=train, idle_threshold=15, make_plots=True, **env_kwargs)
e_test_gym = StockTradingEnv(df=test, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=t_thresh, risk_indicator_col="vix", make_plots=True, **env_kwargs)

## Environment for Training
env_train, _ = e_train_gym.get_sb_env()
## Environment for Testing
env_test, _ = e_test_gym.get_sb_env()

# Train model
model = SB3Trainer(e_train_gym, e_test_gym).train()

# Trade
df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)

logger.info("Account Value Shape: {}".format(df_account_value.shape))
logger.info("Account Value Tail: {}".format(df_account_value.tail()))
logger.info("Actions Head: {}".format(df_actions.head()))

plot_actions(trade, df_actions)


logger.info("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

# baseline stats
logger.info("==============Get Baseline Stats===========")
baseline_df = DatasetFactory(config_tickers.BASELINE_TICKER, "BASELINE").create_dataset(preprocess=False)
stats = backtest_stats(baseline_df, value_col_name="close")

logger.info("==============Compare to DJIA===========")
backtest_plot(
    df_account_value,
    baseline_ticker="^DJI",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)
plt.show()